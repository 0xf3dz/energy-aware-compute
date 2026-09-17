"""PostgreSQL implementation of the queue, cache, and dashboard read model.

One transaction persists a finished job with all of its evidence: inference
metrics, the normalized forecast, the briefing text, and the energy estimate.
Nothing outside this module speaks SQL.
"""

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from contracts import (
    Decision,
    EnergyEstimate,
    EnergyState,
    Job,
    JobResult,
    JobStatus,
)
from db.migrate import CLAIM_LOCK, OWNERSHIP_LOCK, migrate

logger = logging.getLogger(__name__)

# UTC day boundary, used for the "today" totals.
TODAY = "(date_trunc('day', now() at time zone 'utc') at time zone 'utc')"

JOB_COLUMNS = (
    "id, workload, status, priority, created_at, scheduled_at, started_at, completed_at, "
    'deadline, "deferrable", estimated_energy_wh, actual_estimated_energy_wh, payload, result, '
    "error, dedupe_key"
)

PENDING_ORDER = "ORDER BY priority DESC, deadline ASC NULLS LAST, created_at ASC, id ASC"


class PostgresQueue:
    """Durable queue, provider cache, and dashboard read model."""

    def __init__(
        self,
        dsn: str,
        *,
        pool: AsyncConnectionPool | None = None,
        migrate_on_open: bool = True,
    ) -> None:
        self.dsn = dsn
        self.migrate_on_open = migrate_on_open
        self._owns_pool = pool is None
        self.pool = pool or AsyncConnectionPool(
            dsn, min_size=1, max_size=4, open=False, kwargs={"row_factory": dict_row}
        )

    async def open(self) -> None:
        await self.pool.open(wait=True, timeout=30)
        if self.migrate_on_open:
            async with self.pool.connection() as connection:
                await migrate(connection)

    async def close(self) -> None:
        if self._owns_pool:
            await self.pool.close()

    # ---- Queue -----------------------------------------------------------
    async def enqueue(self, job: Job) -> Job:
        """Insert a job. A repeated dedupe key returns the existing job."""
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"INSERT INTO jobs ({JOB_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, %s, %s, %s, NULL, %s, NULL, NULL, %s) "
                "ON CONFLICT (dedupe_key) DO NOTHING RETURNING " + JOB_COLUMNS,
                (
                    job.id, job.workload, job.status.value, job.priority, job.created_at,
                    job.scheduled_at, job.deadline, job.deferrable, job.estimated_energy_wh,
                    Jsonb(job.payload), job.dedupe_key,
                ),
            )
            row = await cursor.fetchone()
            if row is not None:
                return _job(row)
            cursor = await connection.execute(
                f"SELECT {JOB_COLUMNS} FROM jobs WHERE dedupe_key = %s", (job.dedupe_key,)
            )
            existing = await cursor.fetchone()
        if existing is None:
            raise RuntimeError("Job could not be enqueued and no duplicate exists")
        return _job(existing)

    async def pending(self) -> list[Job]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {JOB_COLUMNS} FROM jobs "
                "WHERE status IN ('QUEUED', 'WAITING_FOR_ENERGY') " + PENDING_ORDER
            )
            return [_job(row) for row in await cursor.fetchall()]

    async def claim(self, job_id: str) -> Job | None:
        """Claim one job. The advisory lock keeps exactly one job running."""
        async with self.pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (CLAIM_LOCK,))
            cursor = await connection.execute(
                "UPDATE jobs SET status = 'RUNNING', started_at = CURRENT_TIMESTAMP "
                "WHERE id = %s AND status IN ('QUEUED', 'WAITING_FOR_ENERGY') "
                "AND NOT EXISTS (SELECT 1 FROM jobs WHERE status = 'RUNNING') "
                f"RETURNING {JOB_COLUMNS}",
                (job_id,),
            )
            row = await cursor.fetchone()
        return _job(row) if row is not None else None

    async def defer(self, job_id: str) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "UPDATE jobs SET status = 'WAITING_FOR_ENERGY' "
                "WHERE id = %s AND status = 'QUEUED'",
                (job_id,),
            )

    async def cancel(self, job_id: str) -> bool:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE jobs SET status = 'CANCELLED', completed_at = CURRENT_TIMESTAMP "
                "WHERE id = %s AND status IN ('QUEUED', 'WAITING_FOR_ENERGY')",
                (job_id,),
            )
            return cursor.rowcount == 1

    async def save_energy(self, state: EnergyState) -> None:
        await self._sample(None, "energy_state", state.model_dump(mode="json"))

    async def finish(
        self,
        job: Job,
        result: JobResult | None,
        estimate: EnergyEstimate,
        error: str | None = None,
    ) -> None:
        """Persist the outcome and every piece of evidence in one transaction."""
        status = JobStatus.FAILED if error else JobStatus.COMPLETED
        async with self.pool.connection() as connection, connection.transaction():
            cursor = await connection.execute(
                "UPDATE jobs SET status = %s, completed_at = CURRENT_TIMESTAMP, "
                "result = %s, error = %s, actual_estimated_energy_wh = %s "
                "WHERE id = %s AND status = 'RUNNING' RETURNING id",
                (
                    status.value,
                    Jsonb(result.model_dump(mode="json")) if result is not None else None,
                    error,
                    estimate.estimated_wh,
                    job.id,
                ),
            )
            if await cursor.fetchone() is None:
                logger.warning("Job %s was not RUNNING when it finished", job.id)
            if result is not None:
                for metrics in result.inference_metrics:
                    await connection.execute(
                        "INSERT INTO inference_metrics (job_id, recorded_at, metrics) "
                        "VALUES (%s, %s, %s)",
                        (job.id, metrics.timestamp, Jsonb(metrics.model_dump(mode="json"))),
                    )
                if result.forecast is not None:
                    await connection.execute(
                        "INSERT INTO weather_forecasts (job_id, recorded_at, forecast) "
                        "VALUES (%s, %s, %s)",
                        (
                            job.id,
                            result.forecast.timestamp,
                            Jsonb(result.forecast.model_dump(mode="json")),
                        ),
                    )
                if result.briefing:
                    await connection.execute(
                        "INSERT INTO weather_briefings (job_id, recorded_at, text) "
                        "VALUES (%s, CURRENT_TIMESTAMP, %s)",
                        (job.id, result.briefing),
                    )
            await connection.execute(
                "INSERT INTO energy_samples (job_id, source, sample) VALUES (%s, 'compute', %s)",
                (job.id, Jsonb(estimate.model_dump(mode="json"))),
            )

    async def _sample(self, job_id: str | None, source: str, payload: dict[str, Any]) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "INSERT INTO energy_samples (job_id, source, sample) VALUES (%s, %s, %s)",
                (job_id, source, Jsonb(payload)),
            )

    async def record_decision(self, decision: Decision) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "INSERT INTO scheduler_decisions (job_id, recorded_at, decision, reason, "
                "energy_state) VALUES (%s, %s, %s, %s, %s)",
                (
                    decision.job_id,
                    decision.timestamp,
                    decision.decision,
                    decision.reason,
                    Jsonb(decision.energy_state.model_dump(mode="json")),
                ),
            )

    async def recover_interrupted(self) -> int:
        """Mark jobs left RUNNING by a previous process as failed."""
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE jobs SET status = 'FAILED', completed_at = CURRENT_TIMESTAMP, "
                "error = 'Interrupted by a scheduler restart; external effects are unknown' "
                "WHERE status = 'RUNNING' RETURNING id"
            )
            rows = await cursor.fetchall()
        if rows:
            logger.warning("Recovered %d interrupted job(s)", len(rows))
        return len(rows)

    @contextlib.asynccontextmanager
    async def ownership(self) -> AsyncIterator[bool]:
        """Acquire the single-scheduler lock for the life of the context."""
        async with await AsyncConnection.connect(self.dsn, row_factory=dict_row) as connection:
            cursor = await connection.execute("SELECT pg_try_advisory_lock(%s) AS held", (OWNERSHIP_LOCK,))
            row = await cursor.fetchone()
            held = bool(row and row["held"])
            try:
                yield held
            finally:
                if held:
                    await connection.execute("SELECT pg_advisory_unlock(%s)", (OWNERSHIP_LOCK,))

    # ---- Cache -----------------------------------------------------------
    async def get(self, key: str) -> dict[str, Any] | None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT value FROM provider_cache WHERE key = %s", (key,)
            )
            row = await cursor.fetchone()
        return dict(row["value"]) if row is not None else None

    async def put(self, key: str, value: dict[str, Any]) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "INSERT INTO provider_cache (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                "updated_at = CURRENT_TIMESTAMP",
                (key, Jsonb(value)),
            )

    # ---- Read model ------------------------------------------------------
    async def snapshot(self) -> dict[str, Any]:
        async with self.pool.connection() as connection:
            return {
                "energy": await self._energy(connection),
                "inference": await self._inference(connection),
                "jobs": {
                    "running": await self._jobs(connection, "status = 'RUNNING'"),
                    "queued": await self._jobs(connection, "status = 'QUEUED'"),
                    "deferred": await self._jobs(connection, "status = 'WAITING_FOR_ENERGY'"),
                    "recent": await self._jobs(
                        connection,
                        "status IN ('COMPLETED', 'FAILED', 'CANCELLED')",
                        "ORDER BY completed_at DESC NULLS LAST",
                    ),
                },
                "decisions": await self._decisions(connection),
                "latest_briefing": await self._briefing(connection),
                "today": await self._today(connection),
            }

    async def _energy(self, connection: AsyncConnection) -> dict[str, Any]:
        cursor = await connection.execute(
            "SELECT recorded_at, sample FROM energy_samples WHERE source = 'energy_state' "
            "ORDER BY recorded_at DESC, id DESC LIMIT 60"
        )
        rows = await cursor.fetchall()
        return {
            "latest": rows[0]["sample"] if rows else None,
            "recorded_at": rows[0]["recorded_at"].isoformat() if rows else None,
            "samples": [row["sample"] for row in reversed(rows)],
        }

    async def _inference(self, connection: AsyncConnection) -> dict[str, Any]:
        cursor = await connection.execute(
            "SELECT recorded_at, metrics FROM inference_metrics ORDER BY recorded_at DESC, id DESC "
            "LIMIT 20"
        )
        rows = await cursor.fetchall()
        return {
            "latest": rows[0]["metrics"] if rows else None,
            "recent": [row["metrics"] for row in rows],
        }

    async def _jobs(
        self, connection: AsyncConnection, where: str, order: str = PENDING_ORDER
    ) -> list[dict[str, Any]]:
        cursor = await connection.execute(
            f"SELECT {JOB_COLUMNS} FROM jobs WHERE {where} {order} LIMIT 20"
        )
        return [_job(row).model_dump(mode="json") for row in await cursor.fetchall()]

    async def _decisions(self, connection: AsyncConnection) -> list[dict[str, Any]]:
        cursor = await connection.execute(
            "SELECT d.job_id, d.recorded_at, d.decision, d.reason, d.energy_state, j.workload "
            "FROM scheduler_decisions d LEFT JOIN jobs j ON j.id = d.job_id "
            "ORDER BY d.recorded_at DESC, d.id DESC LIMIT 20"
        )
        return [
            {
                "job_id": row["job_id"],
                "workload": row["workload"],
                "timestamp": row["recorded_at"].isoformat(),
                "decision": row["decision"],
                "reason": row["reason"],
                "energy_state": row["energy_state"],
            }
            for row in await cursor.fetchall()
        ]

    async def _briefing(self, connection: AsyncConnection) -> dict[str, Any] | None:
        cursor = await connection.execute(
            "SELECT b.job_id, b.recorded_at, b.text, j.workload FROM weather_briefings b "
            "LEFT JOIN jobs j ON j.id = b.job_id ORDER BY b.recorded_at DESC, b.id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "workload": row["workload"],
            "created_at": row["recorded_at"].isoformat(),
            "text": row["text"],
        }

    async def _today(self, connection: AsyncConnection) -> dict[str, Any]:
        cursor = await connection.execute(
            f"SELECT COALESCE(SUM((metrics ->> 'generated_tokens')::bigint), 0) AS tokens, "
            f"COUNT(*) AS requests FROM inference_metrics WHERE recorded_at >= {TODAY}"
        )
        inference = await cursor.fetchone()
        # A job with no measurement must not read as zero Wh. Report the sum of
        # the measurements, or null when no job has a measurement today.
        cursor = await connection.execute(
            f"SELECT COUNT(*) AS completed, COUNT(actual_estimated_energy_wh) AS measured, "
            f"SUM(actual_estimated_energy_wh) AS energy_wh FROM jobs "
            f"WHERE status = 'COMPLETED' AND completed_at >= {TODAY}"
        )
        jobs = await cursor.fetchone()
        cursor = await connection.execute(
            f"SELECT COUNT(DISTINCT job_id) AS deferred FROM scheduler_decisions "
            f"WHERE decision = 'DEFER' AND recorded_at >= {TODAY}"
        )
        deferred = await cursor.fetchone()
        cursor = await connection.execute(
            "SELECT COUNT(*) AS waiting FROM jobs WHERE status = 'WAITING_FOR_ENERGY'"
        )
        waiting = await cursor.fetchone()
        return {
            "generated_tokens": int(inference["tokens"]),
            "inference_requests": int(inference["requests"]),
            "estimated_inference_wh": (
                round(float(jobs["energy_wh"]), 6) if jobs["measured"] else None
            ),
            "jobs_completed": int(jobs["completed"]),
            "jobs_deferred_to_solar": int(deferred["deferred"]),
            "jobs_waiting_for_energy": int(waiting["waiting"]),
        }


def _job(row: dict[str, Any]) -> Job:
    return Job(
        id=row["id"],
        workload=row["workload"],
        status=JobStatus(row["status"]),
        priority=row["priority"],
        created_at=row["created_at"],
        scheduled_at=row["scheduled_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        deadline=row["deadline"],
        deferrable=row["deferrable"],
        estimated_energy_wh=row["estimated_energy_wh"],
        actual_estimated_energy_wh=row["actual_estimated_energy_wh"],
        payload=row["payload"] or {},
        result=row["result"],
        error=row["error"],
        dedupe_key=row["dedupe_key"],
    )
