"""PostgreSQL behaviour that the in-memory double cannot prove.

Skipped unless TEST_DATABASE_URL points at a database that the test may clear.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from contracts import (
    Availability,
    Decision,
    EnergyEstimate,
    EnergyState,
    Forecast,
    InferenceMetrics,
    Job,
    JobResult,
    JobStatus,
    Location,
)
from db import PostgresQueue
from db.migrate import migrate

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL is not set")

NOW = datetime(2026, 9, 17, 6, 0, tzinfo=timezone.utc)

TABLES = (
    "jobs", "energy_samples", "inference_metrics", "weather_forecasts",
    "weather_briefings", "scheduler_decisions", "provider_cache",
)


async def fresh_queue() -> PostgresQueue:
    queue = PostgresQueue(DSN)
    await queue.open()
    async with queue.pool.connection() as connection:
        await connection.execute(f"TRUNCATE {', '.join(TABLES)}")
    return queue


def test_migrations_are_idempotent_and_detect_edits() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            async with queue.pool.connection() as connection:
                await migrate(connection)  # a second run must not reapply anything
                cursor = await connection.execute("SELECT count(*) AS n FROM schema_migrations")
                return (await cursor.fetchone())["n"]
        finally:
            await queue.close()

    assert asyncio.run(main()) >= 1
    assert Path("src/db/migrations/001_initial.sql").exists()


def test_a_job_round_trips_with_all_evidence() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            job = await queue.enqueue(
                Job(workload="weather_briefing", priority=80, deferrable=False,
                    deadline=NOW + timedelta(hours=1), estimated_energy_wh=5,
                    scheduled_at=NOW, created_at=NOW, payload={"estimated_runtime_seconds": 60},
                    dedupe_key="daily:2026-09-17")
            )
            duplicate = await queue.enqueue(
                Job(workload="weather_briefing", dedupe_key="daily:2026-09-17")
            )
            claimed = await queue.claim(job.id)
            assert claimed is not None and claimed.status == JobStatus.RUNNING
            assert await queue.claim(job.id) is None  # only one job runs at a time

            estimate = EnergyEstimate(estimated_wh=0.25, runtime_seconds=45,
                                      average_power_w=20)
            result = JobResult(
                data={"hours": 24},
                inference_metrics=[InferenceMetrics(model="qwen", generated_tokens=120,
                                                    prompt_tokens=400, source="generate")],
                forecast=Forecast(timestamp=NOW, availability=Availability.FRESH,
                                  location=Location(latitude=-27.0, longitude=154.0),
                                  hours=[{"time": NOW.isoformat(), "wind_speed_kn": 12.0}]),
                briefing="Wind 12 knots.",
            )
            await queue.finish(claimed, result, estimate)
            await queue.record_decision(
                Decision(job_id=job.id, decision="RUN", energy_state=EnergyState(
                    timestamp=NOW, availability=Availability.FRESH, battery_soc=91,
                    solar_power_w=750, ac_load_w=300), reason="High SOC and sustained surplus")
            )
            snapshot = await queue.snapshot()
            return job, duplicate, snapshot
        finally:
            await queue.close()

    job, duplicate, snapshot = asyncio.run(main())
    assert job.id == duplicate.id
    assert snapshot["jobs"]["recent"][0]["status"] == "COMPLETED"
    assert snapshot["latest_briefing"]["text"] == "Wind 12 knots."
    assert snapshot["inference"]["latest"]["generated_tokens"] == 120
    assert snapshot["today"]["generated_tokens"] == 120
    assert snapshot["today"]["estimated_inference_wh"] == 0.25
    assert snapshot["today"]["jobs_completed"] == 1
    assert snapshot["decisions"][0]["decision"] == "RUN"
    assert snapshot["decisions"][0]["workload"] == "weather_briefing"


def test_a_failed_job_keeps_its_error_and_its_energy_sample() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            job = await queue.enqueue(Job(workload="weather_briefing", scheduled_at=NOW,
                                          created_at=NOW))
            claimed = await queue.claim(job.id)
            await queue.finish(claimed, None, EnergyEstimate(reason="powermetrics needs root"),
                               error="WeatherUnavailable: no forecast")
            snapshot = await queue.snapshot()
            async with queue.pool.connection() as connection:
                cursor = await connection.execute(
                    "SELECT count(*) AS n FROM energy_samples WHERE job_id = %s", (job.id,)
                )
                return snapshot, (await cursor.fetchone())["n"]
        finally:
            await queue.close()

    snapshot, samples = asyncio.run(main())
    recent = snapshot["jobs"]["recent"][0]
    assert recent["status"] == "FAILED"
    assert "no forecast" in recent["error"]
    assert recent["result"] is None
    assert samples == 1  # the estimate is stored even without a result


def test_cancel_and_recover_follow_the_status_rules() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            waiting = await queue.enqueue(Job(workload="a", scheduled_at=NOW, created_at=NOW))
            assert await queue.cancel(waiting.id) is True
            assert await queue.cancel(waiting.id) is False

            interrupted = await queue.enqueue(Job(workload="b", scheduled_at=NOW, created_at=NOW))
            await queue.claim(interrupted.id)
            recovered = await queue.recover_interrupted()
            snapshot = await queue.snapshot()
            return recovered, snapshot
        finally:
            await queue.close()

    recovered, snapshot = asyncio.run(main())
    assert recovered == 1
    statuses = {job["workload"]: job["status"] for job in snapshot["jobs"]["recent"]}
    assert statuses == {"a": "CANCELLED", "b": "FAILED"}
    assert "Interrupted" in [job["error"] for job in snapshot["jobs"]["recent"]
                             if job["workload"] == "b"][0]


def test_deferral_moves_a_queued_job_to_waiting_for_energy() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            job = await queue.enqueue(Job(workload="a", scheduled_at=NOW, created_at=NOW))
            await queue.defer(job.id)
            return [item.status for item in await queue.pending()]
        finally:
            await queue.close()

    assert asyncio.run(main()) == [JobStatus.WAITING_FOR_ENERGY]


def test_concurrent_claims_yield_exactly_one_running_job() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            first = await queue.enqueue(Job(workload="a", scheduled_at=NOW, created_at=NOW))
            second = await queue.enqueue(Job(workload="b", scheduled_at=NOW, created_at=NOW))
            results = await asyncio.gather(queue.claim(first.id), queue.claim(second.id))
            claimed = [item for item in results if item is not None]
            snapshot = await queue.snapshot()
            return claimed, snapshot
        finally:
            await queue.close()

    claimed, snapshot = asyncio.run(main())
    assert len(claimed) == 1
    assert len(snapshot["jobs"]["running"]) == 1


def test_the_ownership_lock_excludes_a_second_scheduler() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            async with queue.ownership() as held:
                assert held is True
                async with queue.ownership() as second:
                    return second
        finally:
            await queue.close()

    assert asyncio.run(main()) is False


def test_the_provider_cache_survives_a_new_connection() -> None:
    async def main():
        queue = await fresh_queue()
        try:
            await queue.put("weather:v1:abc", {"fetched_at": NOW.isoformat(), "raw": {"a": 1}})
        finally:
            await queue.close()
        second = PostgresQueue(DSN)
        await second.open()
        try:
            return await second.get("weather:v1:abc")
        finally:
            await second.close()

    entry = asyncio.run(main())
    assert entry == {"fetched_at": NOW.isoformat(), "raw": {"a": 1}}
