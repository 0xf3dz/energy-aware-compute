"""In-memory queue and cache with the same semantics as the PostgreSQL store.

Used by unit tests and by the offline demonstration. Status transitions,
dedupe keys, and the single running job rule match ``db.PostgresQueue``.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from contracts import (
    Decision,
    EnergyEstimate,
    EnergyState,
    Job,
    JobResult,
    JobStatus,
    utcnow,
)


class MemoryCache:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.puts = 0

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = self.entries.get(key)
        return None if entry is None else dict(entry)

    async def put(self, key: str, value: dict[str, Any]) -> None:
        self.entries[key] = dict(value)
        self.puts += 1


class MockQueue:
    """Deterministic queue double. No database, no network."""

    def __init__(self, *, clock: Callable[[], Any] = utcnow) -> None:
        self.clock = clock
        self.jobs: dict[str, Job] = {}
        self.decisions: list[Decision] = []
        self.energy_samples: list[dict[str, Any]] = []
        self.estimates: dict[str, EnergyEstimate] = {}
        self.results: dict[str, JobResult] = {}
        self.claim_attempts = 0
        self._lock = asyncio.Lock()

    # ---- Queue -----------------------------------------------------------
    async def enqueue(self, job: Job) -> Job:
        async with self._lock:
            if job.dedupe_key:
                for existing in self.jobs.values():
                    if existing.dedupe_key == job.dedupe_key:
                        return existing.model_copy(deep=True)
            self.jobs[job.id] = job.model_copy(deep=True)
            return job.model_copy(deep=True)

    async def pending(self) -> list[Job]:
        jobs = [job for job in self.jobs.values() if job.status in _PENDING]
        return [job.model_copy(deep=True) for job in _ordered(jobs)]

    async def claim(self, job_id: str) -> Job | None:
        async with self._lock:
            self.claim_attempts += 1
            job = self.jobs.get(job_id)
            if job is None or job.status not in _PENDING:
                return None
            if any(other.status == JobStatus.RUNNING for other in self.jobs.values()):
                return None
            job.status = JobStatus.RUNNING
            job.started_at = self.clock()
            return job.model_copy(deep=True)

    async def defer(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is not None and job.status == JobStatus.QUEUED:
            job.status = JobStatus.WAITING_FOR_ENERGY

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status not in _PENDING:
            return False
        job.status = JobStatus.CANCELLED
        job.completed_at = self.clock()
        return True

    async def save_energy(self, state: EnergyState) -> None:
        self.energy_samples.append(
            {"recorded_at": self.clock().isoformat(), "state": state.model_dump(mode="json")}
        )

    async def finish(
        self,
        job: Job,
        result: JobResult | None,
        estimate: EnergyEstimate,
        error: str | None = None,
    ) -> None:
        stored = self.jobs.get(job.id)
        if stored is not None and stored.status == JobStatus.RUNNING:
            stored.status = JobStatus.FAILED if error else JobStatus.COMPLETED
            stored.completed_at = self.clock()
            stored.error = error
            stored.actual_estimated_energy_wh = estimate.estimated_wh
            if result is not None:
                stored.result = result.model_dump(mode="json")
                self.results[job.id] = result
        self.estimates[job.id] = estimate
        self.energy_samples.append(
            {
                "recorded_at": self.clock().isoformat(),
                "job_id": job.id,
                "state": {"compute": estimate.model_dump(mode="json")},
            }
        )

    async def record_decision(self, decision: Decision) -> None:
        self.decisions.append(decision)

    # ---- Read model ------------------------------------------------------
    async def snapshot(self) -> dict[str, Any]:
        latest = self.energy_samples[-1]["state"] if self.energy_samples else None
        return {
            "energy": {
                "latest": latest,
                "recorded_at": self.energy_samples[-1]["recorded_at"] if self.energy_samples else None,
                "samples": [sample["state"] for sample in self.energy_samples[-60:]],
            },
            "jobs": {
                "running": [_dump(job) for job in _by_status(self.jobs, JobStatus.RUNNING)],
                "queued": [_dump(job) for job in _by_status(self.jobs, JobStatus.QUEUED)],
                "deferred": [_dump(job) for job in _by_status(self.jobs, JobStatus.WAITING_FOR_ENERGY)],
                "recent": [_dump(job) for job in _recent(self.jobs)],
            },
            "decisions": [decision.model_dump(mode="json") for decision in self.decisions[-20:]],
            "inference": _inference(self.jobs),
            "latest_briefing": _latest_briefing(self.jobs),
            "today": _today(self.jobs, self.decisions),
        }


def _ordered(jobs: list[Job]) -> list[Job]:
    return sorted(
        jobs,
        key=lambda job: (
            -job.priority,
            job.deadline is None,
            job.deadline or job.created_at,
            job.created_at,
            job.id,
        ),
    )


def _by_status(jobs: dict[str, Job], status: JobStatus) -> list[Job]:
    return _ordered([job for job in jobs.values() if job.status == status])


def _recent(jobs: dict[str, Job], limit: int = 10) -> list[Job]:
    done = [job for job in jobs.values() if job.completed_at is not None]
    return sorted(done, key=lambda job: job.completed_at, reverse=True)[:limit]


def _dump(job: Job) -> dict[str, Any]:
    return job.model_dump(mode="json")


def _today(jobs: dict[str, Job], decisions: list[Decision]) -> dict[str, Any]:
    completed = [job for job in jobs.values() if job.status == JobStatus.COMPLETED]
    tokens = 0
    requests = 0
    for job in completed:
        for item in (job.result or {}).get("inference_metrics", []):
            tokens += item.get("generated_tokens") or 0
            requests += 1
    return {
        "generated_tokens": tokens,
        "inference_requests": requests,
        "estimated_inference_wh": round(
            sum(job.actual_estimated_energy_wh or 0 for job in completed), 6
        ),
        "jobs_completed": len(completed),
        "jobs_deferred_to_solar": len({item.job_id for item in decisions if item.decision == "DEFER"}),
        "jobs_waiting_for_energy": len(_by_status(jobs, JobStatus.WAITING_FOR_ENERGY)),
    }


def _inference(jobs: dict[str, Job]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for job in _recent(jobs, limit=20):
        for item in (job.result or {}).get("inference_metrics", []):
            metrics.append(item)
    return {"latest": metrics[0] if metrics else None, "recent": metrics[:20]}


def _latest_briefing(jobs: dict[str, Job]) -> dict[str, Any] | None:
    for job in _recent(jobs):
        result = job.result or {}
        if result.get("briefing"):
            return {
                "job_id": job.id,
                "workload": job.workload,
                "created_at": job.completed_at.isoformat() if job.completed_at else None,
                "text": result["briefing"],
            }
    return None


_PENDING = (JobStatus.QUEUED, JobStatus.WAITING_FOR_ENERGY)
