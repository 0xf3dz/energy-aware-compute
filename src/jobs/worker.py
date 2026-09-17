"""Generic workload dispatch with durable outcomes.

Monitor failures never change the fate of a job. A briefing that was produced
is a completed deliverable; missing energy telemetry is recorded as a reason on
the estimate, not as a failure of the work. Only a failing workload, or an
interruption during shutdown, marks a job as failed.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping
from datetime import datetime

from contracts import (
    ComputeEnergyMonitor,
    EnergyEstimate,
    Job,
    JobResult,
    Queue,
    Workload,
    utcnow,
)

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: Queue,
        workloads: Mapping[str, Workload],
        monitor: ComputeEnergyMonitor,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.queue = queue
        self.monitor = monitor
        self.clock = clock
        self.workloads = dict(workloads)

    def register(self, name: str, workload: Workload) -> None:
        """Composition can register dependency-injected or entry-point-loaded plugins."""
        if not name or name in self.workloads:
            raise ValueError(f"Workload registration is empty or already exists: {name!r}")
        self.workloads[name] = workload

    async def run(self, job: Job) -> JobResult | None:
        """Execute an already-claimed job. Interruption is FAILED, effects unknown."""
        result: JobResult | None = None
        error: str | None = None
        notes: list[str] = []
        cancelled = False
        try:
            try:
                await self.monitor.start(job.id)
            except Exception as exc:
                notes.append(_note("Energy measurement could not start", exc))
                logger.exception("Energy monitor failed to start", extra={"job_id": job.id})
            workload = self.workloads.get(job.workload)
            if workload is None:
                raise LookupError(f"No workload is registered for {job.workload!r}")
            result = await workload.run(job)
            if not isinstance(result, JobResult):
                raise TypeError("Workload.run must return JobResult")
        except asyncio.CancelledError:
            cancelled = True
            error = "Worker interrupted during shutdown; external effects may be incomplete"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Workload failed", extra={"job_id": job.id})

        # A shutdown signal must not strand a RUNNING row or a live monitor task.
        finishing = asyncio.create_task(self._finish(job, result, notes))
        while not finishing.done():
            try:
                await asyncio.shield(finishing)
            except asyncio.CancelledError:
                cancelled = True
                if error is None:
                    error = "Worker interrupted during shutdown; external effects may be incomplete"
        estimate = finishing.result()
        if error is not None:
            estimate = estimate.model_copy(
                update={"reason": "; ".join(filter(None, [estimate.reason, error]))}
            )
        await self.queue.finish(job, result, estimate, error=error)
        if cancelled:
            raise asyncio.CancelledError
        return result if error is None else None

    async def _finish(self, job: Job, result: JobResult | None, notes: list[str]) -> EnergyEstimate:
        """Stop the monitor and collect the estimate. Never raises."""
        try:
            await self.monitor.stop(job.id)
        except Exception as exc:
            notes.append(_note("Energy measurement could not stop", exc))
            logger.exception("Energy monitor failed to stop", extra={"job_id": job.id})
        try:
            estimate = await self.monitor.estimate(job.id)
        except Exception as exc:
            notes.append(_note("Energy estimate is unavailable", exc))
            logger.exception("Energy estimate failed", extra={"job_id": job.id})
            estimate = EnergyEstimate(reason="Energy estimate is unavailable")
        if notes:
            estimate = estimate.model_copy(
                update={"reason": "; ".join(filter(None, [estimate.reason, *notes]))}
            )
        return estimate


def _note(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {type(exc).__name__}: {exc}"
