"""Generic workload dispatch with durable outcomes and unconditional monitor cleanup."""
import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime
import logging

from contracts import ComputeEnergyMonitor, EnergyEstimate, Job, JobResult, Queue, Workload, utcnow

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
        """Execute an already-claimed job; interruption is FAILED (effects may exist)."""
        result: JobResult | None = None
        error: str | None = None
        cancelled = False
        try:
            await self.monitor.start(job.id)
            result = await self.workloads[job.workload].run(job)
            if not isinstance(result, JobResult):
                raise TypeError("Workload.run must return JobResult")
        except asyncio.CancelledError:
            cancelled = True
            error = "Worker interrupted during shutdown; external effects may be incomplete"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Workload failed", extra={"job_id": job.id})

        async def finish() -> None:
            nonlocal error
            try:
                await self.monitor.stop(job.id)
            except Exception as exc:
                detail = f"Monitor cleanup failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {detail}" if error else detail
                logger.exception("Monitor cleanup failed", extra={"job_id": job.id})
            try:
                estimate = await self.monitor.estimate(job.id)
            except Exception as exc:
                detail = f"Energy estimate unavailable: {type(exc).__name__}: {exc}"
                error = f"{error}; {detail}" if error else detail
                estimate = EnergyEstimate(reason=detail)
                logger.exception("Energy estimate failed", extra={"job_id": job.id})
            await self.queue.finish(job, result, estimate, error=error)

        # A shutdown signal must not strand RUNNING rows or a live monitor task.
        finishing = asyncio.create_task(finish())
        while not finishing.done():
            try:
                await asyncio.shield(finishing)
            except asyncio.CancelledError:
                cancelled = True
                if error is None:
                    error = "Worker interrupted during shutdown; external effects may be incomplete"
        finishing.result()
        if cancelled:
            raise asyncio.CancelledError
        return result if error is None else None
