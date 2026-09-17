"""Scheduler engine: reads energy state, asks the policy, dispatches one job.

The engine holds no provider implementation details. It receives an energy
provider, a queue, a worker, and a policy through the constructor, so a test
can replace each of them independently.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Protocol

from contracts import (
    Availability,
    Decision,
    EnergyProvider,
    EnergyState,
    Job,
    Queue,
    utcnow,
)
from jobs.worker import Worker
from scheduler.models import DEFER, RUN
from scheduler.policies import Policy

logger = logging.getLogger(__name__)


class OwnershipLock(Protocol):
    """Cross-process lock. ``db.PostgresQueue.ownership`` implements it."""

    async def __aenter__(self) -> bool: ...
    async def __aexit__(self, *exc: object) -> bool | None: ...


@dataclass
class DailySchedule:
    """A job that must exist once per UTC day."""

    schedule_id: str
    workload: str
    at: time
    lead_minutes: int = 30
    priority: int = 80
    deferrable: bool = True
    estimated_energy_wh: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    def __init__(
        self,
        queue: Queue,
        energy_provider: EnergyProvider,
        worker: Worker,
        policy: Policy | None = None,
        *,
        clock: Callable[[], datetime] = utcnow,
        poll_seconds: float = 30,
        ownership: OwnershipLock | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.queue = queue
        self.energy_provider = energy_provider
        self.worker = worker
        self.policy = policy or Policy()
        self.clock = clock
        self.poll_seconds = poll_seconds
        self.ownership = ownership
        self.schedules: list[DailySchedule] = []
        self._recorded: dict[str, tuple[str, str]] = {}
        self.tick_count = 0

    def register_daily(
        self,
        schedule_id: str,
        workload: str,
        at: time,
        *,
        lead_minutes: int = 30,
        priority: int = 80,
        deferrable: bool = True,
        estimated_energy_wh: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> DailySchedule:
        """Register a plugin-supplied daily job. Names stay data, not code."""
        if not schedule_id or any(item.schedule_id == schedule_id for item in self.schedules):
            raise ValueError(f"Daily schedule id is empty or already registered: {schedule_id!r}")
        schedule = DailySchedule(
            schedule_id=schedule_id,
            workload=workload,
            at=at,
            lead_minutes=lead_minutes,
            priority=priority,
            deferrable=deferrable,
            estimated_energy_wh=estimated_energy_wh,
            payload=dict(payload or {}),
        )
        self.schedules.append(schedule)
        return schedule

    async def tick(self) -> Decision | None:
        """Evaluate the queue once and run at most one job. Return the decision."""
        self.tick_count += 1
        now = self.clock()
        await self._enqueue_due(now)
        energy = await self._read_energy(now)
        with contextlib.suppress(Exception):
            await self.queue.save_energy(energy)
        fresh = self.policy.observe(energy, now)
        pending = await self.queue.pending()
        blocked: Decision | None = None
        for job in pending:
            decision = self.policy.decide(job, energy, now, fresh=fresh)
            if decision.decision == RUN:
                claimed = await self.queue.claim(job.id)
                if claimed is None:
                    continue
                await self._record(decision)
                await self.worker.run(claimed)
                return decision
            if decision.decision == DEFER:
                await self.queue.defer(job.id)
            if blocked is None:
                blocked = decision
        if blocked is not None:
            await self._record(blocked)
        return blocked

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Poll until the stop event is set. Takes the ownership lock if given."""
        if self.ownership is None:
            await self._loop(stop)
            return
        async with self.ownership as held:
            if not held:
                logger.warning("Another scheduler owns the queue; this instance stays idle")
                return
            await self._loop(stop)

    async def _loop(self, stop: asyncio.Event | None) -> None:
        while stop is None or not stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler tick failed")
            if stop is None:
                await asyncio.sleep(self.poll_seconds)
            else:
                with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), self.poll_seconds)

    async def _read_energy(self, now: datetime) -> EnergyState:
        try:
            return await self.energy_provider.current()
        except Exception as exc:
            logger.warning("Energy provider failed: %s", type(exc).__name__)
            return EnergyState(
                timestamp=now,
                availability=Availability.UNAVAILABLE,
                reason=f"Energy provider failed ({type(exc).__name__})",
            )

    async def _enqueue_due(self, now: datetime) -> None:
        for schedule in self.schedules:
            start = datetime.combine(now.date(), schedule.at, tzinfo=now.tzinfo)
            open_at = start - timedelta(minutes=schedule.lead_minutes)
            if now < open_at:
                continue
            if now - open_at > timedelta(days=1):
                # A schedule missed by more than a day is not replayed.
                continue
            key = f"{schedule.schedule_id}:{start.date().isoformat()}"
            await self.queue.enqueue(
                Job(
                    workload=schedule.workload,
                    priority=schedule.priority,
                    scheduled_at=open_at,
                    deadline=start,
                    deferrable=schedule.deferrable,
                    estimated_energy_wh=schedule.estimated_energy_wh,
                    payload=dict(schedule.payload),
                    dedupe_key=key,
                )
            )

    async def _record(self, decision: Decision) -> None:
        """Store a decision when it differs from the last one for that job."""
        signature = (decision.decision, decision.reason)
        if self._recorded.get(decision.job_id) == signature:
            return
        self._recorded[decision.job_id] = signature
        await self.queue.record_decision(decision)
