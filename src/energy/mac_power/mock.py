"""Substitute monitor. Reports simulated energy and never claims a measurement."""

import time
from collections.abc import Callable

from contracts import ComputeEnergyMonitor, EnergyEstimate

from energy.models import METHOD_MOCK


class MockEnergyMonitor(ComputeEnergyMonitor):
    def __init__(
        self,
        *,
        power_w: float = 20,
        clock: Callable[[], float] = time.monotonic,
        fail_with: Exception | None = None,
    ) -> None:
        if power_w < 0:
            raise ValueError("power_w must not be negative")
        self.power_w = power_w
        self.clock = clock
        self.fail_with = fail_with
        self._started: dict[str, float] = {}
        self.stops = 0

    async def start(self, job_id: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self._started[job_id] = self.clock()

    async def stop(self, job_id: str) -> None:
        self.stops += 1

    async def estimate(self, job_id: str) -> EnergyEstimate:
        runtime = max(0.0, self.clock() - self._started.get(job_id, self.clock()))
        return EnergyEstimate(
            estimated_wh=round(self.power_w * runtime / 3600.0, 6),
            measurement_method=METHOD_MOCK,
            confidence="simulated",
            runtime_seconds=runtime,
            average_power_w=self.power_w,
            reason="Simulated energy monitor; no power meter was read",
        )
