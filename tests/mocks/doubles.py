"""Scriptable doubles and fixture loaders for the test suite."""

import json
import plistlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from contracts import Availability, EnergyState, Job, JobResult

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> dict[str, Any]:
    return json.loads(fixture_text(name))


def plist_stream(measurements: list[dict[str, Any]]) -> bytes:
    """Return a NUL-separated stream of powermetrics-style plist documents."""
    return b"\0".join(plistlib.dumps(item) + b"\0" for item in measurements)


def power_document(cpu_mw: float, gpu_mw: float, pid: int = 45323,
                   cpu_seconds: float = 1.0, impact: int = 100) -> dict[str, Any]:
    """One synthetic powermetrics sample. Power values are milliwatts."""
    return {
        "timestamp": "2026-09-17 03:00:00 +0000",
        "processor": {"cpu_power": cpu_mw, "gpu_power": gpu_mw, "combined_power": cpu_mw + gpu_mw},
        "tasks": [
            {
                "pid": pid,
                "name": "llama-server",
                "cputime_ns": int(cpu_seconds * 1e9),
                "energy_impact": impact,
            }
        ],
    }


class FakeEnergyProvider:
    """Yield a scripted sequence of energy states, then repeat the last one."""

    def __init__(self, states: list[EnergyState], *, fail_with: Exception | None = None) -> None:
        self.states = states
        self.fail_with = fail_with
        self.calls = 0

    async def current(self) -> EnergyState:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        index = min(self.calls - 1, len(self.states) - 1)
        return self.states[index]


def energy_state(
    at: datetime,
    *,
    soc: float | None = 91,
    solar_w: float | None = 750,
    load_w: float | None = 300,
    battery_w: float | None = None,
    availability: Availability = Availability.FRESH,
    forecast_wh: float | None = None,
    reason: str | None = None,
) -> EnergyState:
    return EnergyState(
        timestamp=at,
        availability=availability,
        battery_soc=soc,
        solar_power_w=solar_w,
        ac_load_w=load_w,
        battery_power_w=battery_w,
        solar_forecast_wh=forecast_wh,
        reason=reason,
    )


def surplus_series(
    start: datetime,
    *,
    count: int,
    step_seconds: float = 60,
    soc: float = 91,
    solar_w: float = 750,
    load_w: float = 300,
) -> list[EnergyState]:
    return [
        energy_state(start + timedelta(seconds=index * step_seconds), soc=soc,
                     solar_w=solar_w, load_w=load_w)
        for index in range(count)
    ]


class RecordingWorkload:
    def __init__(self, result: JobResult | None = None, *, fail_with: Exception | None = None):
        self.result = result or JobResult(data={"ran": True})
        self.fail_with = fail_with
        self.jobs: list[Job] = []

    async def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        if self.fail_with is not None:
            raise self.fail_with
        return self.result


class FrozenClock:
    def __init__(self, start: datetime | None = None, step_seconds: float = 0) -> None:
        self.now = start or datetime(2026, 9, 17, 6, 0, tzinfo=timezone.utc)
        self.step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, steps: int = 1) -> datetime:
        self.now = self.now + self.step * steps
        return self.now
