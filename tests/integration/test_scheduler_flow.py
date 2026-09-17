"""The scheduling scenarios from the specification, with no external service."""

import asyncio
from datetime import UTC, datetime, time, timedelta

from contracts import (
    Availability,
    EnergyEstimate,
    EnergyState,
    Job,
    JobStatus,
    Location,
)
from energy.mac_power import MockEnergyMonitor
from inference import MockInferenceProvider
from jobs.queue import MockQueue
from jobs.worker import Worker
from scheduler.engine import Scheduler
from scheduler.models import LOW_PRIORITY
from scheduler.policies import Policy
from tests.mocks.doubles import FakeEnergyProvider, FrozenClock, RecordingWorkload
from workloads.weather import MockWeatherProvider, WeatherBriefingWorkload
from workloads.weather.provider import synthetic_forecast

START = datetime(2026, 9, 17, 6, 0, tzinfo=UTC)
LOCATION = Location(latitude=-27.0, longitude=154.0)


def state(at: datetime, soc=91, solar=750, load=300, availability=Availability.FRESH):
    return EnergyState(
        timestamp=at, availability=availability, battery_soc=soc, solar_power_w=solar,
        ac_load_w=load,
    )


class Harness:
    """Composition for one scenario. The energy monitor uses a seconds clock."""

    def __init__(self, states, *, workloads=None, policy=None, step_seconds=60, monitor=None):
        self.clock = FrozenClock(START, step_seconds=step_seconds)
        self.seconds = {"t": 0.0}

        def seconds_clock() -> float:
            self.seconds["t"] += step_seconds
            return self.seconds["t"]

        self.queue = MockQueue(clock=self.clock)
        self.energy = FakeEnergyProvider(states)
        self.monitor = monitor or MockEnergyMonitor(power_w=25.0, clock=seconds_clock)
        self.recorder = RecordingWorkload()
        self.worker = Worker(self.queue, workloads if workloads is not None
                             else {"weather_briefing": self.recorder}, self.monitor)
        self.scheduler = Scheduler(
            self.queue, self.energy, self.worker, policy or Policy(),
            clock=self.clock, poll_seconds=1,
        )

    async def tick(self) -> object:
        self.clock.advance()
        return await self.scheduler.tick()

    async def ticks(self, count: int) -> list:
        return [await self.tick() for _ in range(count)]


def test_high_surplus_runs_a_deferrable_job_after_the_hold() -> None:
    """battery 91%, solar 750 W, load 300 W, surplus holds."""

    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index)) for index in range(8)])
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=LOW_PRIORITY, deferrable=True,
                estimated_energy_wh=5, scheduled_at=START)
        )
        decisions = await harness.ticks(7)
        return harness, decisions

    harness, decisions = asyncio.run(main())
    assert decisions[6] is None  # nothing is pending once the job has run
    assert [item.decision for item in decisions[:5]] == ["DEFER"] * 5
    assert decisions[5].decision == "RUN"
    assert decisions[5].reason == "Battery 91% and solar surplus 450 W held for 300 s"
    stored = next(iter(harness.queue.jobs.values()))
    assert stored.status == JobStatus.COMPLETED
    assert stored.actual_estimated_energy_wh is not None
    assert harness.recorder.jobs and harness.monitor.stops == 1
    # The decision log explains the deferral as well as the run.
    assert [item.decision for item in harness.queue.decisions].count("DEFER") == 5
    assert harness.queue.decisions[-1].decision == "RUN"


def test_low_battery_keeps_deferrable_compute_queued() -> None:
    """battery 37% → the job stays queued for every tick."""

    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index), soc=37)
                           for index in range(10)])
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=LOW_PRIORITY, deferrable=True,
                scheduled_at=START)
        )
        decisions = await harness.ticks(10)
        return harness, decisions

    harness, decisions = asyncio.run(main())
    assert {item.decision for item in decisions} == {"DEFER"}
    assert "safety floor" in decisions[0].reason
    stored = next(iter(harness.queue.jobs.values()))
    assert stored.status == JobStatus.WAITING_FOR_ENERGY
    assert harness.recorder.jobs == []


def test_a_weather_outage_leaves_the_scheduler_operational() -> None:
    """Weather API unavailable → STALE forecast, and the queue keeps working."""

    async def main():
        stale = synthetic_forecast(LOCATION, START - timedelta(hours=3))
        stale = stale.model_copy(
            update={"availability": Availability.STALE, "reason": "Weather HTTP 503"}
        )
        workload = WeatherBriefingWorkload(
            MockWeatherProvider(stale), MockInferenceProvider(text="Stale briefing"), LOCATION
        )
        harness = Harness(
            [state(START + timedelta(seconds=600 * index)) for index in range(2)],
            workloads={"weather_briefing": workload},
        )
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=80, deferrable=False, scheduled_at=START)
        )
        return harness, await harness.ticks(2)

    harness, decisions = asyncio.run(main())
    assert decisions[0].decision == "RUN"
    stored = next(iter(harness.queue.jobs.values()))
    assert stored.status == JobStatus.COMPLETED
    result = stored.result
    assert result["briefing"] == "Stale briefing"
    assert result["forecast"]["availability"] == "STALE"
    assert result["data"]["forecast_availability"] == "STALE"
    assert "Weather HTTP 503" in result["data"]["forecast_reason"]


def test_a_critical_job_runs_immediately() -> None:
    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index), soc=55, solar=50)
                           for index in range(3)])
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=100, deferrable=False, scheduled_at=START)
        )
        return harness, await harness.tick()

    harness, decision = asyncio.run(main())
    assert decision.decision == "RUN"
    assert "Critical priority" in decision.reason
    assert next(iter(harness.queue.jobs.values())).status == JobStatus.COMPLETED


def test_the_highest_priority_job_runs_first() -> None:
    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index), soc=55, solar=50)
                           for index in range(3)])
        low = await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=10, deferrable=False, scheduled_at=START)
        )
        high = await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=99, deferrable=False, scheduled_at=START)
        )
        decision = await harness.tick()
        return harness, low, high, decision

    harness, low, high, decision = asyncio.run(main())
    assert decision.job_id == high.id
    assert harness.queue.jobs[high.id].status == JobStatus.COMPLETED
    assert harness.queue.jobs[low.id].status == JobStatus.QUEUED


def test_a_daily_schedule_enqueues_once_per_day() -> None:
    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index), soc=95)
                           for index in range(4)])
        harness.scheduler.register_daily(
            "weather-briefing-daily", "weather_briefing", time(7, 0),
            lead_minutes=30, priority=80, deferrable=True, estimated_energy_wh=5,
        )
        before = await harness.tick()  # 06:01, before the 06:30 lead window
        harness.clock.now = START + timedelta(minutes=31)
        opened = await harness.tick()
        repeated = await harness.tick()
        harness.clock.now = START + timedelta(days=1, minutes=31)
        tomorrow = await harness.tick()
        return harness, before, opened, repeated, tomorrow

    harness, before, opened, repeated, tomorrow = asyncio.run(main())
    assert before is None
    assert len(harness.queue.jobs) == 2
    keys = sorted(job.dedupe_key for job in harness.queue.jobs.values())
    assert keys == ["weather-briefing-daily:2026-09-17", "weather-briefing-daily:2026-09-18"]
    assert opened is not None and repeated is not None and tomorrow is not None


def test_a_workload_failure_is_recorded_as_failed_with_an_error() -> None:
    async def main():
        harness = Harness(
            [state(START + timedelta(seconds=60 * index), soc=55, solar=50)
             for index in range(3)],
            workloads={"weather_briefing": RecordingWorkload(fail_with=RuntimeError("no forecast"))},
        )
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=80, deferrable=False, scheduled_at=START)
        )
        await harness.tick()
        return harness

    harness = asyncio.run(main())
    stored = next(iter(harness.queue.jobs.values()))
    assert stored.status == JobStatus.FAILED
    assert "no forecast" in (stored.error or "")
    # The monitor is released even when the workload fails.
    assert harness.monitor.stops == 1


def test_worker_reports_a_missing_workload_instead_of_silently_skipping() -> None:
    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index), soc=55, solar=50)
                           for index in range(3)], workloads={})
        await harness.queue.enqueue(
            Job(workload="unknown_workload", priority=80, deferrable=False, scheduled_at=START)
        )
        await harness.tick()
        return harness

    harness = asyncio.run(main())
    stored = next(iter(harness.queue.jobs.values()))
    assert stored.status == JobStatus.FAILED
    assert "unknown_workload" in (stored.error or "")


def test_the_energy_provider_failure_defers_instead_of_running() -> None:
    async def main():
        harness = Harness([])
        harness.energy.fail_with = RuntimeError("VRM unreachable")
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=100, deferrable=False, scheduled_at=START)
        )
        decision = await harness.tick()
        return harness, decision

    harness, decision = asyncio.run(main())
    assert decision.decision == "DEFER"
    assert "unavailable" in decision.reason
    assert "RuntimeError" in (decision.energy_state.reason or "")
    assert harness.recorder.jobs == []


def test_a_new_scheduler_survives_a_restart_with_the_same_queue() -> None:
    async def main():
        harness = Harness([state(START + timedelta(seconds=60 * index), soc=37)
                          for index in range(3)])
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=80, deferrable=False, scheduled_at=START)
        )
        await harness.tick()
        # A fresh scheduler and policy over the same queue must still defer.
        restarted = Scheduler(harness.queue, harness.energy, harness.worker, Policy(),
                              clock=harness.clock, poll_seconds=1)
        harness.scheduler = restarted
        decision = await harness.tick()
        return harness, decision

    _, decision = asyncio.run(main())
    assert decision.decision == "DEFER"
    assert decision.reason.startswith("Battery state of charge 37%")


def test_a_result_carries_the_estimate_the_monitor_produced() -> None:
    class FixedMonitor:
        async def start(self, job_id):
            pass

        async def stop(self, job_id):
            pass

        async def estimate(self, job_id):
            return EnergyEstimate(estimated_wh=1.5, measurement_method="test",
                                  confidence="estimated", runtime_seconds=10)

    async def main():
        harness = Harness([state(START + timedelta(seconds=60), soc=55, solar=50)],
                          monitor=FixedMonitor())
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=80, deferrable=False, scheduled_at=START,
                estimated_energy_wh=1)
        )
        await harness.tick()
        return harness

    harness = asyncio.run(main())
    assert harness.queue.estimates[
        next(iter(harness.queue.jobs))
    ].measurement_method == "test"
    assert next(iter(harness.queue.jobs.values())).actual_estimated_energy_wh == 1.5


def test_forecast_and_briefing_are_stored_together_for_audit() -> None:
    async def main():
        workload = WeatherBriefingWorkload(
            MockWeatherProvider(synthetic_forecast(LOCATION, START)),
            MockInferenceProvider(text="Audited briefing"),
            LOCATION,
        )
        harness = Harness(
            [state(START + timedelta(seconds=60), soc=95)],
            workloads={"weather_briefing": workload},
        )
        await harness.queue.enqueue(
            Job(workload="weather_briefing", priority=80, deferrable=False, scheduled_at=START,
                estimated_energy_wh=5)
        )
        await harness.tick()
        return await harness.queue.snapshot()

    snapshot = asyncio.run(main())
    assert snapshot["latest_briefing"]["text"] == "Audited briefing"
    assert snapshot["jobs"]["recent"][0]["result"]["forecast"]["hours"]
    assert snapshot["today"]["generated_tokens"] == 256
