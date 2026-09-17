"""Queue semantics that the scheduler and the tests rely on."""

import asyncio
from datetime import datetime, timedelta, timezone


from contracts import (
    Availability,
    Decision,
    EnergyEstimate,
    EnergyState,
    Job,
    JobResult,
    JobStatus,
)
from jobs.queue import MemoryCache, MockQueue

NOW = datetime(2026, 9, 17, 6, 0, tzinfo=timezone.utc)


def queue() -> MockQueue:
    return MockQueue(clock=lambda: NOW)


def test_dedupe_key_returns_the_existing_job() -> None:
    store = queue()

    async def main():
        first = await store.enqueue(Job(workload="weather_briefing", dedupe_key="daily:2026-09-17"))
        second = await store.enqueue(Job(workload="weather_briefing", dedupe_key="daily:2026-09-17"))
        third = await store.enqueue(Job(workload="weather_briefing"))
        return first, second, third

    first, second, third = asyncio.run(main())
    assert first.id == second.id
    assert third.id != first.id


def test_only_one_job_runs_at_a_time() -> None:
    store = queue()

    async def main():
        a = await store.enqueue(Job(workload="weather_briefing"))
        b = await store.enqueue(Job(workload="weather_briefing"))
        claimed_a = await store.claim(a.id)
        claimed_b = await store.claim(b.id)
        return claimed_a, claimed_b, await store.claim(a.id)

    claimed_a, claimed_b, again = asyncio.run(main())
    assert claimed_a is not None and claimed_a.status == JobStatus.RUNNING
    assert claimed_b is None
    assert again is None


def test_pending_order_uses_priority_deadline_and_age() -> None:
    store = queue()

    async def main():
        low = await store.enqueue(Job(workload="a", priority=20, created_at=NOW,
                                      scheduled_at=NOW))
        urgent = await store.enqueue(Job(workload="b", priority=80, deadline=NOW + timedelta(hours=2),
                                         created_at=NOW, scheduled_at=NOW))
        critical = await store.enqueue(Job(workload="c", priority=100, created_at=NOW,
                                           scheduled_at=NOW))
        very_urgent = await store.enqueue(Job(workload="d", priority=80, deadline=NOW,
                                              created_at=NOW, scheduled_at=NOW))
        return await store.pending(), (low, urgent, critical, very_urgent)

    pending, (low, urgent, critical, very_urgent) = asyncio.run(main())
    assert [job.id for job in pending] == [critical.id, very_urgent.id, urgent.id, low.id]


def test_defer_and_cancel_only_touch_waiting_jobs() -> None:
    store = queue()

    async def main():
        job = await store.enqueue(Job(workload="a"))
        await store.defer(job.id)
        status_after_defer = store.jobs[job.id].status
        cancelled = await store.cancel(job.id)
        running = await store.enqueue(Job(workload="b"))
        claimed = await store.claim(running.id)
        refused = await store.cancel(running.id)
        return status_after_defer, cancelled, claimed, refused

    status_after_defer, cancelled, claimed, refused = asyncio.run(main())
    assert status_after_defer == JobStatus.WAITING_FOR_ENERGY
    assert cancelled is True
    assert claimed is not None
    assert refused is False


def test_finish_records_outcome_estimate_and_energy_sample() -> None:
    store = queue()

    async def main():
        job = await store.enqueue(Job(workload="a"))
        claimed = await store.claim(job.id)
        estimate = EnergyEstimate(estimated_wh=0.5, runtime_seconds=60, average_power_w=30)
        await store.finish(claimed, JobResult(briefing="text"), estimate)
        await store.record_decision(
            Decision(job_id=job.id, decision="RUN",
                     energy_state=EnergyState(timestamp=NOW, availability=Availability.FRESH),
                     reason="surplus")
        )
        return store, await store.snapshot()

    store, snapshot = asyncio.run(main())
    stored = next(iter(store.jobs.values()))
    assert stored.status == JobStatus.COMPLETED
    assert stored.actual_estimated_energy_wh == 0.5
    assert snapshot["jobs"]["recent"][0]["status"] == "COMPLETED"
    assert snapshot["latest_briefing"]["text"] == "text"
    assert snapshot["decisions"][0]["decision"] == "RUN"
    assert snapshot["today"]["jobs_completed"] == 1
    assert snapshot["today"]["estimated_inference_wh"] == 0.5


def test_a_repeated_finish_does_not_reopen_a_terminal_job() -> None:
    store = queue()

    async def main():
        job = await store.enqueue(Job(workload="a"))
        claimed = await store.claim(job.id)
        estimate = EnergyEstimate(estimated_wh=0.5)
        await store.finish(claimed, JobResult(), estimate)
        await store.finish(claimed, None, estimate, error="late failure")
        return store.jobs[job.id]

    stored = asyncio.run(main())
    assert stored.status == JobStatus.COMPLETED
    assert stored.error is None


def test_the_energy_read_model_keeps_samples_in_order() -> None:
    store = queue()

    async def main():
        for soc in (50, 60, 70):
            await store.save_energy(
                EnergyState(timestamp=NOW, availability=Availability.FRESH, battery_soc=soc)
            )
        return await store.snapshot()

    snapshot = asyncio.run(main())
    assert [item["battery_soc"] for item in snapshot["energy"]["samples"]] == [50, 60, 70]
    assert snapshot["energy"]["latest"]["battery_soc"] == 70


def test_memory_cache_returns_copies() -> None:
    cache = MemoryCache()

    async def main():
        await cache.put("k", {"value": 1})
        entry = await cache.get("k")
        entry["value"] = 2
        return await cache.get("k"), await cache.get("missing")

    stored, missing = asyncio.run(main())
    assert stored == {"value": 1}
    assert missing is None


def test_finish_accepts_a_failed_job_without_a_result() -> None:
    store = queue()

    async def main():
        job = await store.enqueue(Job(workload="a"))
        claimed = await store.claim(job.id)
        await store.finish(claimed, None, EnergyEstimate(reason="no measurement"), error="boom")
        return store.jobs[job.id]

    stored = asyncio.run(main())
    assert stored.status == JobStatus.FAILED
    assert stored.error == "boom"
    assert stored.actual_estimated_energy_wh is None
