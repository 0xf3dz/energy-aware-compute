"""Admission policy: safety floor, surplus hysteresis, deadlines, priority."""

from datetime import datetime, timedelta, timezone

import pytest

from contracts import Availability, Job
from scheduler.policies import Policy

NOW = datetime(2026, 9, 17, 6, 0, tzinfo=timezone.utc)


def policy(**kwargs) -> Policy:
    defaults = dict(
        safety_floor_soc=40, high_soc=85, surplus_w=250, surplus_hold_seconds=300,
        max_age_seconds=300, max_sample_gap_seconds=120, max_stale_seconds=3600,
    )
    return Policy(**{**defaults, **kwargs})


def job(**kwargs) -> Job:
    defaults = dict(workload="weather_briefing", priority=20, deferrable=True,
                    scheduled_at=NOW - timedelta(minutes=1))
    return Job(**{**defaults, **kwargs})


def state(soc=91, solar=750, load=300, battery=None, at=None, availability=Availability.FRESH,
          reason=None):
    from contracts import EnergyState

    return EnergyState(
        timestamp=at or NOW, availability=availability, battery_soc=soc, solar_power_w=solar,
        ac_load_w=load, battery_power_w=battery, reason=reason,
    )


def test_surplus_must_hold_before_a_deferrable_job_runs() -> None:
    admission = policy()
    first = admission.decide(job(), state(), NOW, fresh=admission.observe(state(), NOW))
    assert first.decision == "DEFER"
    assert "held for 0 s of the required 300 s" in first.reason

    for seconds in (60, 120, 180, 240):
        sample = state(at=NOW + timedelta(seconds=seconds))
        decision = admission.decide(job(), sample, sample.timestamp,
                                    fresh=admission.observe(sample, sample.timestamp))
        assert decision.decision == "DEFER"
        assert f"held for {seconds} s" in decision.reason

    sample = state(at=NOW + timedelta(seconds=300))
    decision = admission.decide(job(), sample, sample.timestamp,
                                fresh=admission.observe(sample, sample.timestamp))
    assert decision.decision == "RUN"
    assert "held for 300 s" in decision.reason


def test_a_brief_solar_spike_does_not_start_a_job() -> None:
    admission = policy(surplus_hold_seconds=300)
    spike = state(solar=900, at=NOW)
    admission.observe(spike, NOW)
    cloud = state(solar=100, at=NOW + timedelta(seconds=60))
    admission.observe(cloud, cloud.timestamp)
    clear = state(solar=900, at=NOW + timedelta(seconds=120))
    fresh = admission.observe(clear, clear.timestamp)
    decision = admission.decide(job(), clear, clear.timestamp, fresh=fresh)
    assert decision.decision == "DEFER"
    assert "held for 0 s" in decision.reason


def test_a_repeated_sample_cannot_prove_a_continuing_surplus() -> None:
    admission = policy()
    for offset in range(0, 601, 60):
        sample = state(at=NOW + timedelta(seconds=offset))
        admission.observe(sample, sample.timestamp)
    last = state(at=NOW + timedelta(seconds=600))
    # The source stops publishing but the scheduler keeps polling.
    later = NOW + timedelta(seconds=1200)
    fresh = admission.observe(last, later)
    assert fresh is False  # the measurement aged past the freshness limit
    decision = admission.decide(job(), last, later, fresh=fresh)
    assert decision.decision == "DEFER"
    assert "old" in decision.reason


def test_a_gap_in_samples_clears_the_hold() -> None:
    admission = policy()
    opening = state(at=NOW)
    admission.observe(opening, NOW)
    after_gap = state(at=NOW + timedelta(seconds=1000))
    fresh = admission.observe(after_gap, after_gap.timestamp)
    decision = admission.decide(job(), after_gap, after_gap.timestamp, fresh=fresh)
    assert decision.decision == "DEFER"
    assert "held for 0 s" in decision.reason


def test_the_battery_floor_stops_every_job() -> None:
    admission = policy()
    low = state(soc=37)
    fresh = admission.observe(low, NOW)
    for priority, deferrable in ((20, True), (80, False), (100, False)):
        decision = admission.decide(job(priority=priority, deferrable=deferrable), low, NOW,
                                    fresh=fresh)
        assert decision.decision == "DEFER"
        assert "37% is at or below the 40% safety floor" in decision.reason


def test_a_critical_job_runs_without_a_surplus_above_the_floor() -> None:
    admission = policy()
    quiet = state(soc=60, solar=50, load=300)
    fresh = admission.observe(quiet, NOW)
    assert admission.decide(job(priority=100), quiet, NOW, fresh=fresh).decision == "RUN"
    assert admission.decide(job(priority=99, deferrable=False), quiet, NOW,
                            fresh=fresh).decision == "RUN"
    deferred = admission.decide(job(priority=99, deferrable=True), quiet, NOW, fresh=fresh)
    assert deferred.decision == "DEFER"
    assert "below the 85% high-SOC threshold" in deferred.reason


def test_a_deadline_job_runs_before_the_deadline_without_a_surplus() -> None:
    admission = policy()
    quiet = state(soc=60, solar=50, load=300)
    fresh = admission.observe(quiet, NOW)
    far = job(priority=80, deferrable=True, deadline=NOW + timedelta(hours=5),
              payload={"estimated_runtime_seconds": 600})
    assert admission.decide(far, quiet, NOW, fresh=fresh).decision == "DEFER"
    soon = job(priority=80, deferrable=True, deadline=NOW + timedelta(minutes=5),
               payload={"estimated_runtime_seconds": 600})
    decision = admission.decide(soon, quiet, NOW, fresh=fresh)
    assert decision.decision == "RUN"
    assert "Deadline" in decision.reason


def test_a_job_that_is_not_due_yet_waits() -> None:
    admission = policy()
    upcoming = job(scheduled_at=NOW + timedelta(minutes=30))
    clean = state()
    fresh = admission.observe(clean, NOW)
    decision = admission.decide(upcoming, clean, NOW, fresh=fresh)
    assert decision.decision == "WAIT"
    assert "future" in decision.reason


def test_unknown_battery_state_defers_instead_of_assuming_one() -> None:
    admission = policy()
    unknown = state(soc=None, solar=None, load=None, availability=Availability.UNAVAILABLE)
    fresh = admission.observe(unknown, NOW)
    decision = admission.decide(job(priority=100), unknown, NOW, fresh=fresh)
    assert decision.decision == "DEFER"
    assert "unknown" in decision.reason


def test_a_stale_but_recent_measurement_still_guards_the_floor() -> None:
    admission = policy(max_stale_seconds=3600)
    old = state(soc=60, solar=50, load=300, at=NOW - timedelta(minutes=20),
                availability=Availability.STALE)
    fresh = admission.observe(old, NOW)
    assert fresh is False
    # A deadline job may use it; a deferrable job may not.
    assert admission.decide(job(priority=100), old, NOW, fresh=fresh).decision == "RUN"
    deferred = admission.decide(job(deferrable=True, deadline=NOW + timedelta(hours=9)),
                                old, NOW, fresh=fresh)
    assert deferred.decision == "DEFER"
    assert "needs fresh telemetry" in deferred.reason


def test_a_forecast_budget_larger_than_the_workload_allows_the_run() -> None:
    admission = policy()
    sample = state(at=NOW)
    budgeted = state(at=NOW)
    from contracts import EnergyState

    budgeted = EnergyState(**{**budgeted.model_dump(), "solar_forecast_wh": 200.0})
    admission.observe(sample, NOW)
    for offset in range(60, 301, 60):
        step = EnergyState(**{**budgeted.model_dump(),
                              "timestamp": NOW + timedelta(seconds=offset)})
        admission.observe(step, step.timestamp)
    heavy = job(estimated_energy_wh=900)
    decision = admission.decide(heavy, step, step.timestamp, fresh=True)
    assert decision.decision == "DEFER"
    assert "exceeds the 200.0 Wh solar forecast budget" in decision.reason
    light = job(estimated_energy_wh=5)
    assert admission.decide(light, step, step.timestamp, fresh=True).decision == "RUN"


def test_surplus_uses_the_smaller_of_load_and_charging_power() -> None:
    from contracts import EnergyState

    charging = EnergyState(timestamp=NOW, availability=Availability.FRESH, battery_soc=90,
                           solar_power_w=800, ac_load_w=200, battery_power_w=100)
    assert charging.surplus_w == 100


def test_policy_rejects_impossible_configuration() -> None:
    with pytest.raises(ValueError):
        Policy(safety_floor_soc=80, high_soc=50)
    with pytest.raises(ValueError):
        Policy(surplus_w=-1)
    with pytest.raises(ValueError):
        Policy(surplus_hold_seconds=-5)
