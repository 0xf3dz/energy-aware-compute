"""Deterministic admission policy. No machine learning, no job-name conditions.

A job is admitted when the policy returns ``RUN``. The three outcomes are:

``WAIT``
    The job is not scheduled yet. Its status stays ``QUEUED``.
``DEFER``
    Energy telemetry, the safety floor, or the solar surplus blocks the job.
    The job status becomes ``WAITING_FOR_ENERGY``.
``RUN``
    The job may start now.

The battery safety floor applies to every job, including critical jobs: an
empty battery stops the boat, not the queue. A critical or deadline job may run
without a solar surplus, and it may use a stale but recent measurement for the
floor check. Deferrable jobs need a fresh measurement, a high state of charge,
and a surplus that held for the configured period.
"""

import math
from datetime import datetime

from contracts import Availability, Decision, EnergyState, Job
from scheduler.models import CRITICAL_PRIORITY, DEFER, RUN, WAIT


class Policy:
    def __init__(
        self,
        *,
        safety_floor_soc: float = 40,
        high_soc: float = 85,
        surplus_w: float = 250,
        surplus_hold_seconds: float = 300,
        max_age_seconds: float = 300,
        max_sample_gap_seconds: float = 120,
        max_stale_seconds: float = 3600,
        max_clock_skew_seconds: float = 5,
        forecast_budget: bool = True,
    ) -> None:
        if not 0 <= safety_floor_soc <= high_soc <= 100:
            raise ValueError("SOC thresholds must satisfy 0 <= floor <= high <= 100")
        durations = (
            surplus_hold_seconds,
            max_age_seconds,
            max_sample_gap_seconds,
            max_stale_seconds,
            max_clock_skew_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in durations):
            raise ValueError("Policy durations must be finite and nonnegative")
        if not math.isfinite(surplus_w) or surplus_w < 0:
            raise ValueError("Surplus threshold must be finite and nonnegative")
        self.safety_floor_soc = safety_floor_soc
        self.high_soc = high_soc
        self.surplus_w = surplus_w
        self.surplus_hold_seconds = surplus_hold_seconds
        self.max_age_seconds = max_age_seconds
        self.max_sample_gap_seconds = max_sample_gap_seconds
        self.max_stale_seconds = max_stale_seconds
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.forecast_budget = forecast_budget
        self._surplus_since: datetime | None = None
        self._last_sample: datetime | None = None

    # ---- hysteresis ------------------------------------------------------
    def reset(self) -> None:
        """Forget the surplus history, for example after a restart or an outage."""
        self._surplus_since = None
        self._last_sample = None

    def observe(self, energy: EnergyState, now: datetime) -> bool:
        """Advance the surplus hold. Return whether the measurement is fresh.

        The hold uses distinct, ordered, timely source samples. A repeated
        cached sample cannot prove that a surplus continued, and a gap in
        samples clears the history. A measurement stamped slightly ahead of the
        local clock is accepted; a larger offset means an untrustworthy clock.
        """
        age = (now - energy.timestamp).total_seconds()
        fresh = (
            energy.availability == Availability.FRESH
            and -self.max_clock_skew_seconds <= age <= self.max_age_seconds
        )
        if not fresh:
            self.reset()
            return False
        if self._last_sample is not None:
            gap = (energy.timestamp - self._last_sample).total_seconds()
            if gap < -self.max_clock_skew_seconds or gap > self.max_sample_gap_seconds:
                self.reset()
        self._last_sample = energy.timestamp
        if not self._qualifying(energy):
            self._surplus_since = None
        elif self._surplus_since is None:
            self._surplus_since = energy.timestamp
        return True

    def _qualifying(self, energy: EnergyState) -> bool:
        surplus = energy.surplus_w
        return (
            energy.battery_soc is not None
            and energy.battery_soc >= self.high_soc
            and surplus is not None
            and surplus >= self.surplus_w
        )

    @property
    def surplus_held_seconds(self) -> float:
        if self._surplus_since is None or self._last_sample is None:
            return 0.0
        return max(0.0, (self._last_sample - self._surplus_since).total_seconds())

    @staticmethod
    def runtime_lead(job: Job) -> float:
        value = job.payload.get("estimated_runtime_seconds", 0)
        try:
            lead = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return lead if math.isfinite(lead) and lead > 0 else 0

    # ---- decision --------------------------------------------------------
    def decide(self, job: Job, energy: EnergyState, now: datetime, *, fresh: bool) -> Decision:
        action, reason = self._admission(job, energy, now, fresh)
        return Decision(
            timestamp=now,
            job_id=job.id,
            decision=action,
            energy_state=energy.model_copy(deep=True),
            reason=reason,
        )

    def _admission(
        self, job: Job, energy: EnergyState, now: datetime, fresh: bool
    ) -> tuple[str, str]:
        if job.scheduled_at > now:
            return WAIT, f"Scheduled start {job.scheduled_at.isoformat()} is in the future"
        soc = energy.battery_soc
        age = max(0.0, (now - energy.timestamp).total_seconds())
        if soc is None:
            return DEFER, f"Battery state of charge is unknown ({_state(energy)})"
        # A stale but recent cached measurement may still guard the safety
        # floor, so that a deadline job survives a VRM outage. Deferrable jobs
        # below require a fresh measurement.
        stale_but_usable = (
            energy.availability == Availability.STALE and age <= self.max_stale_seconds
        )
        if not fresh and not stale_but_usable:
            return DEFER, (
                f"Energy telemetry is {_state(energy)} and {age:.0f} s old; "
                f"the safety floor cannot be evaluated"
            )
        if soc <= self.safety_floor_soc:
            return DEFER, (
                f"Battery state of charge {soc:.0f}% is at or below the "
                f"{self.safety_floor_soc:.0f}% safety floor"
            )
        if job.priority >= CRITICAL_PRIORITY:
            return RUN, f"Critical priority {job.priority}; battery is above the safety floor"
        if not job.deferrable:
            return RUN, "Nondeferrable workload; battery is above the safety floor"
        if job.deadline is not None:
            lead = self.runtime_lead(job)
            remaining = (job.deadline - now).total_seconds()
            if remaining <= lead:
                return RUN, (
                    f"Deadline {job.deadline.isoformat()} requires {lead:.0f} s of runtime "
                    "and the battery is above the safety floor"
                )
        if not fresh:
            return DEFER, (
                f"Deferrable workload needs fresh telemetry; the measurement is "
                f"{_state(energy)} and {age:.0f} s old"
            )
        if soc < self.high_soc:
            return DEFER, (
                f"Battery state of charge {soc:.0f}% is below the "
                f"{self.high_soc:.0f}% high-SOC threshold"
            )
        surplus = energy.surplus_w
        if surplus is None:
            return DEFER, "Solar surplus cannot be computed from the current measurements"
        if surplus < self.surplus_w:
            return DEFER, (
                f"Solar surplus {surplus:.0f} W is below the {self.surplus_w:.0f} W threshold"
            )
        held = self.surplus_held_seconds
        if held < self.surplus_hold_seconds:
            return DEFER, (
                f"Solar surplus {surplus:.0f} W held for {held:.0f} s of the required "
                f"{self.surplus_hold_seconds:.0f} s"
            )
        if (
            self.forecast_budget
            and job.estimated_energy_wh is not None
            and energy.solar_forecast_wh is not None
            and job.estimated_energy_wh > energy.solar_forecast_wh
        ):
            return DEFER, (
                f"Estimated workload energy {job.estimated_energy_wh:.1f} Wh exceeds the "
                f"{energy.solar_forecast_wh:.1f} Wh solar forecast budget"
            )
        return RUN, (
            f"Battery {soc:.0f}% and solar surplus {surplus:.0f} W held for {held:.0f} s"
        )


def _state(energy: EnergyState) -> str:
    return energy.availability.value.lower()
