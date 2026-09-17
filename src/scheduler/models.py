"""Scheduler vocabulary and shared defaults."""
from contracts import Decision, EnergyState, Job, JobStatus

CRITICAL_PRIORITY = 100
HIGH_PRIORITY = 80
LOW_PRIORITY = 20

RUN = "RUN"
WAIT = "WAIT"
DEFER = "DEFER"

__all__ = [
    "CRITICAL_PRIORITY",
    "DEFER",
    "HIGH_PRIORITY",
    "LOW_PRIORITY",
    "RUN",
    "WAIT",
    "Decision",
    "EnergyState",
    "Job",
    "JobStatus",
]
