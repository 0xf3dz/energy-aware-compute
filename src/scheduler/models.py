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
    "Decision",
    "EnergyState",
    "HIGH_PRIORITY",
    "Job",
    "JobStatus",
    "LOW_PRIORITY",
    "RUN",
    "WAIT",
]
