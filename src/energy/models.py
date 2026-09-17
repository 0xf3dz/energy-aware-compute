"""Vessel energy state and the provider interfaces that supply it.

The scheduler consumes :class:`EnergyState` only. It never sees raw VRM JSON and
never knows whether a plug meter, a smart plug, or VRM produced a value.
"""
from contracts import (
    Availability,
    ComputeEnergyMonitor,
    EnergyEstimate,
    EnergyProvider,
    EnergyState,
    utcnow,
)

# Measurement methods a provider may report. "metered" is reserved for a real
# instrument; powermetrics output is always "estimated".
METHOD_POWERMETRICS = "powermetrics"
METHOD_MOCK = "mock"
CONFIDENCE_ESTIMATED = "estimated"


class UnavailableEnergyProvider:
    """Report a missing energy source instead of inventing a state."""

    def __init__(self, reason: str = "No vessel energy source is configured") -> None:
        self.reason = reason

    async def current(self) -> EnergyState:
        return EnergyState(
            timestamp=utcnow(), availability=Availability.UNAVAILABLE, reason=self.reason
        )

__all__ = [
    "Availability",
    "ComputeEnergyMonitor",
    "CONFIDENCE_ESTIMATED",
    "EnergyEstimate",
    "EnergyProvider",
    "EnergyState",
    "METHOD_MOCK",
    "METHOD_POWERMETRICS",
    "UnavailableEnergyProvider",
]
