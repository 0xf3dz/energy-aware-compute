"""macOS compute energy attribution.

``PowermetricsProvider`` estimates energy from SoC power samples. A future
``SmartPlugProvider`` or ``VictronMeterProvider`` implements the same
:class:`contracts.ComputeEnergyMonitor` interface.
"""
from energy.mac_power.mock import MockEnergyMonitor
from energy.mac_power.powermetrics import PowermetricsProvider, attribute

__all__ = ["MockEnergyMonitor", "PowermetricsProvider", "attribute"]
