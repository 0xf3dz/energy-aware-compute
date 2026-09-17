"""Victron VRM adapter. Supplies normalized ``EnergyState`` values only."""
from energy.vrm.parser import parse_diagnostics
from energy.vrm.provider import MockVRMProvider, VRMProvider

__all__ = ["MockVRMProvider", "VRMProvider", "parse_diagnostics"]
