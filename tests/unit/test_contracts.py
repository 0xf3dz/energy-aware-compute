"""Derived values on the shared contracts."""

from contracts import EnergyEstimate


def test_energy_per_token_divides_watt_hours_by_token_count() -> None:
    # 0.5 Wh over 100 tokens is 18 J per token.
    estimate = EnergyEstimate(estimated_wh=0.5)
    assert estimate.joules_per_generated_token(100) == 18.0


def test_energy_per_token_is_undefined_without_both_inputs() -> None:
    assert EnergyEstimate(estimated_wh=None).joules_per_generated_token(100) is None
    assert EnergyEstimate(estimated_wh=0.5).joules_per_generated_token(0) is None
    assert EnergyEstimate(estimated_wh=0.5).joules_per_generated_token(None) is None


def test_energy_per_token_matches_a_measured_power_ratio() -> None:
    """An integrated estimate over a known window agrees with power over rate."""
    # 40 W across 2.5 s is 100 J. At 20.4 tok/s that window holds 51 tokens.
    estimate = EnergyEstimate(estimated_wh=100 / 3600, average_power_w=40,
                              runtime_seconds=2.5)
    joules = estimate.joules_per_generated_token(51)
    assert joules is not None
    assert abs(joules - 1.96) < 0.01
