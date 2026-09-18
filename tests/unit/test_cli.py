"""Benchmark report aggregation."""

from api.cli import _summarize


def _row(tokens: int, estimated_wh: float | None) -> dict:
    return {
        "generated_tokens": tokens,
        "runtime_seconds": 5.0,
        "estimated_wh": estimated_wh,
        "measurement_method": "powermetrics",
        "confidence": "estimated",
    }


def test_summary_ratio_counts_only_the_measured_runs() -> None:
    """Energy has a value for one run only, so the ratio must use one token count."""
    summary = _summarize([_row(100, 0.5), _row(900, None)])
    assert summary["requests"] == 2
    assert summary["measured_requests"] == 1
    assert summary["generated_tokens"] == 1000
    assert summary["joules_per_generated_token"] == 18.0
    assert summary["estimated_wh_per_million_tokens"] == 5000.0


def test_summary_ratios_are_null_without_a_measurement() -> None:
    summary = _summarize([_row(100, None)])
    assert summary["measured_requests"] == 0
    assert summary["estimated_wh"] is None
    assert summary["joules_per_generated_token"] is None
    assert summary["estimated_wh_per_million_tokens"] is None
