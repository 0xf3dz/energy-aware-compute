"""powermetrics stream framing, sample parsing, integration, and attribution."""

import asyncio
from datetime import UTC, datetime

import pytest

from contracts import EnergyEstimate
from energy.mac_power import MockEnergyMonitor, PowermetricsProvider
from energy.mac_power.plist import extract_documents
from energy.mac_power.powermetrics import attribute
from energy.mac_power.samples import PowerSample, parse_power_sample
from tests.mocks.doubles import plist_stream, power_document

LLAMA_PID = 45323


class FakeStream:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def read(self, _size: int) -> bytes:
        payload, self.payload = self.payload, b""
        return payload


class FakeProcess:
    def __init__(self, payload: bytes) -> None:
        self.stdout = FakeStream(payload)
        self.stderr = FakeStream(b"")
        self.returncode = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def send_signal(self, _number: int) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def sample(at: float, power_w: float | None) -> PowerSample:
    return PowerSample(
        at=at, timestamp=datetime.now(UTC), combined_power_w=power_w
    )


def test_stream_framing_keeps_partial_documents() -> None:
    """Arbitrary chunk boundaries must not lose or duplicate a document."""
    stream = plist_stream([power_document(1000, 2000), power_document(3000, 4000)])
    buffer = b""
    documents: list = []
    for start in range(0, len(stream), 37):
        buffer += stream[start : start + 37]
        found, buffer = extract_documents(buffer)
        documents.extend(found)
    assert len(documents) == 2
    assert documents[1]["processor"]["cpu_power"] == 3000
    assert buffer.strip(b"\x00 \t\r\n") == b""


def test_sample_parsing_reports_milliwatts_as_watts_and_finds_the_process() -> None:
    document = power_document(14250.0, 21000.0, pid=LLAMA_PID, cpu_seconds=1.45, impact=312)
    parsed = parse_power_sample(document, at=0.0)
    assert parsed.cpu_power_w == pytest.approx(14.25)
    assert parsed.gpu_power_w == pytest.approx(21.0)
    assert parsed.combined_power_w == pytest.approx(35.25)
    assert parsed.power_w() == pytest.approx(35.25)
    observed = parsed.find(LLAMA_PID, "llama-server")
    assert observed is not None and observed.cpu_seconds == pytest.approx(1.45)
    assert observed.energy_impact == 312


def test_sample_parsing_survives_a_document_without_power_or_tasks() -> None:
    parsed = parse_power_sample({"timestamp": "not a date"}, at=1.0)
    assert parsed.power_w() is None
    assert parsed.processes == []
    assert parsed.timestamp.tzinfo is not None


def test_attribution_subtracts_the_baseline_and_skips_invalid_gaps() -> None:
    samples = [sample(0, 40), sample(1, 40), sample(2, None), sample(5, 40), sample(20, 40)]
    energy_wh, covered, average_w, pairs, gapped = attribute(
        samples, idle_baseline_w=40, max_gap_seconds=3
    )
    # Only the first interval is usable, and the baseline cancels it.
    assert pairs == 1
    assert covered == pytest.approx(1.0)
    assert average_w == pytest.approx(40)
    assert energy_wh == pytest.approx(0)
    # A missing value and an oversized gap are all excluded.
    assert gapped == 3


def test_attribution_reports_no_energy_without_two_valid_samples() -> None:
    assert attribute([], idle_baseline_w=0, max_gap_seconds=3)[0] == 0
    assert attribute([sample(0, 10)], idle_baseline_w=0, max_gap_seconds=3)[1] == 0


def _provider(payload: bytes, **kwargs) -> PowermetricsProvider:
    async def factory(*_args, **_kwargs):
        return FakeProcess(payload)

    return PowermetricsProvider(
        llama_pid=LLAMA_PID,
        process_factory=factory,
        platform="darwin",
        clock=kwargs.pop("clock"),
        **kwargs,
    )


def _run(provider: PowermetricsProvider, seconds: float = 0) -> EnergyEstimate:
    async def main() -> EnergyEstimate:
        await provider.start("job-1")
        await asyncio.sleep(seconds)
        await provider.stop("job-1")
        return await provider.estimate("job-1")

    return asyncio.run(main())


def test_estimate_needs_a_calibrated_baseline() -> None:
    counter = {"t": 0.0}

    def clock() -> float:
        counter["t"] += 1.4
        return counter["t"]

    provider = _provider(plist_stream([power_document(14000, 21000, pid=LLAMA_PID)] * 2),
                         idle_baseline_w=None, clock=clock)
    estimate = _run(provider)
    assert estimate.estimated_wh is None
    assert estimate.measurement_method == "powermetrics"
    assert estimate.confidence == "estimated"
    assert "calibrate" in (estimate.reason or "")
    assert estimate.samples


def test_estimate_requires_the_inference_process_in_the_samples() -> None:
    counter = {"t": 0.0}

    def clock() -> float:
        counter["t"] += 1.4
        return counter["t"]

    provider = _provider(
        plist_stream([power_document(14000, 21000, pid=999_999)] * 2),
        idle_baseline_w=10.0,
        clock=clock,
    )
    estimate = _run(provider)
    assert estimate.estimated_wh is None
    assert "not attributed" in (estimate.reason or "")


def test_estimate_integrates_sampled_power_minus_the_baseline() -> None:
    counter = {"t": 0.0}

    def clock() -> float:
        counter["t"] += 1.4
        return counter["t"]

    provider = _provider(
        plist_stream([power_document(20000, 30000, pid=LLAMA_PID) for _ in range(4)]),
        idle_baseline_w=10.0,
        clock=clock,
    )
    estimate = _run(provider)
    # Four samples at 50 W, three intervals of 1.4 s, minus a 10 W baseline.
    expected = (50.0 - 10.0) * 3 * 1.4 / 3600.0
    assert estimate.estimated_wh == pytest.approx(expected, abs=1e-6)
    assert estimate.runtime_seconds == pytest.approx(4.2)
    assert estimate.average_power_w == pytest.approx(50.0)
    assert "not metered AC consumption" in (estimate.reason or "")


def test_non_macos_platform_reports_a_reason_instead_of_a_number() -> None:
    provider = PowermetricsProvider(llama_pid=1, idle_baseline_w=5.0, platform="linux")
    estimate = _run(provider)
    assert estimate.estimated_wh is None
    assert "macOS" in (estimate.reason or "")


def test_estimate_without_a_measurement_is_null_with_a_reason() -> None:
    provider = PowermetricsProvider(llama_pid=1, idle_baseline_w=5.0)
    estimate = asyncio.run(provider.estimate("missing-job"))
    assert estimate.estimated_wh is None
    assert "No measurement" in (estimate.reason or "")


def test_mock_monitor_reports_simulated_method() -> None:
    counter = {"t": 0.0}

    def clock() -> float:
        counter["t"] += 6.0
        return counter["t"]

    monitor = MockEnergyMonitor(power_w=30.0, clock=clock)

    async def main() -> EnergyEstimate:
        await monitor.start("job")
        await monitor.stop("job")
        return await monitor.estimate("job")

    estimate = asyncio.run(main())
    assert estimate.estimated_wh == pytest.approx(30.0 * 6.0 / 3600.0)
    assert estimate.measurement_method == "mock"
    assert estimate.confidence == "simulated"
    assert monitor.stops == 1
