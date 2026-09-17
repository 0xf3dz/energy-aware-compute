"""VRM normalization, caching, rate limits, and offline behaviour."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from contracts import Availability, EnergyState
from energy.vrm import MockVRMProvider, VRMProvider, parse_diagnostics
from jobs.queue import MemoryCache
from tests.mocks.doubles import fixture_json

DIAGNOSTICS = fixture_json("vrm_diagnostics.json")
NOW = datetime(2026, 9, 17, 6, 0, tzinfo=UTC)


def test_normalization_reads_the_system_service_only() -> None:
    state = parse_diagnostics(_records(), NOW, freshness_seconds=300)
    assert state.availability == Availability.FRESH
    assert state.battery_soc == 91
    assert state.solar_power_w == 750
    assert state.battery_power_w == 450
    assert state.ac_load_w == 300
    assert state.surplus_w == 450
    assert "AC-coupled PV is excluded" in (state.reason or "")
    # The physical battery record at 12% must not override the system value.
    assert state.battery_soc != 12


def test_unrelated_alarm_instance_does_not_hide_system_measurements() -> None:
    raw = _records()
    raw["records"].append({
        "dbusServiceType": "system", "instance": 276,
        "dbusPath": "/Ac/Alarms/GridLost", "rawValue": 0,
        "timestamp": NOW.timestamp(),
    })
    state = parse_diagnostics(raw, NOW, 300)
    assert state.availability == Availability.FRESH
    assert state.battery_soc == 91
    assert state.ac_load_w == 300


def test_old_measurements_are_marked_stale() -> None:
    raw = json.loads(json.dumps(DIAGNOSTICS))
    for record in raw["records"]:
        record["timestamp"] = (NOW - timedelta(hours=2)).timestamp()
    state = parse_diagnostics(raw, NOW, freshness_seconds=300)
    assert state.availability == Availability.STALE
    assert state.timestamp == NOW - timedelta(hours=2)


def test_empty_and_malformed_responses_are_distinguished() -> None:
    """No usable value is unavailable; a malformed payload is an error."""
    empty = parse_diagnostics({"success": True, "records": []}, NOW, 300)
    assert empty.availability == Availability.UNAVAILABLE
    assert empty.battery_soc is None
    assert "No usable" in (empty.reason or "")
    with pytest.raises(ValueError):
        parse_diagnostics({"success": False}, NOW, 300)
    with pytest.raises(ValueError):
        parse_diagnostics({"success": True, "records": "nonsense"}, NOW, 300)


def test_conflicting_measurements_for_one_path_are_rejected() -> None:
    raw = _records()
    duplicate = dict(raw["records"][0])
    duplicate["rawValue"] = 12
    raw["records"].append(duplicate)
    with pytest.raises(ValueError):
        parse_diagnostics(raw, NOW, 300)


def provider(handler, cache=None, **kwargs) -> VRMProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VRMProvider(
        client, cache or MemoryCache(), "12345", "token-value",
        base_url="https://vrm.test/v2", clock=lambda: NOW, **kwargs
    )


def _records(offset_seconds: float = 0.0) -> dict:
    raw = json.loads(json.dumps(DIAGNOSTICS))
    for record in raw["records"]:
        record["timestamp"] = (NOW + timedelta(seconds=offset_seconds)).timestamp()
    return raw


def test_authenticated_request_uses_the_access_token_header() -> None:
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["header"] = request.headers.get("x-authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_records())

    state = asyncio.run(provider(handler).current())
    assert state.availability == Availability.FRESH
    assert seen["header"] == "Token token-value"
    assert seen["url"].startswith("https://vrm.test/v2/installations/12345/diagnostics")


def test_a_cached_measurement_survives_a_restart_and_an_outage() -> None:
    cache = MemoryCache()

    async def available(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_records())

    first = asyncio.run(provider(available, cache).current())
    assert first.availability == Availability.FRESH

    async def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no internet")

    # A new provider over the same cache represents a process restart.
    later = NOW + timedelta(minutes=30)
    client = httpx.AsyncClient(transport=httpx.MockTransport(unreachable))
    restarted = VRMProvider(client, cache, "12345", "token-value",
                            base_url="https://vrm.test/v2", freshness_seconds=300,
                            clock=lambda: later)
    state = asyncio.run(restarted.current())
    assert state.availability == Availability.STALE
    assert state.battery_soc == 91
    assert "Cached VRM measurement is stale" in (state.reason or "")
    assert "unavailable" in (state.reason or "")


def test_no_cache_and_no_internet_is_unavailable_not_a_crash() -> None:
    async def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no internet")

    state = asyncio.run(provider(unreachable).current())
    assert state.availability == Availability.UNAVAILABLE
    assert state.battery_soc is None
    assert state.reason


def test_rate_limit_delays_the_next_poll() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "120"}, json={"success": False})

    client = provider(handler)
    asyncio.run(client.current())
    asyncio.run(client.current())
    assert calls["n"] == 1  # the second call is inside the retry window


def test_mock_provider_labels_synthetic_values() -> None:
    state = asyncio.run(MockVRMProvider(clock=lambda: NOW).current())
    assert state.battery_soc == 85
    assert state.availability == Availability.FRESH
    assert "Synthetic" in (state.reason or "")
    assert isinstance(state, EnergyState)
