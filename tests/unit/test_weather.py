"""Forecast normalization, deterministic parsing, and briefing generation."""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from contracts import Availability, Forecast, Job, Location
from inference import MockInferenceProvider
from jobs.queue import MemoryCache
from tests.mocks.doubles import fixture_json
from workloads.weather import (
    MockWeatherProvider,
    OpenMeteoProvider,
    WeatherBriefingWorkload,
    WeatherUnavailable,
    parser,
)
from workloads.weather.provider import synthetic_forecast

WEATHER = fixture_json("openmeteo_forecast.json")
MARINE = fixture_json("openmeteo_marine.json")
NOW = datetime(2026, 9, 17, 3, 0, tzinfo=UTC)
LOCATION = Location(latitude=-27.0, longitude=154.0)


def provider(handler, cache=None, **kwargs) -> OpenMeteoProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenMeteoProvider(client, cache or MemoryCache(), clock=lambda: NOW, **kwargs)


def routed(*, weather=WEATHER, marine=MARINE, fail_marine=False):
    async def handler(request: httpx.Request) -> httpx.Response:
        if "marine" in request.url.path or request.url.host.startswith("marine"):
            if fail_marine:
                return httpx.Response(503, json={"reason": "unavailable"})
            return httpx.Response(200, json=marine)
        return httpx.Response(200, json=weather)

    return handler


def test_normalization_merges_both_endpoints_with_units() -> None:
    forecast = asyncio.run(provider(routed()).forecast(LOCATION))
    assert forecast.availability == Availability.FRESH
    assert forecast.hours
    first = forecast.hours[0]
    index = WEATHER["hourly"]["time"].index(first["time"][:16])
    assert first["wind_speed_kn"] == WEATHER["hourly"]["wind_speed_10m"][index]
    assert first["wave_height_m"] == MARINE["hourly"]["wave_height"][index]
    assert forecast.units["wind_speed_kn"] == "kn"
    assert forecast.units["wave_height_m"] == "m"
    assert forecast.raw["weather"]["hourly"]["time"][0] == WEATHER["hourly"]["time"][0]


def test_forecast_is_trimmed_to_the_horizon_and_the_future() -> None:
    forecast = asyncio.run(provider(routed(), horizon_hours=3).forecast(LOCATION))
    assert 1 <= len(forecast.hours) <= 4
    stamps = [row["time"] for row in forecast.hours]
    assert stamps == sorted(stamps)
    assert all(stamp >= NOW.isoformat()[:13] for stamp in stamps)


def test_a_marine_outage_keeps_the_land_forecast_and_names_the_gap() -> None:
    forecast = asyncio.run(provider(routed(fail_marine=True)).forecast(LOCATION))
    assert forecast.availability == Availability.FRESH
    assert forecast.hours[0]["wind_speed_kn"] is not None
    assert forecast.hours[0].get("wave_height_m") is None
    assert "marine API HTTP 503" in (forecast.reason or "")


def test_a_full_outage_returns_a_stale_cached_forecast() -> None:
    cache = MemoryCache()

    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no internet")

    asyncio.run(provider(routed(), cache, freshness_seconds=1).forecast(LOCATION))
    client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    later = NOW + timedelta(hours=1)
    offline = OpenMeteoProvider(client, cache, clock=lambda: later, freshness_seconds=60)
    forecast = asyncio.run(offline.forecast(LOCATION))
    assert forecast.availability == Availability.STALE
    assert forecast.hours
    assert "Cached forecast" in (forecast.reason or "")


def test_no_cache_and_no_internet_is_unavailable() -> None:
    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no internet")

    forecast = asyncio.run(provider(unavailable).forecast(LOCATION))
    assert forecast.availability == Availability.UNAVAILABLE
    assert forecast.hours == []
    assert forecast.reason


def test_cached_forecasts_are_kept_per_position() -> None:
    cache = MemoryCache()
    nearby = Location(latitude=-27.0001, longitude=154.0)
    far = Location(latitude=-33.8, longitude=151.5)
    asyncio.run(provider(routed(), cache).forecast(nearby))
    asyncio.run(provider(routed(), cache).forecast(far))
    assert len([key for key in cache.entries if key.startswith("weather:v1:")]) == 2


def test_compaction_states_absent_values_instead_of_guessing() -> None:
    forecast = synthetic_forecast(LOCATION, NOW, hours=3)
    for row in forecast.hours:
        row.pop("wave_height_m", None)
    table = parser.compact(forecast)
    assert "time(UTC)" in table
    assert table.count("\n") == 3
    assert "n/a" in table
    assert "wave" in parser.missing_fields(forecast)


def test_warnings_come_from_measured_values_and_thresholds() -> None:
    forecast = synthetic_forecast(LOCATION, NOW, hours=6)
    notes = parser.warnings(forecast, {"gust_kn": 15.0})
    assert any("Gusts" in note and "limit 15" in note for note in notes)
    raised = dict.fromkeys(parser.DEFAULT_THRESHOLDS, 999.0)
    assert parser.warnings(forecast, raised) == []


def test_trends_are_computed_from_the_hours() -> None:
    forecast = synthetic_forecast(LOCATION, NOW, hours=5)
    trends = parser.trends(forecast)
    assert trends["hours"] == 5
    assert trends["wind"]["max"] == max(row["wind_speed_kn"] for row in forecast.hours)
    assert trends["missing_fields"] == []


def briefing_workload(forecast: Forecast | None = None, text: str = "Briefing body") -> tuple:
    weather = MockWeatherProvider(
        forecast if forecast is not None else synthetic_forecast(LOCATION, NOW)
    )
    inference = MockInferenceProvider(text=text)
    return WeatherBriefingWorkload(weather, inference, LOCATION), inference


def test_briefing_stores_the_forecast_the_prompt_and_the_metrics() -> None:
    workload, inference = briefing_workload(text="<think>\n\n</think>\n\nWind 12 knots.")
    result = asyncio.run(workload.run(Job(workload="weather_briefing")))
    assert result.briefing == "Wind 12 knots."
    assert result.forecast is not None
    assert result.data["hours"] == len(result.forecast.hours)
    assert "Forecast position" in result.data["prompt"]
    assert "Never invent" in inference.requests[0].system
    assert result.inference_metrics[0].generated_tokens is not None


def test_prompt_never_receives_the_raw_payload() -> None:
    workload, inference = briefing_workload()
    asyncio.run(workload.run(Job(workload="weather_briefing")))
    prompt = inference.requests[0].prompt
    assert "hourly_units" not in prompt
    assert "generationtime_ms" not in prompt
    assert "Forecast position" in prompt


def test_an_unavailable_forecast_fails_instead_of_inventing_one() -> None:
    weather = MockWeatherProvider(
        Forecast(timestamp=NOW, availability=Availability.UNAVAILABLE, location=LOCATION,
                 reason="No cached forecast")
    )
    workload = WeatherBriefingWorkload(weather, MockInferenceProvider(), LOCATION)
    with pytest.raises(WeatherUnavailable):
        asyncio.run(workload.run(Job(workload="weather_briefing")))


def test_a_missing_position_is_reported_as_an_error() -> None:
    workload = WeatherBriefingWorkload(MockWeatherProvider(), MockInferenceProvider(), None)
    with pytest.raises(WeatherUnavailable):
        asyncio.run(workload.run(Job(workload="weather_briefing")))
    result = asyncio.run(
        workload.run(Job(workload="weather_briefing",
                         payload={"location": {"latitude": 1.0, "longitude": 2.0}}))
    )
    assert result.data["location"] == {"latitude": 1.0, "longitude": 2.0}
