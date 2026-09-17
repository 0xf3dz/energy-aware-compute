"""Marine weather provider. Returns a normalized :class:`contracts.Forecast`.

The raw API payload is kept for audit only. Nothing downstream of the parser
reads it, and no raw payload ever reaches the language model.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from contracts import Availability, Cache, Forecast, Location, utcnow

logger = logging.getLogger(__name__)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

WEATHER_FIELDS = (
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "pressure_msl",
    "temperature_2m",
    "cloud_cover",
)
MARINE_FIELDS = (
    "wave_height",
    "wave_direction",
    "wave_period",
    "swell_wave_height",
    "swell_wave_direction",
    "swell_wave_period",
    "sea_surface_temperature",
)

# Normalized field names. A missing value stays None and reaches the model as
# an explicit absence instead of a fabricated number.
NORMALIZED = {
    "wind_speed_10m": "wind_speed_kn",
    "wind_gusts_10m": "wind_gust_kn",
    "wind_direction_10m": "wind_direction_deg",
    "precipitation": "precipitation_mm",
    "pressure_msl": "pressure_hpa",
    "temperature_2m": "air_temperature_c",
    "cloud_cover": "cloud_cover_pct",
    "wave_height": "wave_height_m",
    "wave_direction": "wave_direction_deg",
    "wave_period": "wave_period_s",
    "swell_wave_height": "swell_height_m",
    "swell_wave_direction": "swell_direction_deg",
    "swell_wave_period": "swell_period_s",
    "sea_surface_temperature": "sea_temperature_c",
}


class OpenMeteoProvider:
    """Fetch and normalize Open-Meteo forecast and marine data."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        cache: Cache,
        *,
        poll_interval_seconds: float = 1800,
        freshness_seconds: float = 7200,
        horizon_hours: int = 24,
        weather_url: str = WEATHER_URL,
        marine_url: str = MARINE_URL,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        self.client = client
        self.cache = cache
        self.poll_interval_seconds = poll_interval_seconds
        self.freshness_seconds = freshness_seconds
        self.horizon_hours = horizon_hours
        self.weather_url = weather_url
        self.marine_url = marine_url
        self.clock = clock
        self._locks: dict[str, asyncio.Lock] = {}

    def _key(self, location: Location) -> str:
        rounded = f"{location.latitude:.4f}:{location.longitude:.4f}"
        digest = hashlib.sha256(rounded.encode()).hexdigest()[:16]
        return f"weather:v1:{digest}"

    async def forecast(self, location: Location) -> Forecast:
        key = self._key(location)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            return await self._forecast(key, location)

    async def _forecast(self, key: str, location: Location) -> Forecast:
        now = self.clock()
        entry = await self.cache.get(key) or {}
        next_poll = _parse_stamp(entry.get("next_poll"))
        if next_poll is not None and now < next_poll:
            return _cached_forecast(entry, location, now, self.freshness_seconds)
        next_poll = now + timedelta(seconds=self.poll_interval_seconds)
        try:
            weather, marine = await asyncio.gather(
                self._fetch(self.weather_url, location, WEATHER_FIELDS),
                self._fetch(self.marine_url, location, MARINE_FIELDS),
                return_exceptions=True,
            )
            failures = []
            if isinstance(weather, BaseException):
                failures.append(f"weather API {_name(weather)}")
                weather = None
            if isinstance(marine, BaseException):
                failures.append(f"marine API {_name(marine)}")
                marine = None
            if weather is None and marine is None:
                raise ValueError("both forecast endpoints failed")
            forecast = self._normalize(weather, marine, location, now)
            if failures:
                forecast.reason = "Partial forecast: " + ", ".join(failures)
            entry = {
                "raw": forecast.raw,
                "normalized": forecast.model_dump(mode="json"),
                "fetched_at": now.isoformat(),
                "next_poll": next_poll.isoformat(),
            }
            await self.cache.put(key, entry)
            return forecast
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            reason = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else f"unavailable ({type(exc).__name__})"
            )
            entry = {**entry, "next_poll": next_poll.isoformat(), "error": f"Weather {reason}"}
            await self.cache.put(key, entry)
            return _cached_forecast(entry, location, now, self.freshness_seconds)

    async def _fetch(
        self, url: str, location: Location, fields: tuple[str, ...]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "hourly": ",".join(fields),
            "timezone": "UTC",
            "forecast_days": max(1, min(7, self.horizon_hours // 24 + 1)),
        }
        if url == self.weather_url:
            params["wind_speed_unit"] = "kn"
        response = await self.client.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "hourly" not in payload:
            raise ValueError("forecast response has no hourly data")
        return payload

    def _normalize(
        self,
        weather: dict[str, Any] | None,
        marine: dict[str, Any] | None,
        location: Location,
        now: datetime,
    ) -> Forecast:
        hours: dict[str, dict[str, Any]] = {}
        units: dict[str, str] = {}
        cutoff = now + timedelta(hours=self.horizon_hours)
        earliest = now.replace(minute=0, second=0, microsecond=0)
        for payload in (weather, marine):
            if payload is None:
                continue
            hourly = payload.get("hourly") or {}
            stamps = hourly.get("time") or []
            for field, values in hourly.items():
                if field == "time":
                    continue
                normalized = NORMALIZED.get(field, field)
                units[normalized] = (payload.get("hourly_units") or {}).get(field, "")
                for index, raw in enumerate(values):
                    if index >= len(stamps):
                        break
                    stamp = _parse_stamp(stamps[index])
                    if stamp is None or stamp < earliest or stamp > cutoff:
                        continue
                    hours.setdefault(stamp.isoformat(), {})[normalized] = raw
        ordered = [{"time": key, **hours[key]} for key in sorted(hours)]
        return Forecast(
            timestamp=now,
            availability=Availability.FRESH,
            location=location,
            hours=ordered,
            units=units,
            raw={"weather": weather, "marine": marine},
        )


class MockWeatherProvider:
    """Deterministic forecast for tests and for the offline demonstration."""

    def __init__(
        self,
        forecast: Forecast | None = None,
        *,
        clock: Callable[[], datetime] = utcnow,
        fail_with: Exception | None = None,
    ) -> None:
        self._forecast = forecast
        self.clock = clock
        self.fail_with = fail_with
        self.calls = 0

    async def forecast(self, location: Location) -> Forecast:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        if self._forecast is not None:
            return self._forecast.model_copy(deep=True)
        return synthetic_forecast(location, self.clock())


def synthetic_forecast(location: Location, now: datetime, *, hours: int = 24) -> Forecast:
    """Build a labelled synthetic forecast. Never presented as a live forecast."""
    rows = []
    for index in range(hours):
        stamp = (now + timedelta(hours=index)).replace(minute=0, second=0, microsecond=0)
        rows.append(
            {
                "time": stamp.isoformat(),
                "wind_speed_kn": round(8 + 6 * (index % 5), 1),
                "wind_gust_kn": round(14 + 8 * (index % 4), 1),
                "wind_direction_deg": (index * 15) % 360,
                "precipitation_mm": 0.0 if index % 6 else 1.2,
                "pressure_hpa": round(1012 - index * 0.4, 1),
                "air_temperature_c": round(21 + (index % 7) * 0.5, 1),
                "wave_height_m": round(1 + (index % 3) * 0.4, 2),
                "wave_direction_deg": 120,
                "wave_period_s": 8.5,
                "swell_height_m": 0.8,
                "swell_direction_deg": 130,
                "swell_period_s": 11.0,
            }
        )
    return Forecast(
        timestamp=now,
        availability=Availability.FRESH,
        location=location,
        hours=rows,
        units={"wind_speed_kn": "kn", "wave_height_m": "m"},
        raw={"synthetic": True},
        reason="Synthetic forecast for offline use; not live weather data",
    )


def _parse_stamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp


def _cached_forecast(
    entry: dict[str, Any], location: Location, now: datetime, freshness_seconds: float
) -> Forecast:
    try:
        forecast = Forecast.model_validate(entry["normalized"])
    except (KeyError, ValueError, TypeError):
        return Forecast(
            timestamp=now,
            availability=Availability.UNAVAILABLE,
            location=location,
            reason=entry.get("error") or "No cached forecast is available",
        )
    error = entry.get("error")
    age = (now - forecast.timestamp).total_seconds()
    if error or age > freshness_seconds:
        forecast = forecast.model_copy(
            update={
                "availability": Availability.STALE,
                "reason": "; ".join(
                    filter(
                        None,
                        [
                            forecast.reason,
                            error,
                            f"Cached forecast is {age / 60:.0f} min old" if age > freshness_seconds else None,
                        ],
                    )
                ),
            }
        )
    return forecast


def _name(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def dump_raw(forecast: Forecast) -> str:
    """Serialize the retained raw payload, for an audit trail."""
    return json.dumps(forecast.raw, separators=(",", ":"), sort_keys=True)
