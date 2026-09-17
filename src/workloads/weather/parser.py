"""Deterministic forecast calculations.

Every number that reaches the model is computed here or comes from the API.
The model interprets the numbers; it never produces them.
"""

from collections.abc import Iterable
from typing import Any

from contracts import Forecast

# Warning thresholds. Values are conservative defaults for a sailing vessel.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "gust_kn": 25.0,
    "wind_kn": 20.0,
    "wave_m": 2.0,
    "swell_m": 1.5,
    "rain_mm": 2.0,
    "pressure_drop_hpa": 6.0,
}

MISSING = "n/a"

FIELDS: tuple[tuple[str, str, str], ...] = (
    ("wind_speed_kn", "wind", "{:.0f}"),
    ("wind_gust_kn", "gust", "{:.0f}"),
    ("wind_direction_deg", "dir", "{:.0f}"),
    ("pressure_hpa", "pressure", "{:.0f}"),
    ("precipitation_mm", "rain", "{:.1f}"),
    ("air_temperature_c", "temp", "{:.1f}"),
    ("wave_height_m", "wave", "{:.1f}"),
    ("wave_period_s", "period", "{:.0f}"),
    ("swell_height_m", "swell", "{:.1f}"),
)


def compact(forecast: Forecast, *, max_rows: int = 48) -> str:
    """Render hourly rows as a fixed-width table with explicit gaps."""
    if not forecast.hours:
        return "No forecast hours are available."
    header = "time(UTC)  " + "  ".join(name.rjust(8) for _, name, _ in FIELDS)
    lines = [header]
    for row in forecast.hours[:max_rows]:
        cells = []
        for field, _, fmt in FIELDS:
            value = row.get(field)
            cells.append((fmt.format(value) if isinstance(value, (int, float)) else MISSING).rjust(8))
        lines.append(f"{_stamp(row, forecast)}  " + "  ".join(cells))
    if len(forecast.hours) > max_rows:
        lines.append(f"... {len(forecast.hours) - max_rows} further hour(s) available")
    return "\n".join(lines)


def trends(forecast: Forecast) -> dict[str, Any]:
    """Compute the extremes and the trend the model needs to describe."""
    result: dict[str, Any] = {"hours": len(forecast.hours)}
    for field, label, mode in (
        ("wind_speed_kn", "wind", "max"),
        ("wind_gust_kn", "gust", "max"),
        ("wave_height_m", "wave_height", "max"),
        ("swell_height_m", "swell_height", "max"),
        ("precipitation_mm", "precipitation_total", "sum"),
        ("air_temperature_c", "temperature_range", "range"),
        ("pressure_hpa", "pressure", "range"),
        ("sea_temperature_c", "sea_temperature_range", "range"),
    ):
        result[label] = _aggregate(forecast, field, mode)
    result["wind_direction_shift_deg"] = _direction_shift(forecast)
    result["pressure_drop_6h_hpa"] = _pressure_drop(forecast, 6)
    result["missing_fields"] = missing_fields(forecast)
    return result


def warnings(forecast: Forecast, thresholds: dict[str, float] | None = None) -> list[str]:
    """Return deterministic safety notes with the time and the measured value."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    notes: list[str] = []
    for field, label, limit in (
        ("wind_gust_kn", "Gusts", limits["gust_kn"]),
        ("wind_speed_kn", "Wind", limits["wind_kn"]),
        ("wave_height_m", "Wave height", limits["wave_m"]),
        ("swell_height_m", "Swell height", limits["swell_m"]),
        ("precipitation_mm", "Rain", limits["rain_mm"]),
    ):
        worst = _peak(forecast, field)
        if worst is not None and worst[1] >= limit:
            notes.append(f"{label} {worst[1]:.1f} at {_stamp(worst[0], forecast)} (limit {limit:g})")
    drop = _pressure_drop(forecast, 6)
    if drop is not None and drop >= limits["pressure_drop_hpa"]:
        notes.append(f"Pressure drop {drop:.1f} hPa over 6 h (limit {limits['pressure_drop_hpa']:g})")
    if forecast.availability.value != "FRESH":
        notes.append(f"Forecast data is {forecast.availability.value.lower()}")
    return notes


def missing_fields(forecast: Forecast) -> list[str]:
    if not forecast.hours:
        return ["all fields"]
    absent = []
    for field, label, _ in FIELDS:
        if not any(isinstance(row.get(field), (int, float)) for row in forecast.hours):
            absent.append(label)
    return absent


def _stamp(row: dict[str, Any], forecast: Forecast) -> str:
    value = str(row.get("time", ""))
    return value[11:16] if len(value) >= 16 else value or MISSING


def _values(forecast: Forecast, field: str) -> Iterable[tuple[dict[str, Any], float]]:
    for row in forecast.hours:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            yield row, float(value)


def _peak(forecast: Forecast, field: str) -> tuple[dict[str, Any], float] | None:
    best: tuple[dict[str, Any], float] | None = None
    for row in forecast.hours:
        value = row.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (best is None or value > best[1])
        ):
            best = (row, float(value))
    return best


def _aggregate(forecast: Forecast, field: str, mode: str) -> dict[str, Any] | None:
    pairs = list(_values(forecast, field))
    if not pairs:
        return None
    if mode == "sum":
        return {"total": round(sum(value for _, value in pairs), 2)}
    if mode == "range":
        low = min(pairs, key=lambda item: item[1])
        high = max(pairs, key=lambda item: item[1])
        return {
            "min": round(low[1], 2),
            "min_at": _stamp(low[0], forecast),
            "max": round(high[1], 2),
            "max_at": _stamp(high[0], forecast),
        }
    high = max(pairs, key=lambda item: item[1])
    return {"max": round(high[1], 2), "at": _stamp(high[0], forecast)}


def _direction_shift(forecast: Forecast) -> float | None:
    pairs = [value for _, value in _values(forecast, "wind_direction_deg")]
    if len(pairs) < 2:
        return None
    return round(abs(pairs[-1] - pairs[0]) if abs(pairs[-1] - pairs[0]) <= 180
                 else 360 - abs(pairs[-1] - pairs[0]), 1)


def _pressure_drop(forecast: Forecast, hours: int) -> float | None:
    pairs = list(_values(forecast, "pressure_hpa"))
    if len(pairs) < 2:
        return None
    first = pairs[0][1]
    window = pairs[: hours + 1][-1][1]
    drop = first - window
    return round(drop, 2) if drop > 0 else None
