"""Daily maritime weather briefing. The first pluggable workload."""
from workloads.weather.provider import MockWeatherProvider, OpenMeteoProvider
from workloads.weather.service import WeatherBriefingWorkload, WeatherUnavailable

__all__ = [
    "MockWeatherProvider",
    "OpenMeteoProvider",
    "WeatherBriefingWorkload",
    "WeatherUnavailable",
]
