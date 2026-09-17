"""Daily briefing workload.

It receives a weather provider and an inference provider, so it holds no
knowledge of Open-Meteo, of llama.cpp, or of the scheduler.
"""

import re
from typing import Any

from contracts import (
    Availability,
    GenerationRequest,
    InferenceProvider,
    Job,
    JobResult,
    Location,
    WeatherProvider,
)
from workloads.weather import parser, prompts

EMPTY_THINK = re.compile(r"<think>\s*</think>\s*", re.IGNORECASE)


class WeatherUnavailable(RuntimeError):
    """No forecast data exists, so no briefing can be produced."""


class WeatherBriefingWorkload:
    def __init__(
        self,
        weather: WeatherProvider,
        inference: InferenceProvider,
        location: Location | None = None,
        *,
        max_tokens: int = 768,
        temperature: float = 0.2,
        thresholds: dict[str, float] | None = None,
        max_hours: int = 48,
    ) -> None:
        self.weather = weather
        self.inference = inference
        self.location = location
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thresholds = thresholds
        self.max_hours = max_hours

    async def run(self, job: Job) -> JobResult:
        location = self._location(job)
        forecast = await self.weather.forecast(location)
        if forecast.availability == Availability.UNAVAILABLE or not forecast.hours:
            raise WeatherUnavailable(
                forecast.reason or "No forecast data is available; the briefing is not generated"
            )
        table = parser.compact(forecast, max_rows=self.max_hours)
        trends = parser.trends(forecast)
        notes = parser.warnings(forecast, self.thresholds)
        missing = trends.get("missing_fields", [])
        prompt = prompts.build_prompt(
            compact_table=table,
            trends=trends,
            warnings=notes,
            missing=missing,
            latitude=location.latitude,
            longitude=location.longitude,
            generated_at=forecast.timestamp,
            availability=forecast.availability.value,
            reason=forecast.reason,
        )
        response = await self.inference.generate(
            GenerationRequest(
                system=prompts.SYSTEM_PROMPT,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        )
        data: dict[str, Any] = {
            "location": location.model_dump(),
            "forecast_availability": forecast.availability.value,
            "forecast_reason": forecast.reason,
            "hours": len(forecast.hours),
            "warnings": notes,
            "trends": trends,
            "prompt": prompt,
        }
        return JobResult(
            data=data,
            inference_metrics=[response.metrics],
            forecast=forecast,
            briefing=_clean(response.text),
        )

    def _location(self, job: Job) -> Location:
        payload = job.payload.get("location")
        if isinstance(payload, dict):
            return Location.model_validate(payload)
        if self.location is not None:
            return self.location
        raise WeatherUnavailable(
            "No forecast position is configured; set LATITUDE and LONGITUDE or pass a payload"
        )


def _clean(text: str) -> str:
    """Remove an empty reasoning block and trailing whitespace."""
    return EMPTY_THINK.sub("", text).strip()
