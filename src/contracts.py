"""Provider-independent application contracts. All times use UTC."""
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Model(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class Availability(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class EnergyState(Model):
    timestamp: AwareDatetime = Field(default_factory=utcnow)
    availability: Availability = Availability.UNAVAILABLE
    battery_soc: float | None = Field(default=None, ge=0, le=100)
    solar_power_w: float | None = Field(default=None, ge=0)
    battery_power_w: float | None = None
    ac_load_w: float | None = Field(default=None, ge=0)
    solar_forecast_wh: float | None = Field(default=None, ge=0)
    reason: str | None = None

    @property
    def surplus_w(self) -> float | None:
        if self.solar_power_w is None or self.ac_load_w is None:
            return None
        # Battery charging power caps the surplus if DC loads consume solar.
        surplus = self.solar_power_w - self.ac_load_w
        if self.battery_power_w is not None:
            surplus = min(surplus, self.battery_power_w)
        return surplus


class EnergyProvider(Protocol):
    async def current(self) -> EnergyState: ...


class EnergyEstimate(Model):
    estimated_wh: float | None = Field(default=None, ge=0)
    measurement_method: str = "powermetrics"
    confidence: str = "estimated"
    runtime_seconds: float = 0
    average_power_w: float | None = None
    samples: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None


class ComputeEnergyMonitor(Protocol):
    async def start(self, job_id: str) -> None: ...
    async def stop(self, job_id: str) -> None: ...
    async def estimate(self, job_id: str) -> EnergyEstimate: ...


class GenerationRequest(Model):
    system: str = "You are a factual assistant."
    prompt: str
    max_tokens: int = Field(default=1024, ge=1, le=16384)
    temperature: float = Field(default=0.2, ge=0, le=2)


class InferenceMetrics(Model):
    timestamp: AwareDatetime = Field(default_factory=utcnow)
    model: str = "unknown"
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    active_requests: float | None = None
    kv_cache_usage: float | None = None
    runtime_seconds: float = 0
    availability: Availability = Availability.UNAVAILABLE


class GenerationResponse(Model):
    text: str
    metrics: InferenceMetrics


class InferenceProvider(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...
    async def metrics(self) -> InferenceMetrics: ...


class Location(Model):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Forecast(Model):
    timestamp: AwareDatetime = Field(default_factory=utcnow)
    availability: Availability = Availability.UNAVAILABLE
    location: Location
    hours: list[dict[str, Any]] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class WeatherProvider(Protocol):
    async def forecast(self, location: Location) -> Forecast: ...


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    WAITING_FOR_ENERGY = "WAITING_FOR_ENERGY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Job(Model):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workload: str
    status: JobStatus = JobStatus.QUEUED
    priority: int = 50  # 100 or greater: critical
    created_at: AwareDatetime = Field(default_factory=utcnow)
    scheduled_at: AwareDatetime = Field(default_factory=utcnow)
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    deadline: AwareDatetime | None = None
    deferrable: bool = True
    estimated_energy_wh: float | None = Field(default=None, ge=0)
    actual_estimated_energy_wh: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    dedupe_key: str | None = None


class JobResult(Model):
    data: dict[str, Any] = Field(default_factory=dict)
    inference_metrics: list[InferenceMetrics] = Field(default_factory=list)
    forecast: Forecast | None = None
    briefing: str | None = None


class Workload(Protocol):
    async def run(self, job: Job) -> JobResult: ...


class Decision(Model):
    timestamp: AwareDatetime = Field(default_factory=utcnow)
    job_id: str
    decision: str
    energy_state: EnergyState
    reason: str


class Cache(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None: ...
    async def put(self, key: str, value: dict[str, Any]) -> None: ...


class Queue(Protocol):
    async def enqueue(self, job: Job) -> Job: ...
    async def pending(self) -> list[Job]: ...
    async def claim(self, job_id: str) -> Job | None: ...
    async def defer(self, job_id: str) -> None: ...
    async def finish(self, job: Job, result: JobResult | None, estimate: EnergyEstimate,
                     error: str | None = None) -> None: ...
    async def record_decision(self, decision: Decision) -> None: ...
