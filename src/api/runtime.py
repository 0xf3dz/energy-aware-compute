"""Composition root. The only place where concrete providers are chosen.

Every component receives interfaces through its constructor. Swapping VRM for a
plug meter, or llama.cpp for another server, is a change in this file only.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as clock_time
from typing import Any

import httpx

from api.settings import Settings
from contracts import InferenceProvider, Job, Location, Queue, utcnow
from db import PostgresQueue
from energy.mac_power import MockEnergyMonitor, PowermetricsProvider
from energy.models import UnavailableEnergyProvider
from energy.vrm import MockVRMProvider, VRMProvider
from inference import LlamaCppProvider, MockInferenceProvider
from jobs.queue import MemoryCache, MockQueue
from jobs.worker import Worker
from scheduler.engine import Scheduler
from scheduler.models import HIGH_PRIORITY
from scheduler.policies import Policy
from workloads.weather import (
    MockWeatherProvider,
    OpenMeteoProvider,
    WeatherBriefingWorkload,
)
from workloads.weather.provider import synthetic_forecast

logger = logging.getLogger(__name__)

BRIEFING_WORKLOAD = "weather_briefing"


@dataclass
class Runtime:
    settings: Settings
    queue: Any
    cache: Any
    inference: InferenceProvider
    weather: Any
    energy: Any
    monitor: Any
    worker: Worker
    scheduler: Scheduler
    client: httpx.AsyncClient
    owns_queue: bool = True
    tasks: list[asyncio.Task] = field(default_factory=list)

    async def snapshot(self) -> dict[str, Any]:
        data = await self.queue.snapshot()
        metrics = await _safe_metrics(self.inference)
        data["runtime"] = {
            "demo": self.settings.demo,
            "model": self.settings.llama_model,
            "llama_url": self.settings.llama_url,
            "vrm_url": "https://vrm.victronenergy.com/installation/"
            + (self.settings.vrm_installation_id or ""),
            "energy_source": type(self.energy).__name__,
            "monitor": type(self.monitor).__name__,
            "poll_seconds": self.settings.poll_seconds,
            "policy": {
                "battery_floor": self.settings.battery_floor,
                "high_soc": self.settings.high_soc,
                "surplus_w": self.settings.surplus_minimum_watts,
                "surplus_hold_seconds": self.settings.surplus_duration_seconds,
            },
            "surplus_held_seconds": round(self.scheduler.policy.surplus_held_seconds, 1),
            "schedules": [
                {"id": item.schedule_id, "workload": item.workload, "at_utc": item.at.isoformat()}
                for item in self.scheduler.schedules
            ],
        }
        data["inference"]["live"] = metrics
        return data

    async def enqueue_briefing(
        self,
        *,
        deferrable: bool = False,
        priority: int = HIGH_PRIORITY,
        estimated_energy_wh: float | None = 5,
        scheduled_at: datetime | None = None,
    ) -> Job:
        created = scheduled_at or utcnow()
        return await self.queue.enqueue(
            Job(
                workload=BRIEFING_WORKLOAD,
                priority=priority,
                deferrable=deferrable,
                estimated_energy_wh=estimated_energy_wh,
                created_at=created,
                scheduled_at=created,
                payload={"estimated_runtime_seconds": 60},
            )
        )

    async def run_scheduler(self, stop: asyncio.Event) -> None:
        await self.scheduler.run(stop)

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.exception("Runtime task failed during shutdown")
        closer = getattr(self.inference, "aclose", None)
        if closer is not None:
            await closer()
        await self.client.aclose()
        if self.owns_queue:
            await self.queue.close()


async def build_runtime(settings: Settings, *, clock: Callable[[], datetime] | None = None) -> Runtime:
    """Assemble the application from settings. Demo mode uses no external service.

    ``clock`` is for demonstrations and tests. Without it every component uses
    the wall clock.
    """
    now = clock or utcnow
    client = httpx.AsyncClient()
    if settings.demo:
        queue: Queue = MockQueue()
        cache = MemoryCache()
    else:
        queue = PostgresQueue(settings.database_url)
        await queue.open()
        cache = queue
    location = (
        Location(latitude=settings.latitude, longitude=settings.longitude)
        if settings.latitude is not None and settings.longitude is not None
        else None
    )
    if settings.demo:
        point = location or Location(latitude=0, longitude=0)
        location = point
        inference: InferenceProvider = MockInferenceProvider(
            text="Simulated briefing: no model and no forecast provider were used."
        )
        weather = MockWeatherProvider(synthetic_forecast(point, now()), clock=now)
        energy = MockVRMProvider(clock=now)
        monitor = MockEnergyMonitor(power_w=20)
    else:
        inference = LlamaCppProvider(
            settings.llama_url, settings.llama_model, api_key=settings.inference_key()
        )
        weather = OpenMeteoProvider(client, cache)
        if settings.vrm_token and settings.vrm_installation_id:
            energy = VRMProvider(
                client,
                cache,
                settings.vrm_installation_id,
                settings.vrm_token.get_secret_value(),
            )
        else:
            energy = UnavailableEnergyProvider(
                "VRM is not configured; set VRM_TOKEN and VRM_INSTALLATION_ID"
            )
        monitor = PowermetricsProvider(
            llama_pid=settings.llama_pid,
            idle_baseline_w=settings.idle_baseline_w,
            max_duration_seconds=settings.energy_measure_max_seconds,
        )
    workloads = {BRIEFING_WORKLOAD: WeatherBriefingWorkload(weather, inference, location)}
    worker = Worker(queue, workloads, monitor)
    policy = Policy(
        safety_floor_soc=settings.battery_floor,
        high_soc=settings.high_soc,
        surplus_w=settings.surplus_minimum_watts,
        surplus_hold_seconds=settings.surplus_duration_seconds,
        max_age_seconds=settings.energy_max_age_seconds,
    )
    scheduler = Scheduler(
        queue,
        energy,
        worker,
        policy,
        poll_seconds=settings.poll_seconds,
        clock=now,
        ownership=None if settings.demo else queue.ownership(),
    )
    scheduler.register_daily(
        "weather-briefing-daily",
        BRIEFING_WORKLOAD,
        clock_time(hour=settings.briefing_hour_utc),
        lead_minutes=settings.briefing_lead_minutes,
        priority=HIGH_PRIORITY,
        deferrable=False,
        estimated_energy_wh=5,
        payload={"estimated_runtime_seconds": 120},
    )
    runtime = Runtime(
        settings=settings,
        queue=queue,
        cache=cache,
        inference=inference,
        weather=weather,
        energy=energy,
        monitor=monitor,
        worker=worker,
        scheduler=scheduler,
        client=client,
        owns_queue=not settings.demo,
    )
    if not settings.demo:
        await runtime.queue.recover_interrupted()
    return runtime


async def _safe_metrics(inference: InferenceProvider) -> dict[str, Any]:
    metrics = await inference.metrics()
    return metrics.model_dump(mode="json")
