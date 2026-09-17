"""Read-only VRM access-token client with restart-safe polling and backoff."""

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from contracts import Availability, Cache, EnergyState, utcnow
from energy.vrm.parser import parse_diagnostics


class VRMProvider:
    def __init__(self, client: httpx.AsyncClient, cache: Cache, installation_id: str,
                 access_token: str, *, poll_interval_seconds: float = 60,
                 freshness_seconds: float = 300,
                 base_url: str = "https://vrmapi.victronenergy.com/v2",
                 clock: Callable[[], datetime] = utcnow) -> None:
        if poll_interval_seconds < 0 or freshness_seconds < 0:
            raise ValueError("Polling and freshness intervals must be nonnegative")
        self.client, self.cache = client, cache
        self.access_token = access_token
        self.url = f"{base_url.rstrip('/')}/installations/{quote(str(installation_id), safe='')}/diagnostics"
        self.key = "vrm:v1:" + hashlib.sha256(self.url.encode()).hexdigest()
        self.poll_interval_seconds, self.freshness_seconds = poll_interval_seconds, freshness_seconds
        self.clock = clock
        self._lock = asyncio.Lock()

    async def current(self) -> EnergyState:
        async with self._lock:
            now = self.clock()
            entry = await self.cache.get(self.key) or {}
            try:
                next_poll = datetime.fromisoformat(entry["next_poll"])
                if next_poll.tzinfo is not None and now < next_poll:
                    return self._cached(entry, now)
            except (KeyError, TypeError, ValueError):
                pass
            next_poll = now + timedelta(seconds=self.poll_interval_seconds)
            try:
                if not self.access_token:
                    raise ValueError("VRM access token not configured")
                records: list[dict[str, Any]] = []
                for page in range(1, 101):
                    response = await self.client.get(
                        self.url, headers={"x-authorization": f"Token {self.access_token}"},
                        params={"count": 1000, "page": page},
                    )
                    if response.status_code == 429:
                        next_poll = max(next_poll, self._retry_after(response, now))
                    response.raise_for_status()
                    raw = response.json()
                    if not isinstance(raw, dict) or raw.get("success") is not True:
                        raise ValueError("VRM diagnostics request was unsuccessful")
                    batch = raw.get("records")
                    if isinstance(batch, dict):
                        batch = [batch] if "dbusPath" in batch else list(batch.values())
                    if not isinstance(batch, list):
                        raise TypeError("VRM diagnostics records are invalid")
                    records.extend(batch)
                    total = raw.get("num_records")
                    if not batch or (isinstance(total, int) and len(records) >= total):
                        break
                    if total is None and len(batch) < 1000:
                        break
                else:
                    raise ValueError("VRM diagnostics pagination did not complete")
                raw = {"success": True, "records": records, "num_records": len(records)}
                state = parse_diagnostics(raw, now, self.freshness_seconds)
                if state.availability == Availability.UNAVAILABLE:
                    raise ValueError("VRM returned no usable system measurements")
                entry = {"raw": raw, "normalized": state.model_dump(mode="json"),
                         "fetched_at": now.isoformat(), "next_poll": next_poll.isoformat()}
                await self.cache.put(self.key, entry)
                return state
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                # Never persist exception URLs/headers, which may contain credentials.
                reason = (f"VRM HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError)
                          else f"VRM unavailable ({type(exc).__name__})")
                entry = {**entry, "next_poll": next_poll.isoformat(), "error": reason}
                await self.cache.put(self.key, entry)
                return self._cached(entry, now)

    def _cached(self, entry: dict[str, Any], now: datetime) -> EnergyState:
        try:
            state = EnergyState.model_validate(entry["normalized"])
        except (KeyError, ValueError, TypeError):
            return EnergyState(timestamp=now, reason=entry.get("error", "VRM cache unavailable"))
        if state.availability == Availability.UNAVAILABLE:
            return state
        old = (now - state.timestamp).total_seconds() > self.freshness_seconds
        if old or entry.get("error"):
            state = state.model_copy(update={
                "availability": Availability.STALE,
                "reason": "; ".join(filter(None, [state.reason, entry.get("error"),
                                                "Cached VRM measurement is stale" if old else None])),
            })
        return state

    def _retry_after(self, response: httpx.Response, now: datetime) -> datetime:
        value = response.headers.get("Retry-After", "")
        try:
            return now + timedelta(seconds=max(0, int(value)))
        except (ValueError, OverflowError):
            try:
                stamp = parsedate_to_datetime(value)
                return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp
            except (TypeError, ValueError, OverflowError):
                return now + timedelta(seconds=max(self.poll_interval_seconds, 60))


class MockVRMProvider:
    def __init__(self, state: EnergyState | None = None, *,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self.state, self.clock = state, clock

    async def current(self) -> EnergyState:
        if self.state is not None:
            return self.state.model_copy(deep=True)
        return EnergyState(timestamp=self.clock(), availability=Availability.FRESH,
                           battery_soc=85, solar_power_w=650, battery_power_w=300,
                           ac_load_w=200, reason="Synthetic mock energy; not live telemetry")
