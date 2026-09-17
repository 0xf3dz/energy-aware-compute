"""OpenAI-compatible client for a local ``llama-server`` process.

Telemetry comes from two places:

* ``timings`` and ``usage`` in a chat completion response: the exact prompt and
  generation token counts and rates of one request.
* ``GET /metrics``: Prometheus counters and gauges of the whole server. This
  client reports counter differences between two consecutive calls, so the
  numbers describe a window instead of a lifetime total.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from contracts import (
    Availability,
    GenerationRequest,
    GenerationResponse,
    InferenceMetrics,
    utcnow,
)
from inference.interface import InferenceUnavailable

logger = logging.getLogger(__name__)

# Prometheus metric names published by llama.cpp.
PROMPT_TOKENS = "llamacpp:prompt_tokens_total"
GENERATED_TOKENS = "llamacpp:tokens_predicted_total"
PROMPT_RATE = "llamacpp:prompt_tokens_seconds"
GENERATION_RATE = "llamacpp:predicted_tokens_seconds"
ACTIVE_REQUESTS = "llamacpp:requests_processing"
DEFERRED_REQUESTS = "llamacpp:requests_deferred"
KV_USAGE = "llamacpp:kv_cache_usage_ratio"
KV_TOKENS = "llamacpp:kv_cache_tokens"


def parse_prometheus(text: str) -> dict[str, float]:
    """Return the last value of every unlabelled sample in Prometheus text."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        if "{" in name:
            continue
        try:
            values[name.strip()] = float(rest.strip().split()[0])
        except (ValueError, IndexError):
            continue
    return values


class LlamaCppProvider:
    """Serve generation and telemetry from one llama.cpp server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        disable_thinking: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.disable_thinking = disable_thinking
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds, headers=headers)
        self._previous: dict[str, float] | None = None
        self._previous_at: float | None = None
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def health(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5)
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            return False
        try:
            return response.json().get("status") == "ok"
        except ValueError:
            return False

    async def model_name(self) -> str | None:
        try:
            response = await self.client.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("models") or []
            return models[0].get("name") if models and isinstance(models[0], dict) else None
        except (httpx.HTTPError, ValueError, AttributeError, IndexError):
            return None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if self.disable_thinking:
            # Reasoning tokens cost energy and are not part of a briefing.
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["reasoning_format"] = "none"
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise InferenceUnavailable(self._describe(exc)) from exc
        except ValueError as exc:
            raise InferenceUnavailable("Inference server returned invalid JSON") from exc
        return self._parse_generation(payload)

    def _parse_generation(self, payload: dict[str, Any]) -> GenerationResponse:
        try:
            choices = payload["choices"]
            if not choices:
                raise ValueError("no choices")
            text = choices[0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InferenceUnavailable(f"Inference response has no message content: {exc}") from exc
        timings = payload.get("timings") or {}
        usage = payload.get("usage") or {}
        runtime = 0.0
        for key in ("prompt_ms", "predicted_ms"):
            value = timings.get(key)
            if isinstance(value, (int, float)):
                runtime += value / 1000
        prompt_tokens = _int_or_none(timings.get("prompt_n", usage.get("prompt_tokens")))
        generated = _int_or_none(timings.get("predicted_n", usage.get("completion_tokens")))
        metrics = InferenceMetrics(
            source="generate",
            model=str(payload.get("model") or self.model),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            prompt_tokens_per_second=_float_or_none(timings.get("prompt_per_second")),
            generation_tokens_per_second=_float_or_none(timings.get("predicted_per_second")),
            runtime_seconds=runtime,
            availability=Availability.FRESH,
        )
        return GenerationResponse(text=text, metrics=metrics)

    async def metrics(self) -> InferenceMetrics:
        """Report server counters since the previous call plus current gauges."""
        async with self._lock:
            now = self.clock()
            try:
                response = await self.client.get(f"{self.base_url}/metrics", timeout=10)
                response.raise_for_status()
                values = parse_prometheus(response.text)
            except (httpx.HTTPError, ValueError):
                return InferenceMetrics(model=self.model, reason="Metrics endpoint unavailable")
            interval = 0.0 if self._previous_at is None else max(0.0, now - self._previous_at)
            prompt_tokens = self._delta(values, PROMPT_TOKENS, interval)
            generated_tokens = self._delta(values, GENERATED_TOKENS, interval)
            self._previous, self._previous_at = values, now
            return InferenceMetrics(
                source="server_interval",
                interval_seconds=interval,
                model=self.model,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                prompt_tokens_per_second=_float_or_none(values.get(PROMPT_RATE)),
                generation_tokens_per_second=_float_or_none(values.get(GENERATION_RATE)),
                active_requests=_float_or_none(values.get(ACTIVE_REQUESTS)),
                kv_cache_usage=_float_or_none(values.get(KV_USAGE)),
                availability=Availability.FRESH,
            )

    def _delta(self, values: dict[str, float], name: str, interval: float) -> int | None:
        current = values.get(name)
        previous = (self._previous or {}).get(name)
        if current is None:
            return None
        if previous is None or interval <= 0:
            return int(current)
        return int(max(0, round(current - previous)))

    def _describe(self, exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"Inference server returned HTTP {exc.response.status_code}"
        return f"Inference server unreachable ({type(exc).__name__})"


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
