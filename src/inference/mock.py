"""Deterministic inference double. No model, no network, no Apple hardware."""

import time
from collections.abc import Callable

from contracts import (
    Availability,
    GenerationRequest,
    GenerationResponse,
    InferenceMetrics,
)
from inference.interface import InferenceUnavailable


class MockInferenceProvider:
    """Return fixed text and invented telemetry that is labelled as simulated."""

    def __init__(
        self,
        text: str = "Simulated briefing text.",
        *,
        model: str = "mock",
        prompt_tokens: int = 512,
        generated_tokens: int = 256,
        fail_with: Exception | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.fail_with = fail_with
        self.clock = clock
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.fail_with is not None:
            raise self.fail_with
        self.requests.append(request)
        max_tokens = request.max_tokens
        generated = min(self.generated_tokens, max_tokens) if max_tokens else 0
        # One simulated request costs one second of runtime.
        runtime = 1.0
        metrics = InferenceMetrics(
            source="generate",
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=generated,
            prompt_tokens_per_second=self.prompt_tokens / runtime,
            generation_tokens_per_second=generated / runtime,
            runtime_seconds=runtime,
            availability=Availability.FRESH,
            reason="Simulated inference telemetry; no real model ran",
        )
        return GenerationResponse(text=self.text, metrics=metrics)

    async def metrics(self) -> InferenceMetrics:
        return InferenceMetrics(
            source="server_interval",
            model=self.model,
            active_requests=0,
            kv_cache_usage=0,
            availability=Availability.FRESH,
            reason="Simulated server metrics; no real model ran",
        )


def unavailable(reason: str = "Simulated inference outage") -> MockInferenceProvider:
    return MockInferenceProvider(fail_with=InferenceUnavailable(reason))
