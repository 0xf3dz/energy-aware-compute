"""The inference boundary. Every workload sees only these names.

The concrete request and response shapes live in :mod:`contracts` so that a
future ``OllamaProvider``, ``MLXProvider``, ``RemoteProvider``, or
``MockProvider`` can be substituted without touching a workload.
"""
from contracts import (
    GenerationRequest,
    GenerationResponse,
    InferenceMetrics,
    InferenceProvider,
)


class InferenceUnavailable(RuntimeError):
    """The provider cannot serve the request. Callers must not fabricate output."""


async def request_text(
    provider: InferenceProvider, request: GenerationRequest
) -> tuple[str, InferenceMetrics]:
    """Generate text and return it with the metrics of that single request."""
    response: GenerationResponse = await provider.generate(request)
    return response.text, response.metrics


__all__ = [
    "GenerationRequest",
    "GenerationResponse",
    "InferenceMetrics",
    "InferenceProvider",
    "InferenceUnavailable",
    "request_text",
]
