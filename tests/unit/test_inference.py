"""Inference telemetry parsing and failure reporting."""

import asyncio

import httpx
import pytest

from contracts import GenerationRequest
from inference import InferenceUnavailable, LlamaCppProvider
from inference.mock import MockInferenceProvider
from tests.mocks.doubles import fixture_json

COMPLETION = fixture_json("llamacpp_completion.json")
METRICS_TEXT = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1200
llamacpp:tokens_predicted_total 300
llamacpp:prompt_tokens_seconds 41.5
llamacpp:predicted_tokens_seconds 43.1
llamacpp:requests_processing 1
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:requests_deferred{slot="0"} 0
"""


def provider(handler) -> LlamaCppProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LlamaCppProvider("http://127.0.0.1:8080", "qwen3.5-9b", client=client)


def test_generation_reports_request_tokens_and_rates_from_timings() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=COMPLETION)

    response = asyncio.run(provider(handler).generate(GenerationRequest(prompt="hello")))

    assert response.text == COMPLETION["choices"][0]["message"]["content"]
    metrics = response.metrics
    assert metrics.source == "generate"
    assert metrics.prompt_tokens == COMPLETION["timings"]["prompt_n"]
    assert metrics.generated_tokens == COMPLETION["timings"]["predicted_n"]
    assert metrics.generation_tokens_per_second == pytest.approx(
        COMPLETION["timings"]["predicted_per_second"]
    )
    assert metrics.runtime_seconds == pytest.approx(
        (COMPLETION["timings"]["prompt_ms"] + COMPLETION["timings"]["predicted_ms"]) / 1000
    )


def test_server_metrics_report_counter_differences_and_gauges() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=METRICS_TEXT)

    client = provider(handler)
    first = asyncio.run(client.metrics())
    assert first.prompt_tokens == 1200
    assert first.interval_seconds == 0
    assert first.active_requests == 1
    assert first.kv_cache_usage == pytest.approx(0.25)
    assert first.prompt_tokens_per_second == pytest.approx(41.5)


def test_missing_content_is_reported_as_unavailable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(InferenceUnavailable):
        asyncio.run(provider(handler).generate(GenerationRequest(prompt="hello")))


def test_http_failure_never_returns_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "loading model"})

    with pytest.raises(InferenceUnavailable) as error:
        asyncio.run(provider(handler).generate(GenerationRequest(prompt="hello")))
    assert "503" in str(error.value)


def test_health_and_metrics_outage_do_not_raise() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    client = provider(handler)
    assert asyncio.run(client.health()) is False
    metrics = asyncio.run(client.metrics())
    assert metrics.availability.value == "UNAVAILABLE"
    assert "unavailable" in (metrics.reason or "")


def test_mock_provider_labels_simulated_telemetry() -> None:
    client = MockInferenceProvider(text="brief")
    response = asyncio.run(client.generate(GenerationRequest(prompt="hello")))
    assert response.text == "brief"
    assert response.metrics.generated_tokens == 256
    assert "Simulated" in (response.metrics.reason or "")
