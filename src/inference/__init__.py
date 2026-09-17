"""Inference providers. Workloads depend on the contract, never on llama.cpp."""
from inference.interface import (
    InferenceProvider,
    InferenceUnavailable,
    request_text,
)
from inference.llamacpp import LlamaCppProvider
from inference.mock import MockInferenceProvider

__all__ = [
    "InferenceProvider",
    "InferenceUnavailable",
    "LlamaCppProvider",
    "MockInferenceProvider",
    "request_text",
]
