"""Inference providers. Workloads depend on the contract, never on llama.cpp."""
from inference.interface import InferenceProvider, InferenceResult, InferenceUnavailable
from inference.llamacpp import LlamaCppProvider
from inference.mock import MockInferenceProvider

__all__ = [
    "InferenceProvider",
    "InferenceResult",
    "InferenceUnavailable",
    "LlamaCppProvider",
    "MockInferenceProvider",
]
