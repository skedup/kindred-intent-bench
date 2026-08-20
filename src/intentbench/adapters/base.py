"""Provider-independent protocols; no generic SDK abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationResult:
    value: dict[str, Any]
    requested_model: str
    reported_model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    raw_response_sha256: str
    usage_metadata: dict[str, int]
    structured_output_mode: str


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    requested_model: str
    latency_ms: float
    raw_response_sha256: str


class ProviderError(RuntimeError):
    """Sanitized Provider failure safe for readiness metadata and logs."""

    def __init__(self, error_type: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class StructuredGenerationClient(Protocol):
    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
        generation_parameters: dict[str, Any] | None = None,
    ) -> GenerationResult: ...


class EmbeddingClient(Protocol):
    def embed(self, *, model: str, text: str) -> EmbeddingResult: ...
