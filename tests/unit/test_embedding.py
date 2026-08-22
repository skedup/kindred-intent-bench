from __future__ import annotations

import hashlib
from pathlib import Path

from intentbench.adapters.base import EmbeddingResult
from intentbench.embedding import EmbeddingCache


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self,
        *,
        model: str,
        text: str,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbeddingResult:
        self.calls += 1
        digest = hashlib.sha256(text.encode()).hexdigest()
        return EmbeddingResult(
            vector=(1.0, 2.0, 3.0),
            requested_model=model,
            latency_ms=1.0,
            raw_response_sha256=digest,
        )


def test_embedding_cache_keys_content_and_adapter_and_avoids_second_call(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()
    cache = EmbeddingCache(tmp_path)
    arguments = {
        "client": client,
        "text": "想去散步",
        "model": "gemini-embedding-001",
        "adapter_config_sha256": "a" * 64,
        "task_type": "SEMANTIC_SIMILARITY",
        "output_dimensionality": 3,
        "expected_dimension": 3,
    }
    first = cache.get_or_create(**arguments)  # type: ignore[arg-type]
    second = cache.get_or_create(**arguments)  # type: ignore[arg-type]
    assert client.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.vector == first.vector
