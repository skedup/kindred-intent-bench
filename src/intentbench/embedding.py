"""Content-addressed embedding cache used by the dev-only B1 runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from intentbench.adapters.base import EmbeddingClient
from intentbench.schemas import StrictModel


class EmbeddingCacheError(ValueError):
    """A cached vector disagrees with its registered embedding identity."""


class EmbeddingCacheRecord(StrictModel):
    schema_version: Literal[1] = 1
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_type: str | None
    output_dimensionality: int | None = Field(default=None, ge=1)
    vector: list[float] = Field(min_length=1)
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_ms: float = Field(ge=0)


@dataclass(frozen=True)
class CachedEmbedding:
    vector: tuple[float, ...]
    raw_response_sha256: str
    latency_ms: float
    cache_hit: bool


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def embedding_cache_key(
    *,
    text: str,
    model: str,
    adapter_config_sha256: str,
    task_type: str | None,
    output_dimensionality: int | None,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "text_sha256": _sha256_text(text),
                "model": model,
                "adapter_config_sha256": adapter_config_sha256,
                "task_type": task_type,
                "output_dimensionality": output_dimensionality,
            }
        )
    )


class EmbeddingCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, cache_key: str) -> Path:
        return self.directory / f"{cache_key}.json"

    def get_or_create(
        self,
        *,
        client: EmbeddingClient,
        text: str,
        model: str,
        adapter_config_sha256: str,
        task_type: str | None,
        output_dimensionality: int | None,
        expected_dimension: int,
    ) -> CachedEmbedding:
        cache_key = embedding_cache_key(
            text=text,
            model=model,
            adapter_config_sha256=adapter_config_sha256,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        path = self._path(cache_key)
        if path.is_file():
            try:
                record = EmbeddingCacheRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EmbeddingCacheError(f"invalid embedding cache record: {path}") from exc
            if record.cache_key != cache_key or record.text_sha256 != _sha256_text(text):
                raise EmbeddingCacheError("embedding cache identity mismatch")
            vector = tuple(record.vector)
            self._validate_vector(vector, expected_dimension)
            return CachedEmbedding(
                vector=vector,
                raw_response_sha256=record.raw_response_sha256,
                latency_ms=record.latency_ms,
                cache_hit=True,
            )

        result = client.embed(
            model=model,
            text=text,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        self._validate_vector(result.vector, expected_dimension)
        record = EmbeddingCacheRecord(
            cache_key=cache_key,
            text_sha256=_sha256_text(text),
            model=model,
            adapter_config_sha256=adapter_config_sha256,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            vector=list(result.vector),
            raw_response_sha256=result.raw_response_sha256,
            latency_ms=result.latency_ms,
        )
        payload = (_canonical_json(record.model_dump(mode="json")) + "\n").encode()
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{cache_key}-",
            suffix=".tmp",
            dir=self.directory,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
        return CachedEmbedding(
            vector=result.vector,
            raw_response_sha256=result.raw_response_sha256,
            latency_ms=result.latency_ms,
            cache_hit=False,
        )

    @staticmethod
    def _validate_vector(vector: tuple[float, ...], expected_dimension: int) -> None:
        if len(vector) != expected_dimension:
            raise EmbeddingCacheError(
                f"embedding dimension mismatch: expected={expected_dimension} actual={len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingCacheError("embedding vector must be finite")
