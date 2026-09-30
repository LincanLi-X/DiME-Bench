"""Content-addressed model-response cache."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from dimebench.artifacts.hashing import hash_json, write_json_atomic
from dimebench.artifacts.provenance import utc_now
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelSpec
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse
from dimebench.schemas.run import DecodingSpec


class CacheError(ValueError):
    """Raised when a cache entry is corrupt or has the wrong identity."""


class CacheEntry(StrictSchema):
    """One response bound to request, model, and decoding hashes."""

    schema_version: Literal["1.0"] = "1.0"
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response: ModelResponse
    created_at: datetime = Field(default_factory=utc_now)


class ResponseCache:
    """Filesystem cache with atomic writes and identity validation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(
        request: ModelRequest,
        model: ModelSpec,
        decoding: DecodingSpec,
    ) -> str:
        return hash_json(
            {
                "request": request,
                "model": model,
                "decoding": decoding,
            }
        )

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(
            character not in "0123456789abcdef" for character in key
        ):
            raise CacheError("cache key must be a lowercase SHA-256 digest")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise CacheError(f"invalid inference cache entry {path}: {exc}") from exc
        if entry.key != key:
            raise CacheError(f"cache key mismatch in {path}")
        return entry.response

    def put(
        self,
        key: str,
        request: ModelRequest,
        model: ModelSpec,
        decoding: DecodingSpec,
        response: ModelResponse,
    ) -> None:
        expected = self.key_for(request, model, decoding)
        if key != expected:
            raise CacheError("cache key does not match request/model/decoding inputs")
        entry = CacheEntry(
            key=key,
            request_hash=hash_json(request),
            model_hash=hash_json(model),
            decoding_hash=hash_json(decoding),
            response=response,
        )
        write_json_atomic(self._path(key), entry)


__all__ = ["CacheEntry", "CacheError", "ResponseCache"]
