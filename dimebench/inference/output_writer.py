"""Convert adapter responses into append-only raw prediction records."""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from dimebench.artifacts import PredictionRecord, PredictionStore
from dimebench.artifacts.hashing import canonical_json
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse
from dimebench.schemas.run import RunSpec


class PredictionWriter:
    """Inference-only writer that never parses or scores model output."""

    VERSION = "1.0.0"

    def __init__(self, spec: RunSpec, store: PredictionStore) -> None:
        self.spec = spec
        self.store = store

    def write(
        self,
        request: ModelRequest,
        response: ModelResponse,
        *,
        cache_hit: bool,
        attempts: int,
    ) -> PredictionRecord:
        """Persist raw text/scores and complete decoding metadata."""
        usage = response.usage.model_dump(mode="json", exclude_none=False)
        metadata: dict[str, JsonValue] = dict(response.decoding_metadata)
        metadata["inference"] = {
            "writer_version": self.VERSION,
            "finish_reason": response.finish_reason,
            "cache_hit": cache_hit,
            "attempts": attempts,
            "usage": usage,
        }
        if response.option_scores is not None:
            metadata["option_scores"] = cast(JsonValue, response.option_scores)
        request_payload = cast(
            dict[str, JsonValue],
            request.model_dump(mode="json", exclude_none=False),
        )
        if response.status == "success":
            raw_output = (
                response.text
                if response.text is not None
                else canonical_json(response.option_scores)
            )
            record = PredictionRecord(
                run_id=self.spec.run_id,
                config_hash=self.spec.config_hash,
                sample_id=request.sample_id,
                model_id=self.spec.model.id,
                dataset_id=self.spec.dataset.id,
                task_id=self.spec.task.id,
                status="success",
                request=request_payload,
                raw_output=raw_output,
                decoding_metadata=metadata,
            )
        else:
            record = PredictionRecord(
                run_id=self.spec.run_id,
                config_hash=self.spec.config_hash,
                sample_id=request.sample_id,
                model_id=self.spec.model.id,
                dataset_id=self.spec.dataset.id,
                task_id=self.spec.task.id,
                status="failure",
                request=request_payload,
                failure_class="runtime_error",
                error_message=f"{response.error_type}: {response.error_message}",
                decoding_metadata=metadata,
            )
        self.store.append(record)
        return record


__all__ = ["PredictionWriter"]
