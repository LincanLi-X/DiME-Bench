"""Unified model response and resource-usage schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from dimebench.schemas.base import RawText, StrictSchema


class ModelUsage(StrictSchema):
    """Portable generation resource and budget counters."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    forward_passes: int | None = Field(default=None, ge=0)
    denoising_steps: int | None = Field(default=None, ge=0)
    wall_time_seconds: float | None = Field(default=None, ge=0.0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)


class ModelResponse(StrictSchema):
    """One adapter response aligned to exactly one request."""

    model_config = ConfigDict(str_strip_whitespace=False)

    request_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    status: Literal["success", "error"]
    text: RawText | None = None
    option_scores: dict[str, float] | None = None
    finish_reason: Literal["stop", "length", "completed", "error", "unknown"]
    usage: ModelUsage = Field(default_factory=ModelUsage)
    decoding_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_status_contract(self) -> ModelResponse:
        if self.status == "success":
            if (self.text is None) == (self.option_scores is None):
                raise ValueError(
                    "successful responses require exactly one of text or option_scores"
                )
            if self.option_scores is not None and not self.option_scores:
                raise ValueError("option_scores cannot be empty")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful responses cannot contain error details")
            if self.finish_reason == "error":
                raise ValueError("successful responses cannot finish with error")
        else:
            if not self.error_type or not self.error_message:
                raise ValueError("error responses require error_type and error_message")
            if self.finish_reason != "error":
                raise ValueError("error responses require finish_reason='error'")
            if self.text is not None or self.option_scores is not None:
                raise ValueError("error responses cannot contain model outputs")
        return self
