"""Unified model request schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from dimebench.schemas.base import StrictSchema

RequestMode = Literal[
    "generate",
    "score_options",
    "infill",
    "edit",
    "reasoning",
]


class ModelRequest(StrictSchema):
    """One model-independent request passed to any adapter."""

    request_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    mode: RequestMode = "generate"
    prompt: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    seed: int = Field(default=0, ge=0)
    stop_sequences: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    prefix: str | None = None
    suffix: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mode_contract(self) -> ModelRequest:
        if len(set(self.stop_sequences)) != len(self.stop_sequences):
            raise ValueError("stop_sequences must be unique")
        if any(not sequence for sequence in self.stop_sequences):
            raise ValueError("stop_sequences cannot contain empty strings")
        if self.mode == "score_options":
            if len(self.options) < 2:
                raise ValueError("score_options requests require at least two options")
            if len(set(self.options)) != len(self.options):
                raise ValueError("score_options options must be unique")
        elif self.options:
            raise ValueError("options are only valid for score_options requests")
        if self.mode == "infill" and (self.prefix is None or self.suffix is None):
            raise ValueError("infill requests require both prefix and suffix")
        if self.mode != "infill" and (
            self.prefix is not None or self.suffix is not None
        ):
            raise ValueError("prefix and suffix are only valid for infill requests")
        return self
