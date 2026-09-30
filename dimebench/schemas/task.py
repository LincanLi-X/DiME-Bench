"""Task configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dimebench.schemas.base import StrictSchema

TrackId = Literal[
    "track1_general",
    "track2_infilling",
    "track3_editing",
    "track4_reasoning",
]
TaskType = Literal[
    "multiple_choice",
    "generation",
    "code",
    "infilling",
    "editing",
    "reasoning",
]
ReasoningType = Literal["chain", "planning"]


class TaskSpec(StrictSchema):
    """Task semantics, prompt contract, parser, and requested metrics."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    track: TrackId
    dataset_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    type: TaskType
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(default="1.0.0", min_length=1)
    parser_id: str = Field(min_length=1)
    parser_version: str = Field(default="1.0.0", min_length=1)
    metrics: tuple[str, ...] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    reasoning_type: ReasoningType | None = None

    @model_validator(mode="after")
    def validate_metric_and_track_contract(self) -> TaskSpec:
        """Enforce metric uniqueness and Track-specific task semantics."""
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("task metrics must be unique")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must be included in metrics")
        required_type = {
            "track2_infilling": "infilling",
            "track3_editing": "editing",
            "track4_reasoning": "reasoning",
        }.get(self.track)
        if required_type is not None and self.type != required_type:
            raise ValueError(f"{self.track} requires type={required_type!r}")
        if self.track == "track4_reasoning":
            if self.reasoning_type is None:
                raise ValueError(
                    "track4_reasoning requires reasoning_type='chain' or 'planning'"
                )
        elif self.reasoning_type is not None:
            raise ValueError("reasoning_type is only valid for track4_reasoning")
        return self
