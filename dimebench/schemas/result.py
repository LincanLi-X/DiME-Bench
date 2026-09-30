"""Sample-level result schema."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from dimebench.schemas.base import RawText, StrictSchema

FailureClass = Literal[
    "empty_output",
    "parse_error",
    "context_copy",
    "full_document_output",
    "truncated_output",
    "runtime_error",
    "unsafe_code_result",
]
ResultStatus = Literal["success", "failure", "skipped"]


class ResultSpec(StrictSchema):
    """Auditable result for one sample before run-level aggregation."""

    model_config = ConfigDict(str_strip_whitespace=False)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    status: ResultStatus
    raw_output: RawText | None = None
    parsed_output: JsonValue | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_class: FailureClass | None = None
    error_message: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_contract(self) -> ResultSpec:
        """Keep successful and failed result states internally consistent."""
        if self.status == "success" and (
            self.failure_class is not None or self.error_message is not None
        ):
            raise ValueError("successful results cannot contain failure details")
        if self.status == "failure" and self.failure_class is None:
            raise ValueError("failed results require failure_class")
        return self
