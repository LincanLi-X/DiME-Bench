"""Canonical sample records shared by all DiME-Bench tracks."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from dimebench.artifacts.hashing import hash_json
from dimebench.schemas.base import StrictSchema

SampleKind = Literal[
    "multiple_choice",
    "generation",
    "infilling",
    "editing",
    "reasoning",
]
_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]*$"
_SAMPLE_ID_PATTERN = r"^[a-z0-9][a-z0-9_.:-]*$"


class TextSpan(StrictSchema):
    """A half-open character span in the source text."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str | None = None

    @model_validator(mode="after")
    def validate_order(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("text span end must be greater than start")
        return self


class BaseSampleRecord(StrictSchema):
    """Fields required for every normalized benchmark sample."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(min_length=1, pattern=_SAMPLE_ID_PATTERN)
    dataset_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    split: str = Field(min_length=1, pattern=_ID_PATTERN)
    source_id: str = Field(min_length=1)
    kind: SampleKind
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def record_hash(self) -> str:
        """Hash the complete normalized record, including its stable ID."""
        return hash_json(self.model_dump(mode="json", exclude_none=False))


class MultipleChoiceRecord(BaseSampleRecord):
    """A question with a finite ordered answer set."""

    kind: Literal["multiple_choice"] = "multiple_choice"
    question: str = Field(min_length=1)
    choices: tuple[str, ...] = Field(min_length=2)
    answer_index: int = Field(ge=0)
    context: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def validate_answer_index(self) -> MultipleChoiceRecord:
        if self.answer_index >= len(self.choices):
            raise ValueError("answer_index must refer to an available choice")
        if any(not choice.strip() for choice in self.choices):
            raise ValueError("multiple-choice options cannot be empty")
        return self


class GenerationRecord(BaseSampleRecord):
    """A free-form generation, code, or instruction-following sample."""

    kind: Literal["generation"] = "generation"
    instruction: str = Field(min_length=1)
    references: tuple[str, ...] = Field(min_length=1)
    context: str | None = None
    tests: tuple[str, ...] = ()


class InfillingRecord(BaseSampleRecord):
    """A prefix-middle-suffix record for bidirectional generation."""

    kind: Literal["infilling"] = "infilling"
    prefix: str = Field(min_length=1)
    middle: str = Field(min_length=1)
    suffix: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    span_length: int = Field(gt=0)
    prefix_length: int = Field(gt=0)
    suffix_length: int = Field(gt=0)
    span_length_bin: Literal["short", "medium", "long"]
    boundary_density: Literal["low", "high"]

    @model_validator(mode="after")
    def validate_lengths(self) -> InfillingRecord:
        expected = (
            len(self.prefix.split()),
            len(self.middle.split()),
            len(self.suffix.split()),
        )
        observed = (self.prefix_length, self.span_length, self.suffix_length)
        if expected != observed:
            raise ValueError(
                "infilling token lengths must equal whitespace-tokenized lengths"
            )
        return self


class EditingRecord(BaseSampleRecord):
    """A full-text revision sample with explicit locality metadata."""

    kind: Literal["editing"] = "editing"
    source_text: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    reference_edit: str = Field(min_length=1)
    target_spans: tuple[TextSpan, ...] = ()
    non_target_spans: tuple[TextSpan, ...] = ()
    minimum_edit_distance: int = Field(ge=0)
    edit_type: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_spans(self) -> EditingRecord:
        source_length = len(self.source_text)
        all_spans = self.target_spans + self.non_target_spans
        if any(span.end > source_length for span in all_spans):
            raise ValueError("editing spans must fall inside source_text")
        ordered = sorted(all_spans, key=lambda span: (span.start, span.end))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start < previous.end:
                raise ValueError("editing spans cannot overlap")
        return self


class ReasoningRecord(BaseSampleRecord):
    """A chain-style or planning-style direct-answer sample."""

    kind: Literal["reasoning"] = "reasoning"
    problem: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    reasoning_type: Literal["chain", "planning"]
    choices: tuple[str, ...] = ()
    answer_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_choices(self) -> ReasoningRecord:
        if self.choices:
            if self.answer_index is None:
                raise ValueError("reasoning choices require answer_index")
            if self.answer_index >= len(self.choices):
                raise ValueError("answer_index must refer to an available choice")
        elif self.answer_index is not None:
            raise ValueError("answer_index requires reasoning choices")
        return self


SampleRecord: TypeAlias = Annotated[
    MultipleChoiceRecord
    | GenerationRecord
    | InfillingRecord
    | EditingRecord
    | ReasoningRecord,
    Field(discriminator="kind"),
]
_SAMPLE_ADAPTER: TypeAdapter[SampleRecord] = TypeAdapter(SampleRecord)


def stable_sample_id(
    dataset_id: str,
    split: str,
    source_id: str,
    normalized_payload: object,
) -> str:
    """Create a deterministic content-addressed sample identifier."""
    digest = hash_json(
        {
            "dataset_id": dataset_id,
            "split": split,
            "source_id": source_id,
            "payload": normalized_payload,
        }
    )
    return f"{dataset_id}.{split}.{digest[:16]}"


def expected_sample_id(record: BaseSampleRecord) -> str:
    """Recompute the stable ID from a normalized record."""
    payload = record.model_dump(
        mode="json",
        exclude={"sample_id", "schema_version", "dataset_id", "split", "source_id"},
        exclude_none=False,
    )
    return stable_sample_id(
        record.dataset_id,
        record.split,
        record.source_id,
        payload,
    )


def parse_sample_record(payload: object) -> SampleRecord:
    """Parse a discriminated normalized sample record."""
    return _SAMPLE_ADAPTER.validate_python(payload)


__all__ = [
    "BaseSampleRecord",
    "EditingRecord",
    "GenerationRecord",
    "InfillingRecord",
    "MultipleChoiceRecord",
    "ReasoningRecord",
    "SampleKind",
    "SampleRecord",
    "TextSpan",
    "expected_sample_id",
    "parse_sample_record",
    "stable_sample_id",
]
