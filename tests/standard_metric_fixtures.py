"""Builders shared by Step 10 standard-metric tests."""

from __future__ import annotations

from typing import Any

from dimebench.datasets import (
    GenerationRecord,
    InfillingRecord,
    MultipleChoiceRecord,
    ReasoningRecord,
    SampleRecord,
)
from dimebench.schemas.result import ResultSpec


def result_for(sample: SampleRecord, parsed_output: Any, *, status: str = "success"):
    kwargs: dict[str, Any] = {}
    if status == "failure":
        kwargs.update(failure_class="parse_error", error_message="fixture failure")
    return ResultSpec(
        run_id="step10",
        config_hash="a" * 64,
        sample_id=sample.sample_id,
        model_id="fixture-model",
        dataset_id=sample.dataset_id,
        task_id="fixture-task",
        status=status,
        raw_output=str(parsed_output),
        parsed_output=parsed_output,
        metadata={"prediction_hash": "b" * 64},
        **kwargs,
    )


def multiple_choice_sample() -> MultipleChoiceRecord:
    return MultipleChoiceRecord(
        sample_id="fixture.test.multiple-choice",
        dataset_id="fixture",
        split="test",
        source_id="multiple-choice",
        question="Which option is correct?",
        choices=("first", "second", "third"),
        answer_index=1,
    )


def generation_sample(
    reference: str = "reference",
    *,
    tests: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    instruction: str = "Return an answer.",
) -> GenerationRecord:
    return GenerationRecord(
        sample_id="fixture.test.generation",
        dataset_id="fixture",
        split="test",
        source_id="generation",
        instruction=instruction,
        references=(reference,),
        tests=tests,
        metadata=metadata or {},
    )


def infilling_sample(reference: str) -> InfillingRecord:
    return InfillingRecord(
        sample_id="fixture.test.infilling",
        dataset_id="fixture",
        split="test",
        source_id="infilling",
        prefix="left boundary",
        middle=reference,
        suffix="right boundary",
        domain="fixture",
        span_length=len(reference.split()),
        prefix_length=2,
        suffix_length=2,
        span_length_bin="short",
        boundary_density="low",
    )


def option_sample() -> ReasoningRecord:
    return ReasoningRecord(
        sample_id="fixture.test.option",
        dataset_id="fixture",
        split="test",
        source_id="option",
        problem="What is too large?",
        answer="the trophy",
        reasoning_type="planning",
        choices=("the trophy", "the suitcase"),
        answer_index=0,
    )


def path_sample(*, obstacles: list[str] | None = None) -> ReasoningRecord:
    return ReasoningRecord(
        sample_id="fixture.test.path",
        dataset_id="fixture",
        split="test",
        source_id="path",
        problem="Find a path from C to D.",
        answer="C-B-A-D",
        reasoning_type="planning",
        metadata={
            "nodes": ["A", "B", "C", "D"],
            "edges": ["A-B", "B-C", "A-D"],
            "obstacles": obstacles or [],
        },
    )


__all__ = [
    "generation_sample",
    "infilling_sample",
    "multiple_choice_sample",
    "option_sample",
    "path_sample",
    "result_for",
]
