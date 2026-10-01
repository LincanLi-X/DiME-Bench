"""Shared builders for committed Step 9 golden cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from dimebench.artifacts import PredictionRecord
from dimebench.datasets import (
    EditingRecord,
    GenerationRecord,
    InfillingRecord,
    MultipleChoiceRecord,
    ReasoningRecord,
    SampleRecord,
)
from dimebench.schemas.task import TaskSpec, TaskType, TrackId

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "tests" / "fixtures" / "postprocessors" / "golden.json"


def load_golden_cases() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads(GOLDEN_PATH.read_text()))


def sample_for(kind: str) -> SampleRecord:
    common = {
        "sample_id": f"fixture.test.{kind}",
        "dataset_id": "fixture",
        "split": "test",
        "source_id": kind,
    }
    if kind == "multiple_choice":
        return MultipleChoiceRecord(
            **common,
            question="What process turns liquid water into vapor?",
            choices=("Condensation", "Evaporation", "Freezing"),
            answer_index=1,
        )
    if kind == "reasoning_choice":
        return ReasoningRecord(
            **common,
            problem="What is too large?",
            answer="the trophy",
            reasoning_type="planning",
            choices=("the trophy", "the suitcase"),
            answer_index=0,
        )
    if kind == "generation":
        return GenerationRecord(
            **common,
            instruction="Return the requested result.",
            references=("reference",),
        )
    if kind == "infilling":
        return InfillingRecord(
            **common,
            prefix="The astronomer carefully recorded the",
            middle="newly discovered comet",
            suffix="before sunrise at the mountain observatory.",
            domain="fixture",
            span_length=3,
            prefix_length=5,
            suffix_length=6,
            span_length_bin="short",
            boundary_density="high",
        )
    if kind == "editing":
        return EditingRecord(
            **common,
            source_text="She go to school every day.",
            instruction="Correct the grammar.",
            reference_edit="She goes to school every day.",
            minimum_edit_distance=1,
            edit_type="grammar",
        )
    if kind == "reasoning_path":
        return ReasoningRecord(
            **common,
            problem="Find a path from C to D.",
            answer="C-B-A-D",
            reasoning_type="planning",
        )
    raise ValueError(f"unknown fixture sample {kind!r}")


def task_for(parser_id: str, task_type: str) -> TaskSpec:
    track: TrackId = {
        "infilling": "track2_infilling",
        "editing": "track3_editing",
        "reasoning": "track4_reasoning",
    }.get(task_type, "track1_general")
    return TaskSpec(
        id=f"fixture_{parser_id}",
        track=track,
        dataset_id="fixture",
        type=cast(TaskType, task_type),
        prompt_id="fixture.prompt",
        parser_id=parser_id,
        metrics=("fixture_metric",),
        primary_metric="fixture_metric",
        max_output_tokens=32,
        reasoning_type="planning" if task_type == "reasoning" else None,
    )


def prediction_for(
    case: dict[str, Any],
    sample: SampleRecord,
    task: TaskSpec,
) -> PredictionRecord:
    status = case.get("prediction_status", "success")
    if status == "failure":
        return PredictionRecord(
            run_id="step9_golden",
            config_hash="0" * 64,
            sample_id=sample.sample_id,
            model_id="fixture-model",
            dataset_id=sample.dataset_id,
            task_id=task.id,
            status="failure",
            failure_class="runtime_error",
            error_message="fixture backend failure",
            decoding_metadata={"inference": {"finish_reason": case["finish_reason"]}},
        )
    return PredictionRecord(
        run_id="step9_golden",
        config_hash="0" * 64,
        sample_id=sample.sample_id,
        model_id="fixture-model",
        dataset_id=sample.dataset_id,
        task_id=task.id,
        status="success",
        raw_output=case["raw_output"],
        decoding_metadata={"inference": {"finish_reason": case["finish_reason"]}},
    )


__all__ = [
    "GOLDEN_PATH",
    "load_golden_cases",
    "prediction_for",
    "sample_for",
    "task_for",
]
