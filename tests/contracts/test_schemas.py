from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from dimebench.registry import (
    DuplicateRegistrationError,
    Registry,
    RegistryError,
    UnknownRegistrationError,
)
from dimebench.schemas import ResultSpec, TaskSpec


def test_task_rejects_primary_metric_outside_metric_list() -> None:
    with pytest.raises(ValidationError, match="primary_metric must be included"):
        TaskSpec(
            id="bad_task",
            track="track2_infilling",
            dataset_id="dataset",
            type="infilling",
            prompt_id="prompt",
            parser_id="parser",
            metrics=("token_f1",),
            primary_metric="rouge_l",
            max_output_tokens=16,
        )


def test_track4_requires_reasoning_type() -> None:
    with pytest.raises(ValidationError, match="requires type='reasoning'"):
        TaskSpec(
            id="bad_reasoning",
            track="track4_reasoning",
            dataset_id="dataset",
            type="generation",
            prompt_id="prompt",
            parser_id="parser",
            metrics=("normalized_exact_match",),
            primary_metric="normalized_exact_match",
            max_output_tokens=16,
        )


def test_track3_requires_editing_task_type() -> None:
    with pytest.raises(ValidationError, match="requires type='editing'"):
        TaskSpec(
            id="bad_editing",
            track="track3_editing",
            dataset_id="dataset",
            type="generation",
            prompt_id="prompt",
            parser_id="parser",
            metrics=("edit_success",),
            primary_metric="edit_success",
            max_output_tokens=16,
        )


def test_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskSpec.model_validate(
            {
                "id": "task",
                "track": "track1_general",
                "dataset_id": "dataset",
                "type": "generation",
                "prompt_id": "prompt",
                "parser_id": "parser",
                "metrics": ["accuracy"],
                "primary_metric": "accuracy",
                "max_output_tokens": 16,
                "typo_field": True,
            }
        )


def test_result_status_contract() -> None:
    valid = ResultSpec(
        run_id="run",
        config_hash="a" * 64,
        sample_id="sample-1",
        model_id="mock",
        dataset_id="dataset",
        task_id="task",
        status="success",
        raw_output="answer",
        parsed_output="answer",
        metrics={"accuracy": 1.0},
    )
    assert valid.metrics["accuracy"] == 1.0

    with pytest.raises(ValidationError, match="failed results require failure_class"):
        ResultSpec(
            run_id="run",
            config_hash="a" * 64,
            sample_id="sample-2",
            model_id="mock",
            dataset_id="dataset",
            task_id="task",
            status="failure",
        )


def test_result_rejects_non_finite_metric() -> None:
    with pytest.raises(ValidationError):
        ResultSpec(
            run_id="run",
            config_hash="a" * 64,
            sample_id="sample-1",
            model_id="mock",
            dataset_id="dataset",
            task_id="task",
            status="success",
            metrics={"score": math.inf},
        )


def test_registry_supports_direct_and_decorator_registration() -> None:
    registry: Registry[object] = Registry("test component")
    direct = object()
    assert registry.register("direct", direct) is direct

    @registry.register("decorated")
    def decorated() -> str:
        return "ok"

    assert registry.get("direct") is direct
    assert registry.get("decorated") is decorated
    assert registry.names() == ("decorated", "direct")

    with pytest.raises(DuplicateRegistrationError):
        registry.register("direct", object())
    with pytest.raises(UnknownRegistrationError, match="available"):
        registry.get("missing")
    with pytest.raises(RegistryError, match="lowercase slug"):
        registry.register("Not Valid", object())
