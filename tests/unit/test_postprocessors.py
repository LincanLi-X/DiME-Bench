from __future__ import annotations

from typing import Any

import pytest

from dimebench.postprocessors import (
    PostprocessConfigurationError,
    create_parser,
    postprocess_prediction,
)
from tests.postprocessor_fixtures import (
    load_golden_cases,
    prediction_for,
    sample_for,
    task_for,
)


@pytest.mark.parametrize("case", load_golden_cases(), ids=lambda case: case["name"])
def test_postprocessor_golden_cases(case: dict[str, Any]) -> None:
    sample = sample_for(case["sample"])
    task = task_for(case["parser_id"], case["task_type"])
    prediction = prediction_for(case, sample, task)
    result = postprocess_prediction(prediction, sample, task)

    assert result.status == case["expected_status"]
    assert result.raw_output == case["raw_output"]
    assert result.metadata["parser"] == {
        "id": case["parser_id"],
        "version": "1.0.0",
    }
    assert result.metadata["prediction_hash"] == prediction.record_hash
    if "expected_parsed" in case:
        assert result.parsed_output == case["expected_parsed"]
    if "expected_failure" in case:
        assert result.failure_class == case["expected_failure"]
    if "expected_labels" in case:
        assert result.metadata["failure_labels"] == case["expected_labels"]


def test_parser_version_mismatch_is_rejected() -> None:
    with pytest.raises(PostprocessConfigurationError, match="requires version"):
        create_parser("numeric_answer", "9.9.9")


def test_unknown_parser_is_rejected() -> None:
    with pytest.raises(PostprocessConfigurationError, match="unknown parser"):
        create_parser("not_registered", "1.0.0")
