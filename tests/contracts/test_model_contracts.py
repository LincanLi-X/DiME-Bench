from __future__ import annotations

import pytest
from pydantic import ValidationError

from dimebench.schemas import ModelRequest, ModelResponse


def test_score_request_requires_unique_options() -> None:
    with pytest.raises(ValidationError, match="at least two options"):
        ModelRequest(
            request_id="request-1",
            sample_id="sample-1",
            task_id="task",
            mode="score_options",
            prompt="Choose an answer",
            max_output_tokens=1,
            options=("A",),
        )

    with pytest.raises(ValidationError, match="options must be unique"):
        ModelRequest(
            request_id="request-1",
            sample_id="sample-1",
            task_id="task",
            mode="score_options",
            prompt="Choose an answer",
            max_output_tokens=1,
            options=("A", "A"),
        )


def test_infilling_request_requires_both_boundaries() -> None:
    with pytest.raises(ValidationError, match="both prefix and suffix"):
        ModelRequest(
            request_id="request-1",
            sample_id="sample-1",
            task_id="task",
            mode="infill",
            prompt="Fill the middle",
            prefix="before",
            max_output_tokens=8,
        )


def test_response_success_and_error_states_are_exclusive() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ModelResponse(
            request_id="request-1",
            sample_id="sample-1",
            model_id="model",
            status="success",
            text="answer",
            option_scores={"A": 1.0},
            finish_reason="completed",
        )

    with pytest.raises(ValidationError, match="require error_type"):
        ModelResponse(
            request_id="request-1",
            sample_id="sample-1",
            model_id="model",
            status="error",
            finish_reason="error",
        )


def test_valid_error_response_preserves_failure_details() -> None:
    response = ModelResponse(
        request_id="request-1",
        sample_id="sample-1",
        model_id="model",
        status="error",
        finish_reason="error",
        error_type="RuntimeError",
        error_message="mock failure",
    )
    assert response.error_message == "mock failure"
