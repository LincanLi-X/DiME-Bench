from __future__ import annotations

import pytest

from dimebench.decoding import (
    DecodingBudgetError,
    allocate_transfer_counts,
    finish_reason,
    resolve_diffusion_budget,
    truncate_at_stop,
)
from dimebench.decoding.diffusion import (
    CanvasLayout,
    highest_confidence_positions,
    linear_remaining_ratio,
    normalize_mask_policy,
    normalize_schedule,
)
from dimebench.decoding.diffusion.sampler import LLaDASamplingConfig


def test_diffusion_budget_matches_official_block_constraints() -> None:
    budget = resolve_diffusion_budget(128, 128, 32)
    assert budget.block_count == 4
    assert budget.steps_per_block == 32
    assert allocate_transfer_counts(5, 3) == (2, 2, 1)

    with pytest.raises(DecodingBudgetError, match="divisible"):
        resolve_diffusion_budget(127, 128, 32)


def test_stopping_uses_earliest_marker() -> None:
    stopped = truncate_at_stop("answer<END>later###", ("###", "<END>"))
    assert stopped.text == "answer"
    assert stopped.matched_sequence == "<END>"
    assert finish_reason(stopped=True, generated_tokens=3, maximum_tokens=8) == "stop"
    assert (
        finish_reason(stopped=False, generated_tokens=8, maximum_tokens=8) == "length"
    )


def test_confidence_selection_is_stable() -> None:
    positions = highest_confidence_positions(
        [0.1, 0.9, 0.9, 0.2],
        [True, True, True, False],
        2,
    )
    assert positions == (1, 2)
    assert normalize_mask_policy("low-confidence") == "confidence_based"
    assert normalize_schedule("linear") == "linear"


def test_canvas_and_schedule_helpers() -> None:
    layout = CanvasLayout(left_length=4, generated_length=8, right_length=3)
    assert layout.generated_slice == slice(4, 12)
    assert layout.total_length == 15
    assert linear_remaining_ratio(2, 4) == 0.5


def test_llada_sampling_config_validates_without_loading_torch() -> None:
    config = LLaDASamplingConfig(
        generation_length=128,
        steps=128,
        block_length=32,
        unmasking_strategy="confidence_based",
    )
    assert config.resolved_budget().steps_per_block == 32

    with pytest.raises(ValueError, match="unsupported diffusion schedule"):
        LLaDASamplingConfig(
            generation_length=8,
            steps=8,
            schedule="cosine",
        ).resolved_budget()
