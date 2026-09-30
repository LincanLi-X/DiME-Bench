"""Shared AR/dLLM decoding budget validation."""

from __future__ import annotations

from dataclasses import dataclass


class DecodingBudgetError(ValueError):
    """Raised when a decoding budget cannot be executed as declared."""


@dataclass(frozen=True)
class DiffusionBudget:
    """Resolved fixed-canvas diffusion budget."""

    generation_length: int
    steps: int
    block_length: int
    block_count: int
    steps_per_block: int


def resolve_diffusion_budget(
    generation_length: int,
    steps: int,
    block_length: int | None = None,
) -> DiffusionBudget:
    """Validate the divisibility constraints used by LLaDA block decoding."""
    if generation_length <= 0:
        raise DecodingBudgetError("generation_length must be positive")
    if steps <= 0:
        raise DecodingBudgetError("steps must be positive")
    resolved_block = block_length or generation_length
    if resolved_block <= 0 or resolved_block > generation_length:
        raise DecodingBudgetError(
            "block_length must be positive and no larger than generation_length"
        )
    if generation_length % resolved_block != 0:
        raise DecodingBudgetError("generation_length must be divisible by block_length")
    block_count = generation_length // resolved_block
    if steps % block_count != 0:
        raise DecodingBudgetError("steps must be divisible by the number of blocks")
    return DiffusionBudget(
        generation_length=generation_length,
        steps=steps,
        block_length=resolved_block,
        block_count=block_count,
        steps_per_block=steps // block_count,
    )


def allocate_transfer_counts(mask_count: int, steps: int) -> tuple[int, ...]:
    """Evenly allocate token reveals across reverse-process steps."""
    if mask_count < 0:
        raise DecodingBudgetError("mask_count cannot be negative")
    if steps <= 0:
        raise DecodingBudgetError("steps must be positive")
    base, remainder = divmod(mask_count, steps)
    return tuple(base + (1 if index < remainder else 0) for index in range(steps))
