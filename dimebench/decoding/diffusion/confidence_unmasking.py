"""Confidence-based reveal policy shared by sampler tests and adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

MaskPolicy = Literal["confidence_based", "random"]


def normalize_mask_policy(policy: str) -> MaskPolicy:
    """Map public policy aliases to the sampler's canonical names."""
    normalized = policy.strip().lower().replace("-", "_")
    aliases = {
        "confidence": "confidence_based",
        "low_confidence": "confidence_based",
        "confidence_based": "confidence_based",
        "random": "random",
    }
    try:
        return cast(MaskPolicy, aliases[normalized])
    except KeyError as exc:
        raise ValueError(
            f"unsupported mask policy {policy!r}; supported: confidence_based, random"
        ) from exc


def highest_confidence_positions(
    confidences: Sequence[float],
    candidates: Sequence[bool],
    count: int,
) -> tuple[int, ...]:
    """Select candidate positions by descending confidence and stable index."""
    if len(confidences) != len(candidates):
        raise ValueError("confidences and candidates must have equal lengths")
    if count < 0:
        raise ValueError("count cannot be negative")
    ranked = sorted(
        (index for index, is_candidate in enumerate(candidates) if is_candidate),
        key=lambda index: (-confidences[index], index),
    )
    if count > len(ranked):
        raise ValueError("count exceeds the number of candidate positions")
    return tuple(ranked[:count])
