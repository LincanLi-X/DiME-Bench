"""Model-independent output stopping and finish-reason helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["stop", "length", "completed"]


@dataclass(frozen=True)
class StoppedText:
    """Text truncated at the earliest declared stop marker."""

    text: str
    finish_reason: FinishReason
    matched_sequence: str | None = None


def truncate_at_stop(text: str, stop_sequences: tuple[str, ...]) -> StoppedText:
    """Truncate at the earliest stop sequence, breaking ties by declaration order."""
    earliest: tuple[int, int, str] | None = None
    for order, sequence in enumerate(stop_sequences):
        if not sequence:
            raise ValueError("stop sequences cannot be empty")
        position = text.find(sequence)
        if position >= 0:
            candidate = (position, order, sequence)
            if earliest is None or candidate < earliest:
                earliest = candidate
    if earliest is None:
        return StoppedText(text=text, finish_reason="completed")
    position, _, sequence = earliest
    return StoppedText(
        text=text[:position],
        finish_reason="stop",
        matched_sequence=sequence,
    )


def finish_reason(
    *,
    stopped: bool,
    generated_tokens: int,
    maximum_tokens: int,
) -> FinishReason:
    """Resolve a portable finish reason from token and stopping state."""
    if stopped:
        return "stop"
    if generated_tokens >= maximum_tokens:
        return "length"
    return "completed"
