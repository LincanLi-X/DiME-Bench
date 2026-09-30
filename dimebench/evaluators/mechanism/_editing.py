"""Shared versioned token alignment for editing diagnostics."""

from __future__ import annotations

from difflib import SequenceMatcher

from dimebench.datasets import EditingRecord
from dimebench.evaluators.standard._text import word_tokens


def lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Return the token longest-common-subsequence length."""
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def token_edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Return insertion/deletion/substitution Levenshtein distance."""
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_token != right_token)
            current.append(
                min(
                    previous[right_index] + 1,
                    current[-1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def non_target_source_tokens(sample: EditingRecord) -> tuple[str, ...]:
    """Resolve gold non-target content from spans or the source/reference diff."""
    if sample.non_target_spans:
        text = " ".join(
            sample.source_text[span.start : span.end]
            for span in sample.non_target_spans
        )
        return word_tokens(text)

    source = word_tokens(sample.source_text)
    reference = word_tokens(sample.reference_edit)
    matcher = SequenceMatcher(a=source, b=reference, autojunk=False)
    preserved: list[str] = []
    for tag, source_start, source_end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            preserved.extend(source[source_start:source_end])
    return tuple(preserved)


__all__ = ["lcs_length", "non_target_source_tokens", "token_edit_distance"]
