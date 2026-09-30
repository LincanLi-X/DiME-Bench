"""Multiset token-overlap F1."""

from __future__ import annotations

from collections import Counter
from typing import cast

from pydantic import JsonValue

from dimebench.datasets import SampleRecord
from dimebench.evaluators.base import StandardEvaluator
from dimebench.evaluators.standard._text import (
    prediction_text,
    references_for,
    word_tokens,
)
from dimebench.schemas.result import ResultSpec


def token_f1(prediction: str, reference: str) -> float:
    """Compute bag-of-tokens F1 after the frozen v1 tokenizer."""
    predicted = word_tokens(prediction)
    expected = word_tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


class TokenF1Evaluator(StandardEvaluator):
    """Return the best token-overlap F1 over frozen references."""

    METRIC_ID = "token_f1"

    def configuration(self) -> dict[str, JsonValue]:
        return {**super().configuration(), "tokenizer": "nfkc_alphanumeric_v1"}

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        prediction = prediction_text(result)
        scores = [token_f1(prediction, ref) for ref in references_for(sample)]
        value = max(scores)
        return value, {
            "reference_scores": cast(JsonValue, scores),
            "aggregation": "max",
        }


__all__ = ["TokenF1Evaluator", "token_f1"]
