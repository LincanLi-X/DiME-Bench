"""No-stemming ROUGE-L F1 compatible with the standard LCS definition."""

from __future__ import annotations

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


def _lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
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


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute sentence-level ROUGE-L F1 with beta=1."""
    predicted = word_tokens(prediction)
    expected = word_tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = _lcs_length(predicted, expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


class RougeLEvaluator(StandardEvaluator):
    """Return the best no-stemming ROUGE-L F1 over references."""

    METRIC_ID = "rouge_l"

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "tokenizer": "nfkc_alphanumeric_v1",
            "use_stemmer": False,
            "beta": 1,
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        prediction = prediction_text(result)
        scores = [rouge_l_f1(prediction, ref) for ref in references_for(sample)]
        return max(scores), {
            "reference_scores": cast(JsonValue, scores),
            "aggregation": "max",
        }


__all__ = ["RougeLEvaluator", "rouge_l_f1"]
