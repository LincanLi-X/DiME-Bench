"""Standard task metrics frozen by the DiME-Bench v1 protocol."""

from __future__ import annotations

from collections.abc import Callable

from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.evaluators.standard.accuracy import AccuracyEvaluator
from dimebench.evaluators.standard.bertscore import (
    BERTScoreBackend,
    BERTScoreEvaluator,
    BERTScorePackageBackend,
)
from dimebench.evaluators.standard.exact_match import ExactMatchEvaluator
from dimebench.evaluators.standard.ifeval import (
    BuiltinIFEvalChecker,
    IFEvalChecker,
    IFEvalEvaluator,
)
from dimebench.evaluators.standard.option_accuracy import OptionAccuracyEvaluator
from dimebench.evaluators.standard.pass_at_k import (
    CodeExecutor,
    LocalPythonExecutor,
    PassAt1Evaluator,
)
from dimebench.evaluators.standard.path_validity import PathValidityEvaluator
from dimebench.evaluators.standard.rouge_l import RougeLEvaluator
from dimebench.evaluators.standard.token_f1 import TokenF1Evaluator

EvaluatorFactory = Callable[[], StandardEvaluator]

_STANDARD_FACTORIES: dict[str, EvaluatorFactory] = {
    "accuracy": AccuracyEvaluator,
    "normalized_exact_match": ExactMatchEvaluator,
    "pass_at_1": PassAt1Evaluator,
    "instruction_following_rate": IFEvalEvaluator,
    "token_f1": TokenF1Evaluator,
    "rouge_l": RougeLEvaluator,
    "bertscore_f1": BERTScoreEvaluator,
    "option_accuracy": OptionAccuracyEvaluator,
    "path_validity": PathValidityEvaluator,
}


def available_standard_evaluators() -> dict[str, str]:
    """Return standard metric IDs and frozen implementation versions."""
    return {
        metric_id: factory().VERSION
        for metric_id, factory in sorted(_STANDARD_FACTORIES.items())
    }


def create_standard_evaluator(metric_id: str) -> StandardEvaluator:
    """Create a default standard evaluator by configuration-facing ID."""
    try:
        return _STANDARD_FACTORIES[metric_id]()
    except KeyError as exc:
        known = ", ".join(sorted(_STANDARD_FACTORIES))
        raise EvaluationError(
            f"unknown standard metric {metric_id!r}; available metrics: {known}"
        ) from exc


__all__ = [
    "AccuracyEvaluator",
    "BERTScoreBackend",
    "BERTScoreEvaluator",
    "BERTScorePackageBackend",
    "BuiltinIFEvalChecker",
    "CodeExecutor",
    "ExactMatchEvaluator",
    "IFEvalChecker",
    "IFEvalEvaluator",
    "LocalPythonExecutor",
    "OptionAccuracyEvaluator",
    "PassAt1Evaluator",
    "PathValidityEvaluator",
    "RougeLEvaluator",
    "TokenF1Evaluator",
    "available_standard_evaluators",
    "create_standard_evaluator",
]
