"""Mechanism-aware diagnostics for DiME-Bench Tracks 2-4."""

from dimebench.evaluators.mechanism.boundary_consistency import (
    BoundaryConsistencyEvaluator,
    BoundaryPairBackend,
    TransformersPairClassifier,
)
from dimebench.evaluators.mechanism.contradiction_rate import (
    ContradictionBackend,
    ContradictionRateEvaluator,
    TransformersNLIBackend,
)
from dimebench.evaluators.mechanism.contradiction_reduction import (
    ContradictionReductionEvaluator,
)
from dimebench.evaluators.mechanism.edit_success import (
    EditSuccessChecker,
    EditSuccessEvaluator,
    ReferenceEditChecker,
)
from dimebench.evaluators.mechanism.over_edit_rate import OverEditRateEvaluator
from dimebench.evaluators.mechanism.preservation_score import (
    PreservationScoreEvaluator,
)
from dimebench.evaluators.mechanism.reasoning_regime_gap import (
    ReasoningRegimeSummary,
    aggregate_reasoning_regimes,
)
from dimebench.evaluators.mechanism.span_stratification import (
    StratificationResult,
    StratumAggregate,
    stratify_infilling_metric,
)

_MECHANISM_VERSIONS = {
    "boundary_consistency": BoundaryConsistencyEvaluator.VERSION,
    "contradiction_rate": ContradictionRateEvaluator.VERSION,
    "edit_success": EditSuccessEvaluator.VERSION,
    "preservation_score": PreservationScoreEvaluator.VERSION,
    "over_edit_rate": OverEditRateEvaluator.VERSION,
    "contradiction_reduction_rate": ContradictionReductionEvaluator.VERSION,
    "chain_average": "1.0.0",
    "planning_average": "1.0.0",
    "reasoning_regime_gap": "1.0.0",
}


def available_mechanism_metrics() -> dict[str, str]:
    """Return mechanism metric IDs and frozen implementation versions."""
    return dict(sorted(_MECHANISM_VERSIONS.items()))


__all__ = [
    "BoundaryConsistencyEvaluator",
    "BoundaryPairBackend",
    "ContradictionBackend",
    "ContradictionRateEvaluator",
    "ContradictionReductionEvaluator",
    "EditSuccessChecker",
    "EditSuccessEvaluator",
    "OverEditRateEvaluator",
    "PreservationScoreEvaluator",
    "ReasoningRegimeSummary",
    "ReferenceEditChecker",
    "StratificationResult",
    "StratumAggregate",
    "TransformersNLIBackend",
    "TransformersPairClassifier",
    "aggregate_reasoning_regimes",
    "available_mechanism_metrics",
    "stratify_infilling_metric",
]
