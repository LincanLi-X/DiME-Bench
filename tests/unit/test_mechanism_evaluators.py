from __future__ import annotations

import json
from pathlib import Path

import pytest

from dimebench.artifacts import SampleMetricRecord
from dimebench.evaluators import evaluate_sample
from dimebench.evaluators.mechanism import (
    EditSuccessEvaluator,
    OverEditRateEvaluator,
    PreservationScoreEvaluator,
    aggregate_reasoning_regimes,
    available_mechanism_metrics,
    stratify_infilling_metric,
)
from dimebench.evaluators.mechanism._editing import (
    lcs_length,
    token_edit_distance,
)
from tests.mechanism_metric_fixtures import (
    infilling_sample,
    localized_edit_sample,
    over_edit_sample,
)
from tests.standard_metric_fixtures import result_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mechanism_registry_covers_frozen_track_metrics() -> None:
    assert set(available_mechanism_metrics()) == {
        "boundary_consistency",
        "contradiction_rate",
        "edit_success",
        "preservation_score",
        "over_edit_rate",
        "contradiction_reduction_rate",
        "chain_average",
        "planning_average",
        "reasoning_regime_gap",
    }


def test_token_lcs_and_edit_distance_known_values() -> None:
    assert lcs_length(("a", "b", "c"), ("a", "x", "b", "c")) == 3
    assert token_edit_distance(("a", "b", "c"), ("a", "x", "y")) == 2
    assert token_edit_distance((), ("a", "b")) == 2


def test_reference_edit_success() -> None:
    sample = localized_edit_sample()
    evaluator = EditSuccessEvaluator()
    assert (
        evaluator.evaluate(result_for(sample, "Water freezes at 0 C."), sample).value
        == 1.0
    )
    assert (
        evaluator.evaluate(result_for(sample, "Water freezes at 20 C."), sample).value
        == 0.0
    )


def test_preservation_uses_declared_non_target_spans() -> None:
    sample = localized_edit_sample()
    outcome = PreservationScoreEvaluator().evaluate(
        result_for(sample, "Water freezes at 0 C."), sample
    )
    assert outcome.value == 1.0
    assert outcome.details["target_route"] == "declared_non_target_spans"
    assert outcome.details["non_target_token_count"] == 3


def test_preservation_falls_back_to_source_reference_diff() -> None:
    sample = over_edit_sample()
    outcome = PreservationScoreEvaluator().evaluate(result_for(sample, "a x y"), sample)
    assert outcome.value == pytest.approx(0.5)
    assert outcome.details["target_route"] == "source_reference_diff"


def test_preservation_without_non_target_content_is_null() -> None:
    sample = over_edit_sample().model_copy(
        update={
            "source_text": "wrong",
            "reference_edit": "right",
            "minimum_edit_distance": 1,
        }
    )
    result = result_for(sample, "right")
    outcome = PreservationScoreEvaluator().evaluate(result, sample)
    assert outcome.value is None
    assert outcome.eligible is False
    assert outcome.denominator == 0
    record = evaluate_sample(result, sample, [PreservationScoreEvaluator()])
    assert "preservation_score" not in record.metrics
    assert record.evaluator_metadata["preservation_score.eligible"] == "false"


def test_over_edit_rate_matches_frozen_formula() -> None:
    sample = over_edit_sample()
    evaluator = OverEditRateEvaluator()
    assert evaluator.evaluate(result_for(sample, "a x c"), sample).value == 0.0
    outcome = evaluator.evaluate(result_for(sample, "a x y"), sample)
    assert outcome.value == pytest.approx(0.5)
    assert outcome.details["actual_edit_distance"] == 2


def test_failed_mechanism_result_is_null_not_zero() -> None:
    sample = localized_edit_sample()
    outcome = PreservationScoreEvaluator().evaluate(
        result_for(sample, None, status="failure"), sample
    )
    assert outcome.value is None
    assert outcome.eligible is False
    assert outcome.details["reason"] == "result_not_successful"


def _metric_record(
    sample_id: str,
    *,
    value: float | None,
    status: str = "success",
) -> SampleMetricRecord:
    metrics = {} if value is None else {"token_f1": value}
    return SampleMetricRecord(
        run_id="step11",
        config_hash="a" * 64,
        sample_id=sample_id,
        model_id="fixture-model",
        dataset_id="fixture",
        task_id="fixture-task",
        prediction_hash="b" * 64,
        status=status,
        metrics=metrics,
        evaluator_metadata={
            "token_f1.version": "1.0.0",
            "token_f1.eligible": str(value is not None).lower(),
        },
    )


def test_span_stratification_reports_coverage_and_failures() -> None:
    short_ok = infilling_sample("fixture.test.short-ok")
    short_fail = infilling_sample("fixture.test.short-fail")
    long_ok = infilling_sample(
        "fixture.test.long-ok",
        span_length_bin="long",
        boundary_density="high",
    )
    samples = {sample.sample_id: sample for sample in (short_ok, short_fail, long_ok)}
    records = [
        _metric_record(short_ok.sample_id, value=0.8),
        _metric_record(short_fail.sample_id, value=None, status="failure"),
        _metric_record(long_ok.sample_id, value=0.4),
    ]
    by_length = stratify_infilling_metric(
        records,
        samples,
        "token_f1",
        group_by="span_length_bin",
    )
    assert by_length.groups["short"].value == pytest.approx(0.8)
    assert by_length.groups["short"].eligible_count == 1
    assert by_length.groups["short"].total_sample_count == 2
    assert by_length.groups["short"].coverage == pytest.approx(0.5)
    assert by_length.groups["short"].failure_count == 1
    assert by_length.groups["long"].value == pytest.approx(0.4)

    by_density = stratify_infilling_metric(
        records,
        samples,
        "token_f1",
        group_by="boundary_density",
    )
    assert set(by_density.groups) == {"low", "high"}


def test_reasoning_regime_averages_require_complete_pairs() -> None:
    summary = aggregate_reasoning_regimes(
        {
            "gsm8k_chain": 0.8,
            "math500_chain": 0.4,
            "winogrande_planning": 0.6,
            "pathstar_planning": 0.8,
        }
    )
    assert summary.chain_average == pytest.approx(0.6)
    assert summary.planning_average == pytest.approx(0.7)
    assert summary.reasoning_regime_gap == pytest.approx(0.1)
    assert summary.chain_coverage == 2
    assert summary.planning_coverage == 2

    partial = aggregate_reasoning_regimes({"gsm8k_chain": 0.8})
    assert partial.chain_average is None
    assert partial.planning_average is None
    assert partial.reasoning_regime_gap is None


def test_committed_cpu_golden_fixtures() -> None:
    cases = json.loads(
        (PROJECT_ROOT / "tests/fixtures/mechanism_metrics/golden.json").read_text()
    )
    actual = {
        "preservation_score": PreservationScoreEvaluator()
        .evaluate(
            result_for(localized_edit_sample(), "Water freezes at 0 C."),
            localized_edit_sample(),
        )
        .value,
        "over_edit_rate": OverEditRateEvaluator()
        .evaluate(result_for(over_edit_sample(), "a x y"), over_edit_sample())
        .value,
        "edit_success": EditSuccessEvaluator()
        .evaluate(
            result_for(localized_edit_sample(), "Water freezes at 0 C."),
            localized_edit_sample(),
        )
        .value,
        "reasoning_regime_gap": aggregate_reasoning_regimes(
            {
                "gsm8k_chain": 0.8,
                "math500_chain": 0.4,
                "winogrande_planning": 0.6,
                "pathstar_planning": 0.8,
            }
        ).reasoning_regime_gap,
    }
    for case in cases:
        assert actual[case["metric"]] == pytest.approx(case["expected"])
