from __future__ import annotations

import pytest

from dimebench.artifacts import MetricStore
from dimebench.evaluators import evaluate_sample
from dimebench.evaluators.mechanism import PreservationScoreEvaluator
from tests.mechanism_metric_fixtures import localized_edit_sample
from tests.standard_metric_fixtures import result_for


def test_mechanism_aggregate_reports_null_and_zero_coverage(tmp_path) -> None:
    sample = localized_edit_sample()
    failed = result_for(sample, None, status="failure")
    record = evaluate_sample(failed, sample, [PreservationScoreEvaluator()])
    store = MetricStore(tmp_path / "sample_metrics.jsonl", "step10", "a" * 64)
    store.append(record)
    aggregate = store.summarize().metrics["preservation_score"]

    assert aggregate.value is None
    assert aggregate.sample_count == 0
    assert aggregate.total_sample_count == 1
    assert aggregate.coverage == 0.0
    assert aggregate.failure_count == 1
    assert aggregate.skipped_count == 0
    assert aggregate.sample_ids == ()


def test_mechanism_aggregate_uses_only_eligible_values_but_keeps_denominator(
    tmp_path,
) -> None:
    success_sample = localized_edit_sample()
    failed_sample = success_sample.model_copy(
        update={"sample_id": "fixture.test.localized-edit-failed"}
    )
    success = evaluate_sample(
        result_for(success_sample, "Water freezes at 0 C."),
        success_sample,
        [PreservationScoreEvaluator()],
    )
    failure = evaluate_sample(
        result_for(failed_sample, None, status="failure"),
        failed_sample,
        [PreservationScoreEvaluator()],
    )
    store = MetricStore(tmp_path / "sample_metrics.jsonl", "step10", "a" * 64)
    store.append(success)
    store.append(failure)
    aggregate = store.summarize().metrics["preservation_score"]

    assert aggregate.value == pytest.approx(1.0)
    assert aggregate.sample_count == 1
    assert aggregate.total_sample_count == 2
    assert aggregate.coverage == pytest.approx(0.5)
    assert aggregate.failure_count == 1
