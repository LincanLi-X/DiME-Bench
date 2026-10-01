from __future__ import annotations

from dimebench.artifacts import MetricStore
from dimebench.evaluators import evaluate_sample
from dimebench.evaluators.standard import TokenF1Evaluator
from tests.standard_metric_fixtures import infilling_sample, result_for


def test_sample_to_dataset_aggregation_retains_failure_zero(tmp_path) -> None:
    successful_sample = infilling_sample("newly discovered comet")
    failed_sample = successful_sample.model_copy(
        update={"sample_id": "fixture.test.infilling-failed"}
    )
    success = evaluate_sample(
        result_for(successful_sample, "newly comet"),
        successful_sample,
        [TokenF1Evaluator()],
    )
    failure = evaluate_sample(
        result_for(failed_sample, None, status="failure"),
        failed_sample,
        [TokenF1Evaluator()],
    )
    store = MetricStore(tmp_path / "sample_metrics.jsonl", "step10", "a" * 64)
    store.append(success)
    store.append(failure)
    summary = store.summarize()

    assert summary.metrics["token_f1"].sample_count == 2
    assert summary.metrics["token_f1"].value == 0.4
    assert summary.metrics["token_f1"].sample_ids == (
        "fixture.test.infilling",
        "fixture.test.infilling-failed",
    )
