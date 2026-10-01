from __future__ import annotations

from pathlib import Path

from dimebench.summarizers import (
    build_benchmark_summary,
    build_comparisons,
    evaluate_run,
)
from tests.step14_fixtures import make_sealed_run


def test_sample_to_benchmark_summary_is_traceable(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    ar_run = make_sealed_run(
        runs,
        run_id="fixture-ar",
        family="autoregressive",
        value=0.25,
        model_id="fixture-ar",
    )
    make_sealed_run(
        runs,
        run_id="fixture-dllm",
        family="diffusion",
        value=0.75,
        model_id="fixture-dllm",
    )

    evaluation = evaluate_run(ar_run)
    summary = build_benchmark_summary(runs)
    comparisons = build_comparisons(summary)

    assert evaluation.status == "passed"
    assert evaluation.recomputation_matches is True
    assert len(summary.datasets) == 2
    assert len(summary.tracks) == 2
    assert len(summary.models) == 2
    assert summary.models[0].benchmark_score is not None
    evidence = summary.datasets[0].metrics["accuracy"]
    assert evidence.sample_count == 2
    assert len(evidence.sample_ids_hash) == 64
    assert len(evidence.source_sha256) == 64
    assert len(comparisons.comparisons) == 1
    assert comparisons.comparisons[0].raw_delta == 0.5
    assert comparisons.comparisons[0].oriented_delta == 0.5
