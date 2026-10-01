from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dimebench.reporting import ResultSubmission, load_leaderboard_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "results/releases/v1.0.0"


def test_committed_submission_satisfies_frozen_v1_schema() -> None:
    submission = ResultSubmission.model_validate_json(
        (RELEASE_ROOT / "result-submission.json").read_text(encoding="utf-8")
    )
    assert submission.benchmark_version == "1.0.0"
    assert submission.summary.run_root == "reference-runs"
    assert len(submission.summary.datasets) == 8
    assert len(submission.summary.models) == 2


def test_submission_rejects_tampered_embedded_summary() -> None:
    payload = json.loads(
        (RELEASE_ROOT / "result-submission.json").read_text(encoding="utf-8")
    )
    payload["summary"]["models"][0]["benchmark_score"] = 0.999
    with pytest.raises(ValidationError, match="summary_sha256"):
        ResultSubmission.model_validate(payload)


def test_leaderboard_rejects_unversioned_bare_summary() -> None:
    with pytest.raises(ValidationError):
        load_leaderboard_summary(RELEASE_ROOT / "benchmark-summary.json")
