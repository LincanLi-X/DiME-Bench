"""Versioned, tamper-evident result submissions for public leaderboards."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dimebench.artifacts.hashing import hash_json
from dimebench.schemas.base import StrictSchema
from dimebench.summarizers import BenchmarkSummary


class ResultSubmission(StrictSchema):
    """One immutable v1 benchmark summary accepted by the leaderboard."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["dime_bench_v1"] = "dime_bench_v1"
    benchmark_version: Literal["1.0.0"] = "1.0.0"
    submission_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    )
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: BenchmarkSummary

    @model_validator(mode="after")
    def validate_summary_identity(self) -> ResultSubmission:
        if self.summary.benchmark_id != self.benchmark_id:
            raise ValueError("submission and summary benchmark IDs do not match")
        observed = hash_json(self.summary)
        if observed != self.summary_sha256:
            raise ValueError(
                "summary_sha256 does not match the canonical embedded summary"
            )
        return self


def create_result_submission(
    summary: BenchmarkSummary,
    *,
    submission_id: str,
) -> ResultSubmission:
    """Seal a benchmark summary into the frozen v1 submission schema."""
    return ResultSubmission(
        submission_id=submission_id,
        summary_sha256=hash_json(summary),
        summary=summary,
    )


__all__ = ["ResultSubmission", "create_result_submission"]
