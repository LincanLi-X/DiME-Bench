"""Runner contracts and retry-safe subprocess execution."""

from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from dimebench.artifacts.hashing import write_json_atomic
from dimebench.artifacts.provenance import utc_now
from dimebench.schemas.base import StrictSchema


class ResourceRequest(StrictSchema):
    """Per-job resources used for placement and admission checks."""

    gpu_count: int = Field(default=1, ge=0, le=1)
    gpu_memory_gb: float = Field(default=0.0, ge=0.0)
    cpu_count: int = Field(default=1, gt=0)
    memory_gb: float = Field(default=1.0, gt=0.0)
    batch_size: int = Field(default=1, gt=0)
    estimated_cost: float = Field(default=1.0, gt=0.0)


class RetrySettings(StrictSchema):
    """Job-level retry policy; attempts resume existing job artifacts."""

    max_retries: int = Field(default=1, ge=0)
    backoff_seconds: float = Field(default=0.0, ge=0.0)
    timeout_seconds: float | None = Field(default=None, gt=0.0)


class BenchmarkJob(StrictSchema):
    """One isolated model × task command with a private output directory."""

    job_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    model_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    command: tuple[str, ...] = Field(min_length=1)
    work_dir: Path
    output_dir: Path
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    environment: dict[str, str] = Field(default_factory=dict)

    @property
    def partition_key(self) -> tuple[str, str]:
        return self.model_id, self.task_id

    @model_validator(mode="after")
    def validate_paths_and_environment(self) -> BenchmarkJob:
        if not self.work_dir.is_dir():
            raise ValueError(f"job work_dir is not a directory: {self.work_dir}")
        if any(not key or "=" in key for key in self.environment):
            raise ValueError("environment keys must be non-empty names without '='")
        return self


class JobAttempt(StrictSchema):
    """One subprocess attempt with durable stdout/stderr evidence."""

    attempt: int = Field(gt=0)
    worker_id: int = Field(ge=0)
    gpu_id: int | None = Field(default=None, ge=0)
    returncode: int
    started_at: datetime
    finished_at: datetime
    stdout_path: Path
    stderr_path: Path


class JobExecution(StrictSchema):
    """Terminal state of one benchmark job."""

    job_id: str
    model_id: str
    task_id: str
    status: Literal["completed", "failed"]
    worker_id: int = Field(ge=0)
    gpu_id: int | None = Field(default=None, ge=0)
    output_dir: Path
    attempts: tuple[JobAttempt, ...]


class RunnerReport(StrictSchema):
    """Deterministic job ordering plus operational runner outcomes."""

    schema_version: Literal["1.0"] = "1.0"
    runner: str
    status: Literal["completed", "failed"]
    worker_count: int = Field(gt=0)
    started_at: datetime
    finished_at: datetime
    jobs: tuple[JobExecution, ...]

    @property
    def completed_job_ids(self) -> tuple[str, ...]:
        return tuple(job.job_id for job in self.jobs if job.status == "completed")


class BaseRunner(ABC):
    """Common validation, subprocess execution, logging, and retries."""

    RUNNER_NAME = "base"

    def __init__(
        self,
        *,
        log_dir: str | Path,
        retry: RetrySettings | None = None,
    ) -> None:
        self.log_dir = Path(log_dir).resolve()
        self.retry = retry or RetrySettings()

    @staticmethod
    def validate_jobs(jobs: tuple[BenchmarkJob, ...]) -> None:
        if not jobs:
            raise ValueError("a runner requires at least one job")
        job_ids = [job.job_id for job in jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("runner job_id values must be unique")
        keys = [job.partition_key for job in jobs]
        if len(set(keys)) != len(keys):
            raise ValueError("each model × task pair must have exactly one job")
        outputs = [job.output_dir.resolve() for job in jobs]
        if len(set(outputs)) != len(outputs):
            raise ValueError("jobs cannot share an output directory")

    def _execute_job(
        self,
        job: BenchmarkJob,
        *,
        worker_id: int,
        gpu_id: int | None,
    ) -> JobExecution:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job_log_dir = self.log_dir / f"worker-{worker_id:02d}" / job.job_id
        job_log_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[JobAttempt] = []
        maximum_attempts = self.retry.max_retries + 1
        returncode = 1
        for attempt in range(1, maximum_attempts + 1):
            stdout_path = job_log_dir / f"attempt-{attempt:02d}.out"
            stderr_path = job_log_dir / f"attempt-{attempt:02d}.err"
            environment = os.environ.copy()
            environment.update(job.environment)
            environment["DIMEBENCH_JOB_ID"] = job.job_id
            environment["DIMEBENCH_WORKER_ID"] = str(worker_id)
            environment["DIMEBENCH_OUTPUT_DIR"] = str(job.output_dir.resolve())
            if gpu_id is not None:
                environment["CUDA_VISIBLE_DEVICES"] = self._device_token(gpu_id)
            started_at = utc_now()
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                try:
                    completed = subprocess.run(
                        list(job.command),
                        cwd=job.work_dir,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        check=False,
                        timeout=self.retry.timeout_seconds,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    returncode = 124
                    stderr.write("runner timeout expired\n")
            attempts.append(
                JobAttempt(
                    attempt=attempt,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    returncode=returncode,
                    started_at=started_at,
                    finished_at=utc_now(),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
            if returncode == 0:
                break
            if attempt < maximum_attempts and self.retry.backoff_seconds:
                time.sleep(self.retry.backoff_seconds)
        return JobExecution(
            job_id=job.job_id,
            model_id=job.model_id,
            task_id=job.task_id,
            status="completed" if returncode == 0 else "failed",
            worker_id=worker_id,
            gpu_id=gpu_id,
            output_dir=job.output_dir,
            attempts=tuple(attempts),
        )

    @staticmethod
    def _device_token(gpu_id: int) -> str:
        """Map an allocation-local GPU index back to its scheduler token."""
        allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
        if allocated is None:
            return str(gpu_id)
        tokens = [token.strip() for token in allocated.split(",") if token.strip()]
        if gpu_id >= len(tokens):
            raise ValueError(
                f"GPU {gpu_id} is outside CUDA_VISIBLE_DEVICES={allocated!r}"
            )
        return tokens[gpu_id]

    def _report(
        self,
        executions: tuple[JobExecution, ...],
        *,
        worker_count: int,
        started_at: datetime,
    ) -> RunnerReport:
        ordered = tuple(sorted(executions, key=lambda item: item.job_id))
        return RunnerReport(
            runner=self.RUNNER_NAME,
            status=(
                "completed"
                if all(job.status == "completed" for job in ordered)
                else "failed"
            ),
            worker_count=worker_count,
            started_at=started_at,
            finished_at=utc_now(),
            jobs=ordered,
        )

    def write_report(self, path: str | Path, report: RunnerReport) -> None:
        write_json_atomic(path, report)

    @abstractmethod
    def run(self, jobs: tuple[BenchmarkJob, ...]) -> RunnerReport:
        """Execute all jobs and return a terminal report."""


__all__ = [
    "BaseRunner",
    "BenchmarkJob",
    "JobAttempt",
    "JobExecution",
    "ResourceRequest",
    "RetrySettings",
    "RunnerReport",
]
