"""Single-worker local execution backend."""

from __future__ import annotations

from dimebench.artifacts.provenance import utc_now
from dimebench.runners.base import BaseRunner, BenchmarkJob, RunnerReport
from dimebench.runners.resource_manager import ResourceManager, RuntimeConfig


class LocalRunner(BaseRunner):
    """Execute model × task jobs sequentially on one optional GPU."""

    RUNNER_NAME = "local"

    def __init__(
        self,
        runtime: RuntimeConfig,
        *,
        log_dir: str,
        resources: ResourceManager | None = None,
    ) -> None:
        if runtime.backend != "local":
            raise ValueError("LocalRunner requires backend='local'")
        super().__init__(log_dir=log_dir, retry=runtime.retry)
        self.runtime = runtime
        self.resources = resources or ResourceManager()

    def run(self, jobs: tuple[BenchmarkJob, ...]) -> RunnerReport:
        self.validate_jobs(jobs)
        self.resources.validate(jobs, self.runtime)
        started_at = utc_now()
        gpu_id = self.runtime.gpu_ids[0] if self.runtime.gpu_ids else None
        executions = tuple(
            self._execute_job(job, worker_id=0, gpu_id=gpu_id) for job in jobs
        )
        return self._report(executions, worker_count=1, started_at=started_at)


__all__ = ["LocalRunner"]
