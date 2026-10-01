"""Single-node data-parallel subprocess runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from dimebench.artifacts.provenance import utc_now
from dimebench.runners.base import (
    BaseRunner,
    BenchmarkJob,
    JobExecution,
    RunnerReport,
)
from dimebench.runners.partitioner import JobPartition, partition_jobs
from dimebench.runners.resource_manager import ResourceManager, RuntimeConfig


class DistributedRunner(BaseRunner):
    """Run one isolated subprocess queue per visible GPU."""

    RUNNER_NAME = "distributed"

    def __init__(
        self,
        runtime: RuntimeConfig,
        *,
        log_dir: str,
        resources: ResourceManager | None = None,
    ) -> None:
        if runtime.backend != "distributed":
            raise ValueError("DistributedRunner requires backend='distributed'")
        super().__init__(log_dir=log_dir, retry=runtime.retry)
        self.runtime = runtime
        self.resources = resources or ResourceManager()

    def _run_partition(self, partition: JobPartition) -> tuple[JobExecution, ...]:
        gpu_id = (
            self.runtime.gpu_ids[partition.worker_id] if self.runtime.gpu_ids else None
        )
        return tuple(
            self._execute_job(
                job,
                worker_id=partition.worker_id,
                gpu_id=gpu_id,
            )
            for job in partition.jobs
        )

    def run(self, jobs: tuple[BenchmarkJob, ...]) -> RunnerReport:
        self.validate_jobs(jobs)
        self.resources.validate(jobs, self.runtime)
        started_at = utc_now()
        partitions = partition_jobs(jobs, self.runtime.worker_count)
        executions: list[JobExecution] = []
        with ThreadPoolExecutor(max_workers=self.runtime.worker_count) as pool:
            futures = [pool.submit(self._run_partition, item) for item in partitions]
            for future in futures:
                executions.extend(future.result())
        return self._report(
            tuple(executions),
            worker_count=self.runtime.worker_count,
            started_at=started_at,
        )


__all__ = ["DistributedRunner"]
