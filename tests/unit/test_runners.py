from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dimebench.runners import (
    BaseRunner,
    BenchmarkJob,
    DistributedRunner,
    LocalRunner,
    ResourceRequest,
    RetrySettings,
    RuntimeConfig,
    SlurmOptions,
    SlurmRunner,
    load_runtime_config,
    partition_jobs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "scripts/step13_worker_fixture.py"


def _jobs(root: Path, *, fail_first: bool = False) -> tuple[BenchmarkJob, ...]:
    jobs = []
    for index in range(4):
        output = root / f"job-{index}"
        command = [
            sys.executable,
            str(FIXTURE),
            "--output-dir",
            str(output),
            "--job-id",
            f"model-a-task-{index}",
            "--score",
            str(index / 4),
        ]
        if fail_first and index == 0:
            command.append("--fail-first")
        jobs.append(
            BenchmarkJob(
                job_id=f"model-a-task-{index}",
                model_id="model-a",
                task_id=f"task-{index}",
                command=tuple(command),
                work_dir=PROJECT_ROOT,
                output_dir=output,
                resources=ResourceRequest(
                    gpu_count=0,
                    batch_size=2,
                    estimated_cost=float(index + 1),
                ),
            )
        )
    return tuple(jobs)


def _scores(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/score.json"))
    }


def test_runtime_configs_are_strict_and_loadable() -> None:
    local = load_runtime_config(PROJECT_ROOT / "configs/runtime/local_1gpu.yaml")
    distributed = load_runtime_config(PROJECT_ROOT / "configs/runtime/b200_4gpu.yaml")
    assert (local.backend, local.worker_count, local.gpu_ids) == ("local", 1, (0,))
    assert distributed.worker_count == 4
    assert distributed.gpu_ids == (0, 1, 2, 3)


def test_scheduler_gpu_tokens_are_mapped_from_local_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alpha,GPU-beta")
    assert BaseRunner._device_token(0) == "GPU-alpha"
    assert BaseRunner._device_token(1) == "GPU-beta"
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        BaseRunner._device_token(2)


def test_partitioner_is_deterministic_and_balances_cost(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path / "outputs")
    first = partition_jobs(jobs, 2)
    second = partition_jobs(tuple(reversed(jobs)), 2)
    assert first == second
    observed = sorted(
        sum((list(item.jobs) for item in first), []),
        key=lambda job: job.job_id,
    )
    assert observed == sorted(jobs, key=lambda job: job.job_id)
    assert [item.estimated_cost for item in first] == [5.0, 5.0]


def test_single_and_multi_worker_scores_match_and_retry_is_safe(
    tmp_path: Path,
) -> None:
    retry = RetrySettings(max_retries=1)
    local_runtime = RuntimeConfig(
        backend="local",
        worker_count=1,
        gpu_ids=(),
        enforce_hardware=False,
        retry=retry,
    )
    multi_runtime = RuntimeConfig(
        backend="distributed",
        worker_count=2,
        gpu_ids=(),
        enforce_hardware=False,
        retry=retry,
    )
    local_root = tmp_path / "single"
    multi_root = tmp_path / "multi"
    local = LocalRunner(local_runtime, log_dir=str(tmp_path / "logs-single"))
    distributed = DistributedRunner(
        multi_runtime,
        log_dir=str(tmp_path / "logs-multi"),
    )

    local_report = local.run(_jobs(local_root))
    multi_report = distributed.run(_jobs(multi_root, fail_first=True))

    assert local_report.status == "completed"
    assert multi_report.status == "completed"
    assert _scores(local_root) == _scores(multi_root)
    retried = next(job for job in multi_report.jobs if job.job_id.endswith("task-0"))
    assert [attempt.returncode for attempt in retried.attempts] == [42, 0]
    checkpoint = multi_root / "job-0/checkpoint.json"
    before = checkpoint.read_bytes()
    resumed = distributed.run(_jobs(multi_root, fail_first=True))
    assert resumed.status == "completed"
    assert checkpoint.read_bytes() == before


def test_jobs_cannot_share_output_directories(tmp_path: Path) -> None:
    jobs = list(_jobs(tmp_path / "outputs"))
    jobs[1] = jobs[1].model_copy(update={"output_dir": jobs[0].output_dir})
    runtime = RuntimeConfig(
        backend="local",
        worker_count=1,
        gpu_ids=(),
        enforce_hardware=False,
    )
    with pytest.raises(ValueError, match="share an output directory"):
        LocalRunner(runtime, log_dir=str(tmp_path / "logs")).run(tuple(jobs))


def test_slurm_backend_generates_isolated_gpu_scripts(tmp_path: Path) -> None:
    job = _jobs(tmp_path / "outputs")[0].model_copy(
        update={
            "resources": ResourceRequest(
                gpu_count=1,
                gpu_memory_gb=80,
                cpu_count=16,
                memory_gb=64,
            )
        }
    )
    runner = SlurmRunner(
        SlurmOptions(
            account="example.group",
            qos="example.group",
            partition="hpg-rtx6000",
            time_limit="02:00:00",
            modules=("conda",),
        ),
        script_dir=tmp_path / "scripts",
        log_dir=tmp_path / "logs",
    )
    generated = runner.write_scripts((job,))
    text = generated[0].script_path.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert str(job.output_dir) in text
