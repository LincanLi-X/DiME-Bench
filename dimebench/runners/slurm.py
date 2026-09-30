"""Optional Slurm script generation and submission backend."""

from __future__ import annotations

import math
import shlex
import subprocess
from pathlib import Path

from pydantic import Field

from dimebench.runners.base import BaseRunner, BenchmarkJob
from dimebench.schemas.base import StrictSchema


class SlurmOptions(StrictSchema):
    """Cluster-level settings shared by generated job scripts."""

    account: str = Field(min_length=1)
    qos: str = Field(min_length=1)
    partition: str = Field(min_length=1)
    time_limit: str = Field(pattern=r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")
    python_environment: Path | None = None
    modules: tuple[str, ...] = ()


class SlurmSubmission(StrictSchema):
    """Generated script and optional scheduler identity."""

    job_id: str
    script_path: Path
    scheduler_job_id: str | None = None


class SlurmRunner:
    """Generate isolated sbatch files; submission is explicit and optional."""

    def __init__(
        self,
        options: SlurmOptions,
        *,
        script_dir: str | Path,
        log_dir: str | Path,
    ) -> None:
        self.options = options
        self.script_dir = Path(script_dir).resolve()
        self.log_dir = Path(log_dir).resolve()

    def render_script(self, job: BenchmarkJob) -> str:
        """Render one model × task job without sharing output files."""
        options = self.options
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job.job_id}",
            f"#SBATCH --account={options.account}",
            f"#SBATCH --qos={options.qos}",
            f"#SBATCH --partition={options.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={job.resources.cpu_count}",
            f"#SBATCH --mem={math.ceil(job.resources.memory_gb)}gb",
            f"#SBATCH --time={options.time_limit}",
            f"#SBATCH --output={self.log_dir}/{job.job_id}-%j.out",
            f"#SBATCH --error={self.log_dir}/{job.job_id}-%j.err",
        ]
        if job.resources.gpu_count:
            lines.append(f"#SBATCH --gpus={job.resources.gpu_count}")
        lines.extend(("", "set -euo pipefail", "", "module purge"))
        lines.extend(f"module load {shlex.quote(module)}" for module in options.modules)
        if options.python_environment is not None:
            binary = options.python_environment / "bin"
            lines.append(f"export PATH={shlex.quote(str(binary))}:$PATH")
        for key, value in sorted(job.environment.items()):
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.extend(
            (
                f"export DIMEBENCH_JOB_ID={shlex.quote(job.job_id)}",
                f"export DIMEBENCH_OUTPUT_DIR={shlex.quote(str(job.output_dir))}",
                f"mkdir -p {shlex.quote(str(job.output_dir))}",
                f"cd {shlex.quote(str(job.work_dir))}",
                shlex.join(job.command),
                "",
            )
        )
        return "\n".join(lines)

    def write_scripts(
        self,
        jobs: tuple[BenchmarkJob, ...],
    ) -> tuple[SlurmSubmission, ...]:
        BaseRunner.validate_jobs(jobs)
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        submissions = []
        for job in sorted(jobs, key=lambda item: item.job_id):
            path = self.script_dir / f"{job.job_id}.sbatch"
            path.write_text(self.render_script(job), encoding="utf-8")
            submissions.append(SlurmSubmission(job_id=job.job_id, script_path=path))
        return tuple(submissions)

    def submit(
        self,
        generated: tuple[SlurmSubmission, ...],
    ) -> tuple[SlurmSubmission, ...]:
        """Submit previously reviewed scripts and return scheduler job IDs."""
        submitted = []
        for item in generated:
            completed = subprocess.run(
                ["sbatch", "--parsable", str(item.script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            scheduler_job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
            submitted.append(
                item.model_copy(update={"scheduler_job_id": scheduler_job_id})
            )
        return tuple(submitted)


__all__ = ["SlurmOptions", "SlurmRunner", "SlurmSubmission"]
