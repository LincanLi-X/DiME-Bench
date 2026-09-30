"""Reproducibility metadata collection without requiring GPU libraries."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from dimebench.schemas.base import StrictSchema
from dimebench.version import __version__


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class GitInfo(StrictSchema):
    """Git state associated with a benchmark run."""

    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    dirty: bool | None = None
    root: str | None = None


class HardwareInfo(StrictSchema):
    """Portable hardware inventory available without model loading."""

    machine: str
    processor: str
    cpu_count: int | None = Field(default=None, ge=1)
    memory_bytes: int | None = Field(default=None, ge=1)
    gpu_devices: tuple[str, ...] = ()


class EnvironmentInfo(StrictSchema):
    """Software, source, and hardware provenance for one run."""

    schema_version: str = "1.0"
    captured_at: datetime = Field(default_factory=utc_now)
    dimebench_version: str
    python_version: str
    python_implementation: str
    platform: str
    git: GitInfo
    hardware: HardwareInfo
    dependencies: dict[str, str]
    accelerator_environment: dict[str, str]


def _run_command(arguments: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_git_info(cwd: str | Path | None = None) -> GitInfo:
    """Collect the containing repository commit and dirty state, if any."""
    location = Path(cwd).resolve() if cwd is not None else Path.cwd()
    root = _run_command(["git", "rev-parse", "--show-toplevel"], location)
    if root is None:
        return GitInfo()
    commit = _run_command(["git", "rev-parse", "HEAD"], location)
    status = _run_command(["git", "status", "--porcelain"], location)
    return GitInfo(
        commit=commit if commit and len(commit) == 40 else None,
        dirty=None if status is None else bool(status),
        root=root,
    )


def collect_dependency_versions() -> dict[str, str]:
    """Return an alphabetically ordered snapshot of installed distributions."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions[name.lower()] = distribution.version
    return dict(sorted(versions.items()))


def _memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    return page_size * page_count


def _gpu_devices() -> tuple[str, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    output = _run_command(
        [executable, "--query-gpu=name,uuid", "--format=csv,noheader"]
    )
    if not output:
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def collect_environment(cwd: str | Path | None = None) -> EnvironmentInfo:
    """Collect a complete, privacy-limited reproducibility snapshot."""
    accelerator_variables = {
        key: value
        for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        if (value := os.environ.get(key)) is not None
    }
    return EnvironmentInfo(
        dimebench_version=__version__,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        git=collect_git_info(cwd),
        hardware=HardwareInfo(
            machine=platform.machine(),
            processor=platform.processor(),
            cpu_count=os.cpu_count(),
            memory_bytes=_memory_bytes(),
            gpu_devices=_gpu_devices(),
        ),
        dependencies=collect_dependency_versions(),
        accelerator_environment=accelerator_variables,
    )
