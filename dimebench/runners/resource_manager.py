"""Runtime configuration and GPU capacity validation."""

from __future__ import annotations

import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from dimebench.config import load_yaml
from dimebench.runners.base import BenchmarkJob, RetrySettings
from dimebench.schemas.base import StrictSchema


class ResourceError(RuntimeError):
    """Raised when requested worker resources are unavailable or inconsistent."""


class GPUDevice(StrictSchema):
    """One allocation-local CUDA device."""

    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    total_memory_gb: float = Field(gt=0.0)


class RuntimeConfig(StrictSchema):
    """Configuration for local or local multi-GPU execution."""

    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["local", "distributed"]
    worker_count: int = Field(gt=0)
    gpu_ids: tuple[int, ...]
    enforce_hardware: bool = True
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @model_validator(mode="after")
    def validate_worker_devices(self) -> RuntimeConfig:
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("runtime gpu_ids must be unique")
        if self.backend == "local" and self.worker_count != 1:
            raise ValueError("local runtime requires worker_count=1")
        if self.gpu_ids and len(self.gpu_ids) < self.worker_count:
            raise ValueError("runtime requires at least one gpu_id per worker")
        return self


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load one strict runtime YAML configuration."""
    return RuntimeConfig.model_validate(load_yaml(path))


class ResourceManager:
    """Discover visible GPUs and validate a job set before execution."""

    def __init__(self, devices: tuple[GPUDevice, ...] | None = None) -> None:
        self._devices = devices

    def discover_gpus(self) -> tuple[GPUDevice, ...]:
        if self._devices is not None:
            return self._devices
        torch_devices = self._discover_gpus_with_torch()
        if torch_devices is not None:
            self._devices = torch_devices
            return self._devices
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ResourceError("cannot discover GPUs with nvidia-smi") from exc
        devices: list[GPUDevice] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",", maxsplit=2)]
            if len(parts) != 3:
                raise ResourceError(f"unexpected nvidia-smi row: {line!r}")
            index, name, memory_mib = parts
            devices.append(
                GPUDevice(
                    index=int(index),
                    name=name,
                    total_memory_gb=float(memory_mib) / 1024.0,
                )
            )
        self._devices = tuple(devices)
        return self._devices

    @staticmethod
    def _discover_gpus_with_torch() -> tuple[GPUDevice, ...] | None:
        """Prefer CUDA's allocation-local numbering when PyTorch is installed."""
        try:
            torch: Any = import_module("torch")
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return tuple(
            GPUDevice(
                index=index,
                name=str(torch.cuda.get_device_name(index)),
                total_memory_gb=(
                    float(torch.cuda.get_device_properties(index).total_memory)
                    / (1024.0**3)
                ),
            )
            for index in range(torch.cuda.device_count())
        )

    def validate(
        self,
        jobs: tuple[BenchmarkJob, ...],
        runtime: RuntimeConfig,
    ) -> None:
        """Reject unsupported placement or insufficient per-GPU memory."""
        if not runtime.enforce_hardware:
            return
        requires_gpu = any(job.resources.gpu_count for job in jobs)
        if not requires_gpu:
            return
        devices = {device.index: device for device in self.discover_gpus()}
        missing = [gpu_id for gpu_id in runtime.gpu_ids if gpu_id not in devices]
        if missing:
            raise ResourceError(f"requested GPU IDs are not visible: {missing}")
        if len(runtime.gpu_ids) < runtime.worker_count:
            raise ResourceError("not enough GPU IDs for configured workers")
        maximum_required = max(job.resources.gpu_memory_gb for job in jobs)
        insufficient = [
            gpu_id
            for gpu_id in runtime.gpu_ids[: runtime.worker_count]
            if devices[gpu_id].total_memory_gb < maximum_required
        ]
        if insufficient:
            raise ResourceError(
                f"GPUs {insufficient} have less than {maximum_required:g} GB"
            )


__all__ = [
    "GPUDevice",
    "ResourceError",
    "ResourceManager",
    "RuntimeConfig",
    "load_runtime_config",
]
