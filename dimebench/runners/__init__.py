"""Deterministic local, multi-GPU, and Slurm execution backends."""

from dimebench.runners.base import (
    BaseRunner,
    BenchmarkJob,
    JobExecution,
    ResourceRequest,
    RetrySettings,
    RunnerReport,
)
from dimebench.runners.distributed import DistributedRunner
from dimebench.runners.local import LocalRunner
from dimebench.runners.partitioner import JobPartition, partition_jobs
from dimebench.runners.resource_manager import (
    GPUDevice,
    ResourceError,
    ResourceManager,
    RuntimeConfig,
    load_runtime_config,
)
from dimebench.runners.slurm import SlurmOptions, SlurmRunner

__all__ = [
    "BaseRunner",
    "BenchmarkJob",
    "DistributedRunner",
    "GPUDevice",
    "JobExecution",
    "JobPartition",
    "LocalRunner",
    "ResourceError",
    "ResourceManager",
    "ResourceRequest",
    "RetrySettings",
    "RunnerReport",
    "RuntimeConfig",
    "SlurmOptions",
    "SlurmRunner",
    "load_runtime_config",
    "partition_jobs",
]
