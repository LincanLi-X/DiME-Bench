"""Top-level benchmark run schema and deterministic configuration hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from dimebench.schemas.base import StrictSchema
from dimebench.schemas.dataset import DatasetSpec
from dimebench.schemas.model import ModelSpec
from dimebench.schemas.task import TaskSpec


class DecodingSpec(StrictSchema):
    """Shared and mechanism-specific decoding controls."""

    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    do_sample: bool = False
    denoising_steps: int | None = Field(default=None, gt=0)
    unmasking_strategy: str | None = None
    schedule: str | None = None
    mask_policy: str | None = None


class RunSpec(StrictSchema):
    """Complete validated configuration for one model-task run."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["dime_bench_v1"] = "dime_bench_v1"
    run_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    seed: int = Field(default=0, ge=0)
    deterministic: bool = True
    batch_size: int = Field(default=1, gt=0)
    output_dir: Path = Path("outputs")
    model: ModelSpec
    dataset: DatasetSpec
    task: TaskSpec
    decoding: DecodingSpec
    tags: tuple[str, ...] = ()

    @property
    def config_hash(self) -> str:
        """Return a stable SHA-256 hash of the fully resolved configuration."""
        payload = self.model_dump(mode="json", exclude_none=False)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
