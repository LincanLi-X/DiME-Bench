"""Dataset configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from dimebench.schemas.base import StrictSchema

DatasetSource = Literal["huggingface", "local", "builtin"]


class DatasetSpec(StrictSchema):
    """A frozen dataset selection used by a benchmark task."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    source: DatasetSource
    name_or_path: str = Field(min_length=1)
    subset: str | None = None
    revision: str | None = None
    split: str = Field(default="test", min_length=1)
    manifest_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    sample_limit: int | None = Field(default=None, gt=0)
    shuffle: bool = False
    seed: int = Field(default=0, ge=0)
