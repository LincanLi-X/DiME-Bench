"""Model configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from dimebench.schemas.base import StrictSchema

ModelFamily = Literal["autoregressive", "diffusion"]
DType = Literal["float32", "float16", "bfloat16"]


class ModelSpec(StrictSchema):
    """Declarative model and adapter selection for one benchmark run."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    family: ModelFamily
    adapter: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    adapter_version: str = Field(default="1.0.0", min_length=1)
    name_or_path: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    dtype: DType = "bfloat16"
    device: str = Field(default="auto", min_length=1)
    trust_remote_code: bool = False
    capabilities: tuple[str, ...] = ()
    adapter_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
