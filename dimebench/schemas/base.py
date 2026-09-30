"""Shared schema primitives for DiME-Bench configuration and results."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

RawText = Annotated[str, StringConstraints(strip_whitespace=False)]


class StrictSchema(BaseModel):
    """Base class that rejects unknown fields and produces stable models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )
