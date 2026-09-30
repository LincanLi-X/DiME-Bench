"""Base class for left-to-right autoregressive model adapters."""

from __future__ import annotations

from abc import ABC

from dimebench.models.base import ModelAdapter


class AutoregressiveModelAdapter(ModelAdapter, ABC):
    """Family-constrained base for autoregressive adapters."""

    MODEL_FAMILY = "autoregressive"
