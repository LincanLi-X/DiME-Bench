"""Model adapter registration and configuration-driven construction."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from dimebench.models.base import ModelAdapter
from dimebench.registry import Registry
from dimebench.schemas.model import ModelSpec

ModelAdapterType = type[ModelAdapter]
MODEL_ADAPTER_REGISTRY: Registry[ModelAdapterType] = Registry("model adapter")


def register_adapter(name: str) -> Callable[[ModelAdapterType], ModelAdapterType]:
    """Register an adapter class under a configuration-facing slug."""
    decorator = MODEL_ADAPTER_REGISTRY.register(name)
    return decorator


def load_builtin_adapters() -> None:
    """Import built-in implementations so their registrations execute."""
    importlib.import_module("dimebench.models.autoregressive.huggingface")
    importlib.import_module("dimebench.models.diffusion.llada")
    importlib.import_module("dimebench.models.mock")


def create_adapter(spec: ModelSpec) -> ModelAdapter:
    """Construct, but do not load, the adapter selected by a model spec."""
    load_builtin_adapters()
    adapter_type = MODEL_ADAPTER_REGISTRY.get(spec.adapter)
    return adapter_type(spec)
