"""Unified model adapter API and built-in implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dimebench.models.base import AdapterState, ModelAdapter
from dimebench.models.capabilities import (
    AdapterContractError,
    AdapterDependencyError,
    AdapterError,
    AdapterStateError,
    Capability,
    UnsupportedCapabilityError,
)
from dimebench.models.registry import (
    MODEL_ADAPTER_REGISTRY,
    create_adapter,
    register_adapter,
)

if TYPE_CHECKING:
    from dimebench.models.mock import MockModelAdapter


def __getattr__(name: str) -> Any:
    """Lazily expose implementations without creating import-order cycles."""
    if name == "MockModelAdapter":
        from dimebench.models.mock import MockModelAdapter

        return MockModelAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_ADAPTER_REGISTRY",
    "AdapterContractError",
    "AdapterDependencyError",
    "AdapterError",
    "AdapterState",
    "AdapterStateError",
    "Capability",
    "MockModelAdapter",
    "ModelAdapter",
    "UnsupportedCapabilityError",
    "create_adapter",
    "register_adapter",
]
