"""Typed registries for pluggable DiME-Bench components."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from types import MappingProxyType
from typing import Generic, TypeVar, overload

T = TypeVar("T")
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class RegistryError(LookupError):
    """Base error for registry operations."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a name is registered more than once."""


class UnknownRegistrationError(RegistryError):
    """Raised when a requested name has not been registered."""


class Registry(Generic[T]):
    """Small explicit registry with direct and decorator registration APIs."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    @overload
    def register(self, name: str, item: T, *, replace: bool = False) -> T: ...

    @overload
    def register(
        self,
        name: str,
        item: None = None,
        *,
        replace: bool = False,
    ) -> Callable[[T], T]: ...

    def register(
        self,
        name: str,
        item: T | None = None,
        *,
        replace: bool = False,
    ) -> T | Callable[[T], T]:
        """Register an item directly or return a registration decorator."""
        self._validate_name(name)

        def add(candidate: T) -> T:
            if name in self._items and not replace:
                raise DuplicateRegistrationError(
                    f"{self.kind} {name!r} is already registered"
                )
            self._items[name] = candidate
            return candidate

        if item is None:
            return add
        return add(item)

    def get(self, name: str) -> T:
        """Return a registered item with an actionable missing-name error."""
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise UnknownRegistrationError(
                f"unknown {self.kind} {name!r}; available: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._items))

    def items(self) -> MappingProxyType[str, T]:
        """Return a read-only view of registered entries."""
        return MappingProxyType(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def _validate_name(self, name: str) -> None:
        if not _NAME_PATTERN.fullmatch(name):
            raise RegistryError(
                f"invalid {self.kind} name {name!r}; use lowercase slug syntax"
            )


MODEL_REGISTRY: Registry[object] = Registry("model")
DATASET_REGISTRY: Registry[object] = Registry("dataset")
TASK_REGISTRY: Registry[object] = Registry("task")
METRIC_REGISTRY: Registry[object] = Registry("metric")

__all__ = [
    "DATASET_REGISTRY",
    "METRIC_REGISTRY",
    "MODEL_REGISTRY",
    "TASK_REGISTRY",
    "DuplicateRegistrationError",
    "Registry",
    "RegistryError",
    "UnknownRegistrationError",
]
