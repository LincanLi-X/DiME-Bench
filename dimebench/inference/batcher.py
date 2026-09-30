"""Deterministic ordered batching."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[tuple[T, ...]]:
    """Yield immutable, order-preserving batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield tuple(items[start : start + batch_size])


__all__ = ["iter_batches"]
