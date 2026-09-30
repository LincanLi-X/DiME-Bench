"""Dotted-path CLI overrides for YAML configurations."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OverrideError(ValueError):
    """Raised when a CLI override is malformed or cannot be applied."""


def parse_override(expression: str) -> tuple[tuple[str, ...], Any]:
    """Parse ``key.path=value`` using YAML semantics for the value."""
    if "=" not in expression:
        raise OverrideError(f"override {expression!r} must use the form key.path=value")
    raw_path, raw_value = expression.split("=", 1)
    path = tuple(raw_path.split("."))
    if not path or any(not _KEY_PATTERN.fullmatch(part) for part in path):
        raise OverrideError(f"invalid override path: {raw_path!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise OverrideError(f"invalid YAML override value in {expression!r}") from exc
    return path, value


def apply_overrides(
    config: Mapping[str, Any],
    overrides: Sequence[str],
) -> dict[str, Any]:
    """Return a deep-copied config with dotted-path overrides applied."""
    resolved: dict[str, Any] = copy.deepcopy(dict(config))
    for expression in overrides:
        path, value = parse_override(expression)
        target: dict[str, Any] = resolved
        for part in path[:-1]:
            child = target.get(part)
            if child is None:
                child = {}
                target[part] = child
            if not isinstance(child, dict):
                joined = ".".join(path[:-1])
                raise OverrideError(f"override target {joined!r} is not a mapping")
            target = child
        target[path[-1]] = value
    return resolved
