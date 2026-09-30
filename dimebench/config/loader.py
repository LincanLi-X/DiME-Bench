"""YAML loading and validation entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dimebench.config.overrides import OverrideError, apply_overrides
from dimebench.config.validation import ConfigValidationError, validate_run_spec
from dimebench.schemas.model import ModelSpec
from dimebench.schemas.run import RunSpec


class ConfigError(ValueError):
    """Public error raised for unreadable or invalid configuration files."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping without executing arbitrary constructors."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"config {config_path} must contain a YAML mapping")
    return dict(payload)


def load_config(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> RunSpec:
    """Load, override, parse, and cross-validate one run configuration."""
    payload = load_yaml(path)
    try:
        resolved = apply_overrides(payload, overrides)
        spec = RunSpec.model_validate(resolved)
        return validate_run_spec(spec)
    except (OverrideError, ValidationError, ConfigValidationError) as exc:
        raise ConfigError(f"invalid config {Path(path)}: {exc}") from exc


def load_model_config(path: str | Path) -> ModelSpec:
    """Load and validate a standalone model configuration."""
    payload = load_yaml(path)
    try:
        return ModelSpec.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(f"invalid model config {Path(path)}: {exc}") from exc
