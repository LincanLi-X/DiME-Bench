"""Strict, versioned prompt-template schemas and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from string import Formatter
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from dimebench.artifacts.hashing import hash_json
from dimebench.datasets.records import SampleKind
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.model import ModelFamily
from dimebench.schemas.task import TrackId


class PromptError(ValueError):
    """Raised when a prompt definition or rendering violates its contract."""


class PromptSemanticContract(StrictSchema):
    """Model-family-independent task meaning shared by paired templates."""

    instruction: str = Field(min_length=1)
    input_fields: tuple[str, ...] = Field(min_length=1)
    output_contract: str = Field(min_length=1)
    answer_visibility: Literal["hidden"] = "hidden"

    @model_validator(mode="after")
    def validate_input_fields(self) -> PromptSemanticContract:
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError("prompt semantic input_fields must be unique")
        if any(not field.isidentifier() for field in self.input_fields):
            raise ValueError("prompt semantic input_fields must be identifiers")
        return self


class PromptTemplateSpec(StrictSchema):
    """A semantic prompt contract with matched AR and diffusion templates."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    version: str = Field(min_length=1)
    track: TrackId
    sample_kind: SampleKind
    semantic: PromptSemanticContract
    templates: dict[ModelFamily, str]

    @model_validator(mode="after")
    def validate_templates(self) -> PromptTemplateSpec:
        expected_families = {"autoregressive", "diffusion"}
        if set(self.templates) != expected_families:
            raise ValueError(
                "prompt templates must define autoregressive and diffusion variants"
            )
        expected_fields = set(self.semantic.input_fields)
        formatter = Formatter()
        for family, template in self.templates.items():
            if not template.strip():
                raise ValueError(f"{family} prompt template cannot be empty")
            fields: set[str] = set()
            try:
                for _, field_name, format_spec, conversion in formatter.parse(template):
                    if field_name is None:
                        continue
                    if not field_name.isidentifier() or format_spec or conversion:
                        raise ValueError(
                            "prompt placeholders must be plain Python identifiers"
                        )
                    fields.add(field_name)
            except ValueError as exc:
                raise ValueError(f"invalid {family} prompt template: {exc}") from exc
            if fields != expected_fields:
                missing = sorted(expected_fields - fields)
                extra = sorted(fields - expected_fields)
                raise ValueError(
                    f"{family} template field mismatch; "
                    f"missing={missing}, extra={extra}"
                )
        return self

    @property
    def semantic_hash(self) -> str:
        """Hash the family-independent instruction and output contract."""
        return hash_json(self.semantic)

    def template_hash(self, family: ModelFamily) -> str:
        """Hash one versioned family-specific template."""
        return hash_json(
            {
                "id": self.id,
                "version": self.version,
                "track": self.track,
                "sample_kind": self.sample_kind,
                "semantic_hash": self.semantic_hash,
                "model_family": family,
                "template": self.templates[family],
            }
        )

    def render(self, family: ModelFamily, fields: Mapping[str, str]) -> str:
        """Render one prompt after validating its complete input field set."""
        expected = set(self.semantic.input_fields)
        missing = sorted(expected - set(fields))
        if missing:
            raise PromptError(f"prompt {self.id!r} is missing fields: {missing}")
        selected = {name: fields[name] for name in self.semantic.input_fields}
        if any(not isinstance(value, str) for value in selected.values()):
            raise PromptError("all rendered prompt fields must be strings")
        try:
            rendered = self.templates[family].format_map(selected).strip()
        except (KeyError, ValueError) as exc:
            raise PromptError(f"cannot render prompt {self.id!r}: {exc}") from exc
        if not rendered:
            raise PromptError(f"prompt {self.id!r} rendered an empty string")
        return rendered


def load_prompt_spec(path: str | Path) -> PromptTemplateSpec:
    """Load and validate one prompt YAML definition."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise PromptError(f"cannot read prompt definition {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PromptError(f"invalid YAML in prompt definition {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PromptError(f"prompt definition {source} must contain a mapping")
    try:
        return PromptTemplateSpec.model_validate(payload)
    except ValidationError as exc:
        raise PromptError(f"invalid prompt definition {source}: {exc}") from exc


__all__ = [
    "PromptError",
    "PromptSemanticContract",
    "PromptTemplateSpec",
    "load_prompt_spec",
]
