from __future__ import annotations

import pytest
from pydantic import ValidationError

from dimebench.prompts import PromptTemplateSpec


def _payload() -> dict[str, object]:
    return {
        "id": "track2.test",
        "version": "1.0.0",
        "track": "track2_infilling",
        "sample_kind": "infilling",
        "semantic": {
            "instruction": "Fill the missing span.",
            "input_fields": ["prefix", "suffix"],
            "output_contract": "Return only the middle.",
        },
        "templates": {
            "autoregressive": "Prefix: {prefix}\nSuffix: {suffix}\nMiddle:",
            "diffusion": "<P>{prefix}<MASK><S>{suffix}<OUT>",
        },
    }


def test_template_pair_has_shared_semantics_and_stable_hashes() -> None:
    spec = PromptTemplateSpec.model_validate(_payload())
    fields = {"prefix": "left", "suffix": "right"}
    ar = spec.render("autoregressive", fields)
    dllm = spec.render("diffusion", fields)
    assert ar != dllm
    assert spec.semantic_hash == spec.semantic_hash
    assert spec.template_hash("autoregressive") != spec.template_hash("diffusion")
    assert spec.render("autoregressive", fields) == ar


def test_template_rejects_family_field_drift() -> None:
    payload = _payload()
    templates = payload["templates"]
    assert isinstance(templates, dict)
    templates["diffusion"] = "Only {prefix}"
    with pytest.raises(ValidationError, match="field mismatch"):
        PromptTemplateSpec.model_validate(payload)
