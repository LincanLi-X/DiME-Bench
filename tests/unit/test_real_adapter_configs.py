from __future__ import annotations

from pathlib import Path

from dimebench.config import load_model_config
from dimebench.models import create_adapter
from dimebench.models.autoregressive import HuggingFaceCausalLMAdapter
from dimebench.models.diffusion import LLaDAAdapter
from dimebench.schemas import ModelRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_huggingface_model_config_constructs_without_loading_weights() -> None:
    spec = load_model_config(
        PROJECT_ROOT / "configs/models/autoregressive/llama3_8b_instruct.yaml"
    )
    adapter = create_adapter(spec)
    assert isinstance(adapter, HuggingFaceCausalLMAdapter)
    assert adapter.is_loaded is False

    class PlainTokenizer:
        chat_template = None

    request = ModelRequest(
        request_id="infill",
        sample_id="infill",
        task_id="infill",
        mode="infill",
        prompt="Fill the missing text.",
        prefix="left boundary",
        suffix="right boundary",
        max_output_tokens=8,
    )
    rendered = adapter._format_prompt(request, PlainTokenizer())
    assert "left boundary" in rendered
    assert "right boundary" in rendered
    assert "Return only the missing middle span" in rendered


def test_llada_model_config_constructs_without_loading_weights() -> None:
    spec = load_model_config(
        PROJECT_ROOT / "configs/models/diffusion/llada_8b_instruct.yaml"
    )
    adapter = create_adapter(spec)
    assert isinstance(adapter, LLaDAAdapter)
    assert adapter.is_loaded is False
