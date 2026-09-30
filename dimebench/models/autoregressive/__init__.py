"""Autoregressive model adapter interfaces."""

from dimebench.models.autoregressive.base import AutoregressiveModelAdapter
from dimebench.models.autoregressive.huggingface import HuggingFaceCausalLMAdapter

__all__ = ["AutoregressiveModelAdapter", "HuggingFaceCausalLMAdapter"]
