"""Discrete diffusion language model adapter interfaces."""

from dimebench.models.diffusion.base import DiffusionModelAdapter
from dimebench.models.diffusion.llada import LLaDAAdapter

__all__ = ["DiffusionModelAdapter", "LLaDAAdapter"]
