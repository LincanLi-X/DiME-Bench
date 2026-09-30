"""LLaDA-compatible masked diffusion sampler with lazy Torch imports."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from dimebench.decoding.budgets import (
    DiffusionBudget,
    allocate_transfer_counts,
    resolve_diffusion_budget,
)
from dimebench.decoding.diffusion.confidence_unmasking import (
    MaskPolicy,
    normalize_mask_policy,
)
from dimebench.decoding.diffusion.schedules import normalize_schedule
from dimebench.models.capabilities import AdapterDependencyError


@dataclass(frozen=True)
class LLaDASamplingConfig:
    """Sampling parameters matching the official fixed-canvas LLaDA process."""

    generation_length: int
    steps: int
    block_length: int | None = None
    temperature: float = 0.0
    guidance_scale: float = 0.0
    schedule: str = "linear"
    unmasking_strategy: str = "confidence_based"
    mask_token_id: int = 126336
    seed: int = 0

    def resolved_budget(self) -> DiffusionBudget:
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale cannot be negative")
        if self.mask_token_id < 0:
            raise ValueError("mask_token_id cannot be negative")
        normalize_schedule(self.schedule)
        normalize_mask_policy(self.unmasking_strategy)
        return resolve_diffusion_budget(
            self.generation_length,
            self.steps,
            self.block_length,
        )


@dataclass(frozen=True)
class DiffusionSamplingOutput:
    """Final token canvas plus auditable compute counters."""

    token_ids: Any
    forward_passes: int


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise AdapterDependencyError(
            "LLaDA sampling requires Torch; install dime-bench[diffusion]"
        ) from exc


def _add_gumbel_noise(
    logits: Any,
    temperature: float,
    generator: Any,
    torch: Any,
) -> Any:
    if temperature == 0:
        return logits
    high_precision_logits = logits.to(torch.float64)
    uniform = torch.rand(
        high_precision_logits.shape,
        device=high_precision_logits.device,
        dtype=torch.float64,
        generator=generator,
    )
    noise = (-torch.log(uniform)) ** temperature
    return high_precision_logits.exp() / noise


class LLaDASampler:
    """Reference masked sampler derived from the official LLaDA algorithm."""

    def __init__(self, torch_module: Any | None = None) -> None:
        self._torch = torch_module

    @property
    def torch(self) -> Any:
        if self._torch is None:
            self._torch = _load_torch()
        return self._torch

    def sample(
        self,
        model: Any,
        input_ids: Any,
        editable_mask: Any,
        config: LLaDASamplingConfig,
        *,
        attention_mask: Any | None = None,
    ) -> DiffusionSamplingOutput:
        """Denoise masked editable positions while freezing all other tokens."""
        torch = self.torch
        budget = config.resolved_budget()
        policy: MaskPolicy = normalize_mask_policy(config.unmasking_strategy)
        tokens = input_ids.clone()
        editable = editable_mask.to(dtype=torch.bool, device=tokens.device)
        if tokens.ndim != 2 or editable.shape != tokens.shape:
            raise ValueError(
                "input_ids and editable_mask must share shape [batch, length]"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens)
        else:
            attention_mask = attention_mask.to(device=tokens.device)

        editable_positions: list[list[int]] = []
        for batch_index in range(tokens.shape[0]):
            positions = (
                torch.nonzero(editable[batch_index], as_tuple=False).flatten().tolist()
            )
            if len(positions) != budget.generation_length:
                raise ValueError(
                    "each sample must expose generation_length editable positions"
                )
            if any(
                int(tokens[batch_index, position].item()) != config.mask_token_id
                for position in positions
            ):
                raise ValueError(
                    "all editable positions must initially contain mask_token_id"
                )
            editable_positions.append(positions)

        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(config.seed)
        fixed_context = (~editable) & attention_mask.to(dtype=torch.bool)
        forward_passes = 0

        with torch.no_grad():
            for block_index in range(budget.block_count):
                block_mask = torch.zeros_like(editable, dtype=torch.bool)
                start = block_index * budget.block_length
                end = start + budget.block_length
                for batch_index, positions in enumerate(editable_positions):
                    block_mask[batch_index, positions[start:end]] = True
                transfers = allocate_transfer_counts(
                    budget.block_length,
                    budget.steps_per_block,
                )
                for reveal_count in transfers:
                    current_masks = tokens == config.mask_token_id
                    candidates = current_masks & block_mask
                    if config.guidance_scale > 0:
                        unconditional = tokens.clone()
                        unconditional[fixed_context] = config.mask_token_id
                        combined = torch.cat([tokens, unconditional], dim=0)
                        combined_attention = torch.cat(
                            [attention_mask, attention_mask], dim=0
                        )
                        logits = model(
                            combined,
                            attention_mask=combined_attention,
                        ).logits
                        conditional_logits, unconditional_logits = torch.chunk(
                            logits, 2, dim=0
                        )
                        logits = unconditional_logits + (
                            config.guidance_scale + 1.0
                        ) * (conditional_logits - unconditional_logits)
                    else:
                        logits = model(
                            tokens,
                            attention_mask=attention_mask,
                        ).logits
                    forward_passes += 1
                    noisy_logits = _add_gumbel_noise(
                        logits,
                        config.temperature,
                        generator,
                        torch,
                    )
                    predictions = torch.argmax(noisy_logits, dim=-1)
                    predictions = torch.where(current_masks, predictions, tokens)

                    if policy == "confidence_based":
                        probabilities = torch.softmax(logits, dim=-1)
                        confidence = torch.gather(
                            probabilities,
                            dim=-1,
                            index=predictions.unsqueeze(-1),
                        ).squeeze(-1)
                    else:
                        confidence = torch.rand(
                            predictions.shape,
                            device=predictions.device,
                            generator=generator,
                        )
                    confidence = torch.where(
                        candidates,
                        confidence,
                        torch.full_like(confidence, float("-inf")),
                    )
                    if reveal_count == 0:
                        continue
                    reveal = torch.zeros_like(candidates, dtype=torch.bool)
                    for batch_index in range(confidence.shape[0]):
                        _, indices = torch.topk(
                            confidence[batch_index],
                            k=reveal_count,
                        )
                        reveal[batch_index, indices] = True
                    tokens[reveal] = predictions[reveal]

        if bool(((tokens == config.mask_token_id) & editable).any().item()):
            raise RuntimeError("diffusion sampling ended with unrevealed mask tokens")
        return DiffusionSamplingOutput(
            token_ids=tokens,
            forward_passes=forward_passes,
        )
