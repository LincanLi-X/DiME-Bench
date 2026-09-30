from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dimebench.decoding.diffusion.sampler import (  # noqa: E402
    LLaDASampler,
    LLaDASamplingConfig,
)


class TinyMaskPredictor:
    """Always predicts token 3, providing a deterministic sampler contract test."""

    def __call__(self, input_ids, attention_mask=None):
        del attention_mask
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 8), dtype=torch.float32)
        logits[..., 3] = 10.0
        return SimpleNamespace(logits=logits)


def test_llada_sampler_reveals_only_editable_canvas() -> None:
    mask_token_id = 7
    input_ids = torch.tensor([[1, mask_token_id, mask_token_id, 2]])
    editable = torch.tensor([[False, True, True, False]])
    output = LLaDASampler(torch).sample(
        TinyMaskPredictor(),
        input_ids,
        editable,
        LLaDASamplingConfig(
            generation_length=2,
            steps=2,
            block_length=2,
            mask_token_id=mask_token_id,
            unmasking_strategy="confidence_based",
        ),
    )

    assert output.token_ids.tolist() == [[1, 3, 3, 2]]
    assert output.forward_passes == 2
