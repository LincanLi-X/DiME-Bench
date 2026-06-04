from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.models.base import BaseGenerator


class HFLocalGenerator(BaseGenerator):
    def __init__(self, model_cfg: dict[str, Any], generation_cfg: dict[str, Any] | None = None):
        self.model_cfg = model_cfg
        self.generation_cfg = generation_cfg or {}
        self.model_name = model_cfg["name"]
        self.model_path = _resolve_model_path(model_cfg)
        self.trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        self.use_chat_template = bool(model_cfg.get("use_chat_template", False))
        self.dtype = _dtype(model_cfg.get("dtype", "bfloat16"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loader = model_cfg.get("loader", "auto")
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.dtype,
            "device_map": model_cfg.get("device_map", "auto"),
        }
        if loader == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
            except Exception:
                self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)
        self.model.eval()

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        if self.model_cfg.get("family") == "dllm" and getattr(self.model.config, "mask_token_id", None) is not None:
            return self._generate_diffusion(prompts, max_new_tokens=max_new_tokens)
        outputs: list[str] = []
        batch_size = int(self.generation_cfg.get("batch_size", 1))
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            rendered = [self._render_prompt(prompt) for prompt in batch]
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.generation_cfg.get("max_input_tokens", 4096)),
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            input_width = encoded["input_ids"].shape[1]
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(self.generation_cfg.get("do_sample", False)),
                "temperature": float(self.generation_cfg.get("temperature", 0.0)) or None,
                "top_p": float(self.generation_cfg.get("top_p", 1.0)),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.model_cfg.get("family") == "dllm":
                gen_kwargs["use_cache"] = False
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            t0 = time.perf_counter()
            with torch.no_grad():
                generated = self.model.generate(**encoded, **gen_kwargs)
            _ = time.perf_counter() - t0
            for seq in generated:
                suffix_ids = seq[input_width:]
                outputs.append(self.tokenizer.decode(suffix_ids, skip_special_tokens=True).strip())
        return outputs

    def _generate_diffusion(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        outputs: list[str] = []
        batch_size = int(self.generation_cfg.get("batch_size", 1))
        steps = int(self.generation_cfg.get("steps", 64))
        mask_token_id = int(getattr(self.model.config, "mask_token_id"))
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            rendered = [self._render_prompt(prompt) for prompt in batch]
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.generation_cfg.get("max_input_tokens", 4096)),
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]
            mask_block = torch.full(
                (encoded["input_ids"].shape[0], max_new_tokens),
                mask_token_id,
                dtype=encoded["input_ids"].dtype,
                device=encoded["input_ids"].device,
            )
            input_ids = torch.cat([encoded["input_ids"], mask_block], dim=1)
            attention_mask = torch.cat(
                [
                    encoded["attention_mask"],
                    torch.ones_like(mask_block, dtype=encoded["attention_mask"].dtype),
                ],
                dim=1,
            )
            # Only appended answer positions are allowed to change.
            mutable_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mutable_mask[:, prompt_width:] = True

            with torch.no_grad():
                for step in range(max(1, steps)):
                    current_mask = (input_ids == mask_token_id) & mutable_mask
                    if not current_mask.any():
                        break
                    model_type = str(getattr(self.model.config, "model_type", "")).lower()
                    forward_attention_mask = None if model_type == "dream" else attention_mask
                    model_out = self.model(
                        input_ids=input_ids,
                        attention_mask=forward_attention_mask,
                        use_cache=False,
                    )
                    logits = model_out.logits
                    if model_type == "dream":
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                    probs = F.softmax(logits.float(), dim=-1)
                    confidence, candidates = probs.max(dim=-1)

                    # Avoid re-selecting mask/pad as generated content when possible.
                    candidates = candidates.masked_fill(candidates == mask_token_id, eos_token_id or pad_token_id)

                    remaining_by_row = current_mask.sum(dim=1)
                    remaining_steps = max(1, steps - step)
                    for row_idx in range(input_ids.shape[0]):
                        positions = torch.nonzero(current_mask[row_idx], as_tuple=False).flatten()
                        if positions.numel() == 0:
                            continue
                        num_transfer = max(1, int(torch.ceil(remaining_by_row[row_idx].float() / remaining_steps).item()))
                        num_transfer = min(num_transfer, positions.numel())
                        row_conf = confidence[row_idx, positions]
                        selected = positions[torch.topk(row_conf, k=num_transfer).indices]
                        input_ids[row_idx, selected] = candidates[row_idx, selected]

            for row_idx in range(input_ids.shape[0]):
                suffix = input_ids[row_idx, prompt_width:].tolist()
                cleaned = []
                for tok in suffix:
                    if tok in {eos_token_id, pad_token_id, mask_token_id}:
                        break
                    cleaned.append(tok)
                outputs.append(self.tokenizer.decode(cleaned, skip_special_tokens=True).strip())
        return outputs

    def _render_prompt(self, prompt: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return prompt

    def close(self) -> None:
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_model_path(model_cfg: dict[str, Any]) -> str:
    local_dir = Path(model_cfg["local_dir"])
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)
    return model_cfg["repo_id"]


def _dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return "auto"
