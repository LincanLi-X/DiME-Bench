"""Generic Hugging Face causal-language-model adapter."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import JsonValue

from dimebench.decoding.stopping import finish_reason, truncate_at_stop
from dimebench.models.autoregressive.base import AutoregressiveModelAdapter
from dimebench.models.capabilities import (
    AdapterContractError,
    AdapterDependencyError,
    Capability,
)
from dimebench.models.registry import register_adapter
from dimebench.schemas.model import ModelSpec
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse, ModelUsage


def _mapping_option(
    options: Mapping[str, JsonValue],
    key: str,
) -> dict[str, Any]:
    value = options.get(key, {})
    if not isinstance(value, dict):
        raise AdapterContractError(f"adapter_kwargs.{key} must be a mapping")
    return dict(value)


@register_adapter("huggingface")
class HuggingFaceCausalLMAdapter(AutoregressiveModelAdapter):
    """Transformers-backed causal LM with greedy generation and option scoring."""

    ADAPTER_NAME = "huggingface"
    CAPABILITIES = frozenset(
        {
            Capability.GENERATE,
            Capability.SCORE_OPTIONS,
            Capability.INFILL,
            Capability.EDIT,
            Capability.REASONING,
        }
    )

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._resolved_model_revision: str = spec.revision
        self._resolved_tokenizer_revision: str = (
            spec.tokenizer_revision or spec.revision
        )

    def _load(self) -> None:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise AdapterDependencyError(
                "Hugging Face inference requires optional dependencies; "
                "install dime-bench[hf]"
            ) from exc

        tokenizer_name = self.spec.tokenizer_name_or_path or self.spec.name_or_path
        tokenizer_revision = self.spec.tokenizer_revision or self.spec.revision
        tokenizer_kwargs = _mapping_option(
            self.spec.adapter_kwargs,
            "tokenizer_load_kwargs",
        )
        tokenizer_kwargs.update(
            {
                "revision": tokenizer_revision,
                "trust_remote_code": self.spec.trust_remote_code,
            }
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            **tokenizer_kwargs,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise AdapterContractError(
                    "tokenizer defines neither pad_token_id nor eos_token_id"
                )
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model_kwargs = _mapping_option(
            self.spec.adapter_kwargs,
            "model_load_kwargs",
        )
        model_kwargs.update(
            {
                "revision": self.spec.revision,
                "trust_remote_code": self.spec.trust_remote_code,
                "torch_dtype": getattr(torch, self.spec.dtype),
            }
        )
        if self.spec.device == "auto":
            model_kwargs.setdefault("device_map", "auto")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.spec.name_or_path,
            **model_kwargs,
        )
        if self.spec.device != "auto":
            model = model.to(self.spec.device)
        model.eval()
        resolved_model_revision = getattr(model.config, "_commit_hash", None)
        if isinstance(resolved_model_revision, str) and resolved_model_revision:
            self._resolved_model_revision = resolved_model_revision
        tokenizer_init_kwargs = getattr(tokenizer, "init_kwargs", {})
        resolved_tokenizer_revision = (
            tokenizer_init_kwargs.get("_commit_hash")
            if isinstance(tokenizer_init_kwargs, dict)
            else None
        )
        if isinstance(resolved_tokenizer_revision, str) and resolved_tokenizer_revision:
            self._resolved_tokenizer_revision = resolved_tokenizer_revision
        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer

    def _close(self) -> None:
        self._model = None
        self._tokenizer = None
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._torch = None

    def _runtime(self) -> tuple[Any, Any, Any]:
        if self._torch is None or self._model is None or self._tokenizer is None:
            raise RuntimeError("Hugging Face adapter runtime is not loaded")
        return self._torch, self._model, self._tokenizer

    def _device(self, model: Any) -> Any:
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            return model.device

    def _format_prompt(self, request: ModelRequest, tokenizer: Any) -> str:
        use_chat = self.spec.adapter_kwargs.get("use_chat_template", True)
        if not isinstance(use_chat, bool):
            raise AdapterContractError(
                "adapter_kwargs.use_chat_template must be a boolean"
            )
        content = request.prompt
        if request.mode == "infill":
            content = (
                f"{request.prompt}\n\nPrefix:\n{request.prefix or ''}\n\n"
                f"Suffix:\n{request.suffix or ''}\n\n"
                "Return only the missing middle span."
            )
        if use_chat and getattr(tokenizer, "chat_template", None):
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        return content

    def _cuda_start(self, torch: Any, device: Any) -> float:
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        return time.perf_counter()

    def _cuda_finish(
        self, torch: Any, device: Any, started: float
    ) -> tuple[float, int]:
        peak_memory = 0
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        return time.perf_counter() - started, peak_memory

    def _generate(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        return tuple(self._generate_one(request) for request in requests)

    def _generate_one(self, request: ModelRequest) -> ModelResponse:
        torch, model, tokenizer = self._runtime()
        device = self._device(model)
        prompt = self._format_prompt(request, tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])

        generation_kwargs = _mapping_option(
            self.spec.adapter_kwargs,
            "generation_kwargs",
        )
        do_sample = request.metadata.get("do_sample", False)
        temperature = request.metadata.get("temperature", 0.0)
        if not isinstance(do_sample, bool):
            raise AdapterContractError("request metadata do_sample must be boolean")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise AdapterContractError(
                "request metadata temperature must be non-negative"
            )
        if not do_sample and float(temperature) != 0.0:
            raise AdapterContractError(
                "temperature must be zero when do_sample is false"
            )
        generation_kwargs.update(
            {
                "max_new_tokens": request.max_output_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        )
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generator = torch.Generator(device=device)
            generator.manual_seed(request.seed)
            generation_kwargs["generator"] = generator

        started = self._cuda_start(torch, device)
        with torch.inference_mode():
            output = model.generate(**encoded, **generation_kwargs)
        elapsed, peak_memory = self._cuda_finish(torch, device, started)
        sequences = output.sequences if hasattr(output, "sequences") else output
        generated_ids = sequences[0, input_tokens:]
        token_ids = [int(token) for token in generated_ids.tolist()]
        eos_ids = tokenizer.eos_token_id
        eos_set = (
            set(eos_ids)
            if isinstance(eos_ids, list)
            else ({int(eos_ids)} if eos_ids is not None else set())
        )
        eos_position = next(
            (index for index, token in enumerate(token_ids) if token in eos_set),
            None,
        )
        visible_ids = token_ids if eos_position is None else token_ids[:eos_position]
        text = str(tokenizer.decode(visible_ids, skip_special_tokens=True))
        stopped_text = truncate_at_stop(text, request.stop_sequences)
        stopped = eos_position is not None or stopped_text.finish_reason == "stop"
        reason = finish_reason(
            stopped=stopped,
            generated_tokens=len(token_ids),
            maximum_tokens=request.max_output_tokens,
        )
        return ModelResponse(
            request_id=request.request_id,
            sample_id=request.sample_id,
            model_id=self.spec.id,
            status="success",
            text=stopped_text.text,
            finish_reason=reason,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=len(visible_ids),
                forward_passes=len(token_ids),
                wall_time_seconds=elapsed,
                peak_memory_bytes=peak_memory,
            ),
            decoding_metadata={
                "adapter": self.ADAPTER_NAME,
                "adapter_version": self.ADAPTER_VERSION,
                "requested_model_revision": self.spec.revision,
                "model_revision": self._resolved_model_revision,
                "tokenizer_revision": self._resolved_tokenizer_revision,
                "dtype": self.spec.dtype,
                "device": str(device),
                "do_sample": do_sample,
                "temperature": float(temperature),
                "matched_stop_sequence": stopped_text.matched_sequence,
                "raw_text": text,
                "raw_generated_token_ids": cast(JsonValue, token_ids),
            },
        )

    def _score_options(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        return tuple(self._score_options_one(request) for request in requests)

    def _score_options_one(self, request: ModelRequest) -> ModelResponse:
        torch, model, tokenizer = self._runtime()
        device = self._device(model)
        prompt = self._format_prompt(request, tokenizer)
        separator = self.spec.adapter_kwargs.get("option_separator", " ")
        if not isinstance(separator, str):
            raise AdapterContractError(
                "adapter_kwargs.option_separator must be a string"
            )
        scores: dict[str, float] = {}
        total_input_tokens = 0
        started = self._cuda_start(torch, device)
        with torch.inference_mode():
            for option in request.options:
                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=True,
                )["input_ids"]
                full_ids = tokenizer(
                    f"{prompt}{separator}{option}",
                    add_special_tokens=True,
                )["input_ids"]
                if len(full_ids) <= len(prompt_ids):
                    raise AdapterContractError(
                        f"option {option!r} produced no conditional tokens"
                    )
                input_ids = torch.tensor(
                    [full_ids],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                log_probabilities = torch.log_softmax(logits[:, :-1, :], dim=-1)
                target_ids = input_ids[:, 1:]
                token_scores = torch.gather(
                    log_probabilities,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)
                option_scores = token_scores[:, len(prompt_ids) - 1 :]
                scores[option] = float(option_scores.mean().item())
                total_input_tokens += len(full_ids)
        elapsed, peak_memory = self._cuda_finish(torch, device, started)
        return ModelResponse(
            request_id=request.request_id,
            sample_id=request.sample_id,
            model_id=self.spec.id,
            status="success",
            option_scores=scores,
            finish_reason="completed",
            usage=ModelUsage(
                input_tokens=total_input_tokens,
                output_tokens=0,
                forward_passes=len(request.options),
                wall_time_seconds=elapsed,
                peak_memory_bytes=peak_memory,
            ),
            decoding_metadata={
                "adapter": self.ADAPTER_NAME,
                "adapter_version": self.ADAPTER_VERSION,
                "requested_model_revision": self.spec.revision,
                "model_revision": self._resolved_model_revision,
                "tokenizer_revision": self._resolved_tokenizer_revision,
                "scoring": "mean_conditional_log_likelihood",
                "option_separator": separator,
            },
        )
