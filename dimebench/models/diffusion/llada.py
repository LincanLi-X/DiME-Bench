"""LLaDA adapter using the official masked diffusion sampling procedure."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import JsonValue

from dimebench.decoding.diffusion.length_control import CanvasLayout
from dimebench.decoding.diffusion.sampler import LLaDASampler, LLaDASamplingConfig
from dimebench.decoding.stopping import finish_reason, truncate_at_stop
from dimebench.models.capabilities import (
    AdapterContractError,
    AdapterDependencyError,
    Capability,
)
from dimebench.models.diffusion.base import DiffusionModelAdapter
from dimebench.models.registry import register_adapter
from dimebench.schemas.model import ModelSpec
from dimebench.schemas.request import ModelRequest
from dimebench.schemas.response import ModelResponse, ModelUsage

_INFILL_MARKER = "<|dimebench_missing_span_f3a6b9|>"


def _mapping_option(
    options: Mapping[str, JsonValue],
    key: str,
) -> dict[str, Any]:
    value = options.get(key, {})
    if not isinstance(value, dict):
        raise AdapterContractError(f"adapter_kwargs.{key} must be a mapping")
    return dict(value)


def _numeric_option(
    value: JsonValue | None,
    *,
    name: str,
    default: float,
) -> float:
    resolved = default if value is None else value
    if not isinstance(resolved, (int, float)) or isinstance(resolved, bool):
        raise AdapterContractError(f"{name} must be numeric")
    return float(resolved)


@register_adapter("llada")
class LLaDAAdapter(DiffusionModelAdapter):
    """Adapter for GSAI-ML LLaDA checkpoints with native middle infilling."""

    ADAPTER_NAME = "llada"
    CAPABILITIES = frozenset(
        {
            Capability.GENERATE,
            Capability.INFILL,
            Capability.EDIT,
            Capability.REASONING,
            Capability.DENOISE,
        }
    )

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._sampler: LLaDASampler | None = None
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
                "LLaDA inference requires optional dependencies; "
                "install dime-bench[diffusion]"
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
        tokenizer.padding_side = "left"

        mask_token_id = self._mask_token_id(tokenizer)
        if tokenizer.pad_token_id == mask_token_id:
            raise AdapterContractError(
                "LLaDA pad_token_id must differ from mask_token_id"
            )
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
        model = transformers.AutoModel.from_pretrained(
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
        self._sampler = LLaDASampler(torch)

    def _close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._sampler = None
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._torch = None

    def _runtime(self) -> tuple[Any, Any, Any, LLaDASampler]:
        if (
            self._torch is None
            or self._model is None
            or self._tokenizer is None
            or self._sampler is None
        ):
            raise RuntimeError("LLaDA adapter runtime is not loaded")
        return self._torch, self._model, self._tokenizer, self._sampler

    def _mask_token_id(self, tokenizer: Any) -> int:
        configured = self.spec.adapter_kwargs.get("mask_token_id")
        if configured is not None:
            if not isinstance(configured, int) or isinstance(configured, bool):
                raise AdapterContractError(
                    "adapter_kwargs.mask_token_id must be an integer"
                )
            return configured
        tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
        if tokenizer_mask is not None:
            return int(tokenizer_mask)
        return 126336

    def _device(self, model: Any) -> Any:
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            return model.device

    def _use_chat_template(self) -> bool:
        value = self.spec.adapter_kwargs.get("use_chat_template", True)
        if not isinstance(value, bool):
            raise AdapterContractError(
                "adapter_kwargs.use_chat_template must be a boolean"
            )
        return value

    def _render_content(self, content: str, tokenizer: Any) -> str:
        if self._use_chat_template() and getattr(tokenizer, "chat_template", None):
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        return content

    def _encode(self, tokenizer: Any, text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        return [int(token) for token in encoded]

    def _build_canvas(
        self,
        request: ModelRequest,
        tokenizer: Any,
        torch: Any,
        device: Any,
        mask_token_id: int,
    ) -> tuple[Any, Any, Any, CanvasLayout]:
        if request.mode == "infill":
            if _INFILL_MARKER in request.prompt:
                raise AdapterContractError("prompt contains the reserved infill marker")
            prefix = request.prefix or ""
            suffix = request.suffix or ""
            content = (
                f"{request.prompt}\n\nPrefix and suffix context:\n"
                f"{prefix}{_INFILL_MARKER}{suffix}"
            )
            rendered = self._render_content(content, tokenizer)
            if rendered.count(_INFILL_MARKER) != 1:
                raise AdapterContractError(
                    "chat template did not preserve the infill marker exactly once"
                )
            left_text, right_text = rendered.split(_INFILL_MARKER)
            left_ids = self._encode(tokenizer, left_text)
            right_ids = self._encode(tokenizer, right_text)
        else:
            rendered = self._render_content(request.prompt, tokenizer)
            left_ids = self._encode(tokenizer, rendered)
            right_ids = []

        layout = CanvasLayout(
            left_length=len(left_ids),
            generated_length=request.max_output_tokens,
            right_length=len(right_ids),
        )
        token_ids = left_ids + [mask_token_id] * request.max_output_tokens + right_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        editable_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        editable_mask[:, layout.generated_slice] = True
        attention_mask = torch.ones_like(input_ids)
        maximum_positions = getattr(
            getattr(self._model, "config", None),
            "max_position_embeddings",
            None,
        )
        if maximum_positions is not None and layout.total_length > maximum_positions:
            raise AdapterContractError(
                f"request requires {layout.total_length} tokens but model limit is "
                f"{maximum_positions}"
            )
        return input_ids, editable_mask, attention_mask, layout

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

    def _default_steps(self, request: ModelRequest) -> int:
        value = request.metadata.get(
            "denoising_steps",
            self.spec.adapter_kwargs.get("default_steps", 128),
        )
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AdapterContractError("denoising_steps must be a positive integer")
        return value

    def _generate(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        responses = []
        for request in requests:
            region_policy = "bounded_middle" if request.mode == "infill" else "append"
            responses.append(
                self._run_request(
                    request,
                    steps=self._default_steps(request),
                    schedule=str(
                        request.metadata.get(
                            "schedule",
                            self.spec.adapter_kwargs.get("schedule", "linear"),
                        )
                    ),
                    region_policy=region_policy,
                )
            )
        return tuple(responses)

    def _denoise(
        self,
        requests: tuple[ModelRequest, ...],
        *,
        steps: int,
        schedule: str,
        mask_policy: str,
    ) -> Sequence[ModelResponse]:
        return tuple(
            self._run_request(
                request,
                steps=steps,
                schedule=schedule,
                region_policy=mask_policy,
            )
            for request in requests
        )

    def _run_request(
        self,
        request: ModelRequest,
        *,
        steps: int,
        schedule: str,
        region_policy: str,
    ) -> ModelResponse:
        torch, model, tokenizer, sampler = self._runtime()
        device = self._device(model)
        expected_policy = "bounded_middle" if request.mode == "infill" else "append"
        if region_policy != expected_policy:
            raise AdapterContractError(
                f"request mode {request.mode!r} requires mask_policy "
                f"{expected_policy!r}, got {region_policy!r}"
            )
        mask_token_id = self._mask_token_id(tokenizer)
        input_ids, editable_mask, attention_mask, layout = self._build_canvas(
            request,
            tokenizer,
            torch,
            device,
            mask_token_id,
        )
        block_value = request.metadata.get(
            "block_length",
            self.spec.adapter_kwargs.get("block_length"),
        )
        if block_value is not None and (
            not isinstance(block_value, int)
            or isinstance(block_value, bool)
            or block_value <= 0
        ):
            raise AdapterContractError("block_length must be a positive integer")
        temperature = _numeric_option(
            request.metadata.get("temperature"),
            name="temperature",
            default=_numeric_option(
                self.spec.adapter_kwargs.get("temperature"),
                name="adapter_kwargs.temperature",
                default=0.0,
            ),
        )
        guidance_scale = _numeric_option(
            request.metadata.get("guidance_scale"),
            name="guidance_scale",
            default=_numeric_option(
                self.spec.adapter_kwargs.get("guidance_scale"),
                name="adapter_kwargs.guidance_scale",
                default=0.0,
            ),
        )
        unmasking = request.metadata.get(
            "unmasking_strategy",
            self.spec.adapter_kwargs.get(
                "unmasking_strategy",
                "confidence_based",
            ),
        )
        if not isinstance(unmasking, str):
            raise AdapterContractError("unmasking_strategy must be a string")
        sampling_config = LLaDASamplingConfig(
            generation_length=request.max_output_tokens,
            steps=steps,
            block_length=block_value,
            temperature=temperature,
            guidance_scale=guidance_scale,
            schedule=schedule,
            unmasking_strategy=unmasking,
            mask_token_id=mask_token_id,
            seed=request.seed,
        )

        started = self._cuda_start(torch, device)
        sampled = sampler.sample(
            model,
            input_ids,
            editable_mask,
            sampling_config,
            attention_mask=attention_mask,
        )
        elapsed, peak_memory = self._cuda_finish(torch, device, started)
        generated_ids = sampled.token_ids[0, layout.generated_slice]
        token_ids = [int(token) for token in generated_ids.tolist()]
        eos_ids = getattr(tokenizer, "eos_token_id", None)
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
                input_tokens=layout.left_length + layout.right_length,
                output_tokens=len(visible_ids),
                forward_passes=sampled.forward_passes,
                denoising_steps=steps,
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
                "mask_token_id": mask_token_id,
                "mask_policy": region_policy,
                "unmasking_strategy": unmasking,
                "schedule": schedule,
                "block_length": sampling_config.resolved_budget().block_length,
                "temperature": temperature,
                "guidance_scale": guidance_scale,
                "matched_stop_sequence": stopped_text.matched_sequence,
                "raw_text": text,
                "raw_generated_token_ids": cast(JsonValue, token_ids),
            },
        )
