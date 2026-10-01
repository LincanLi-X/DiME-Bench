"""Minimal custom autoregressive adapter with no external dependencies."""

from __future__ import annotations

from collections.abc import Sequence

from dimebench.models import Capability, create_adapter
from dimebench.models.autoregressive import AutoregressiveModelAdapter
from dimebench.models.registry import register_adapter
from dimebench.schemas import ModelRequest, ModelResponse, ModelSpec


@register_adapter("example-echo")
class EchoAdapter(AutoregressiveModelAdapter):
    ADAPTER_NAME = "example-echo"
    CAPABILITIES = frozenset({Capability.GENERATE})

    def _load(self) -> None:
        return None

    def _close(self) -> None:
        return None

    def _generate(
        self,
        requests: tuple[ModelRequest, ...],
    ) -> Sequence[ModelResponse]:
        return tuple(
            ModelResponse(
                request_id=request.request_id,
                sample_id=request.sample_id,
                model_id=self.spec.id,
                status="success",
                text=f"echo: {request.prompt}",
                finish_reason="completed",
            )
            for request in requests
        )


def main() -> None:
    spec = ModelSpec(
        id="example-echo",
        family="autoregressive",
        adapter="example-echo",
        name_or_path="builtin/example-echo",
        revision="1",
        dtype="float32",
        device="cpu",
        capabilities=("generate",),
    )
    request = ModelRequest(
        request_id="request-1",
        sample_id="sample-1",
        task_id="example-task",
        mode="generate",
        prompt="hello",
        max_output_tokens=8,
    )
    with create_adapter(spec) as adapter:
        print(adapter.generate((request,))[0].text)


if __name__ == "__main__":
    main()
