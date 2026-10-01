# Add a model adapter

1. Subclass `ModelAdapter`, `AutoregressiveModelAdapter`, or
   `DiffusionModelAdapter`.
2. Declare a lowercase `ADAPTER_NAME`, implementation version, model family,
   and exact capabilities.
3. Implement resource loading, inference, and cleanup behind the protected
   methods. Return one ordered `ModelResponse` per request.
4. Register the class with `@register_adapter("name")`.
5. Add a revision-pinned YAML file under `configs/models/`.
6. Add lifecycle, capability, request/response, and real-model tests. Keep the
   real-model gate separately marked `gpu`.

Never download weights during module import. Record model/tokenizer revisions,
usage, wall time, peak memory, stopping reason, and diffusion controls.

See [`examples/add_custom_model.py`](../../examples/add_custom_model.py).
