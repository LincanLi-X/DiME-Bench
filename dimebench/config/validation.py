"""Cross-schema validation for a complete DiME-Bench run."""

from __future__ import annotations

from dimebench.schemas.run import RunSpec


class ConfigValidationError(ValueError):
    """Raised when individually valid schemas violate the run contract."""


def validate_run_spec(spec: RunSpec) -> RunSpec:
    """Validate references and controlled-decoding invariants."""
    errors: list[str] = []

    if spec.task.dataset_id != spec.dataset.id:
        errors.append(
            "task.dataset_id must match dataset.id "
            f"({spec.task.dataset_id!r} != {spec.dataset.id!r})"
        )
    if spec.decoding.max_new_tokens != spec.task.max_output_tokens:
        errors.append(
            "decoding.max_new_tokens must equal task.max_output_tokens "
            "for controlled comparison"
        )
    if spec.deterministic and (
        spec.decoding.do_sample or spec.decoding.temperature != 0.0
    ):
        errors.append("deterministic runs require do_sample=false and temperature=0")

    diffusion_fields = (
        spec.decoding.denoising_steps,
        spec.decoding.unmasking_strategy,
        spec.decoding.schedule,
        spec.decoding.mask_policy,
    )
    if spec.model.family == "diffusion":
        if spec.decoding.denoising_steps is None:
            errors.append("diffusion models require decoding.denoising_steps")
        if spec.decoding.unmasking_strategy is None:
            errors.append("diffusion models require decoding.unmasking_strategy")
        if spec.decoding.schedule is None:
            errors.append("diffusion models require decoding.schedule")
        if spec.decoding.mask_policy is None:
            errors.append("diffusion models require decoding.mask_policy")
    elif any(value is not None for value in diffusion_fields):
        errors.append(
            "autoregressive models cannot define diffusion-only decoding fields"
        )

    if errors:
        detail = "\n- ".join(errors)
        raise ConfigValidationError(f"invalid run configuration:\n- {detail}")
    return spec
