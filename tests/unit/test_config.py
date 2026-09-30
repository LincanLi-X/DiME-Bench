from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dimebench.config import ConfigError, load_config, parse_override

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def test_load_smoke_config() -> None:
    spec = load_config(SMOKE_CONFIG)

    assert spec.run_id == "smoke_diffusion_infilling"
    assert spec.model.family == "diffusion"
    assert spec.task.primary_metric == "token_f1"
    assert len(spec.config_hash) == 64


def test_config_hash_is_stable_across_loads_and_yaml_key_order(
    tmp_path: Path,
) -> None:
    first = load_config(SMOKE_CONFIG)
    payload = {
        "run_id": "stable_hash",
        "schema_version": "1.0",
        "benchmark_id": "dime_bench_v1",
        "model": {
            "name_or_path": "model",
            "adapter": "mock",
            "family": "autoregressive",
            "id": "mock-ar",
            "dtype": "float32",
            "device": "cpu",
        },
        "dataset": {
            "name_or_path": "smoke",
            "source": "builtin",
            "id": "smoke",
            "split": "test",
        },
        "task": {
            "max_output_tokens": 8,
            "primary_metric": "accuracy",
            "metrics": ["accuracy"],
            "parser_id": "option",
            "prompt_id": "smoke",
            "type": "multiple_choice",
            "dataset_id": "smoke",
            "track": "track1_general",
            "id": "smoke",
        },
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 8,
        },
    }
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    second = load_config(reordered)
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )
    third = load_config(canonical)

    assert first.config_hash == load_config(SMOKE_CONFIG).config_hash
    assert second.config_hash == third.config_hash


def test_cli_overrides_are_typed_and_change_hash() -> None:
    baseline = load_config(SMOKE_CONFIG)
    overridden = load_config(
        SMOKE_CONFIG,
        ["batch_size=4", "dataset.sample_limit=2", "tags=[smoke, overridden]"],
    )

    assert overridden.batch_size == 4
    assert overridden.dataset.sample_limit == 2
    assert overridden.tags == ("smoke", "overridden")
    assert overridden.config_hash != baseline.config_hash
    assert parse_override("deterministic=false") == (("deterministic",), False)


@pytest.mark.parametrize(
    "override, message",
    [
        ("task.dataset_id=other", "must match dataset.id"),
        ("decoding.temperature=0.5", "deterministic runs require"),
        ("decoding.denoising_steps=null", "require decoding.denoising_steps"),
        ("unknown_field=true", "Extra inputs are not permitted"),
    ],
)
def test_invalid_config_is_rejected_before_execution(
    override: str,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(SMOKE_CONFIG, [override])


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "list.yaml"
    config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(config)
