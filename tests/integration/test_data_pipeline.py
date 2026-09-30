from __future__ import annotations

from pathlib import Path

from dimebench.cli.main import main
from dimebench.datasets import prepare_suite, validate_prepared_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "data" / "registry.yaml"


def test_full_v1_sample_suite_is_reproducible(tmp_path: Path) -> None:
    first = prepare_suite(REGISTRY, "dime_bench_v1", tmp_path / "first")
    second = prepare_suite(REGISTRY, "dime_bench_v1", tmp_path / "second")
    first_report = validate_prepared_data(tmp_path / "first" / "dime_bench_v1")
    second_report = validate_prepared_data(tmp_path / "second" / "dime_bench_v1")
    assert first["suite_hash"] == second["suite_hash"]
    assert first_report.status == "passed"
    assert second_report.status == "passed"
    assert first_report.dataset_count == 15
    assert first_report.sample_count == 15
    assert first_report.dataset_hashes == second_report.dataset_hashes
    assert first_report.lock_hashes == second_report.lock_hashes


def test_prepare_and_validate_cli(tmp_path: Path, capsys) -> None:
    output = tmp_path / "prepared"
    assert (
        main(
            [
                "prepare",
                "--suite",
                "dime_bench_v1_smoke",
                "--registry",
                str(REGISTRY),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert "suite_hash:" in capsys.readouterr().out
    assert main(["data", "validate", str(output / "dime_bench_v1_smoke")]) == 0
    assert '"status": "passed"' in capsys.readouterr().out
