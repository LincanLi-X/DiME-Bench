from __future__ import annotations

import json
from pathlib import Path

from dimebench.reporting import PaperReproductionConfig, reproduce_paper
from tests.step14_fixtures import make_sealed_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_paper_outputs_are_deterministic_and_traceable(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_sealed_run(
        runs,
        run_id="fixture-ar",
        family="autoregressive",
        value=0.25,
        model_id="fixture-ar",
    )
    make_sealed_run(
        runs,
        run_id="fixture-dllm",
        family="diffusion",
        value=0.75,
        model_id="fixture-dllm",
    )
    first_dir = tmp_path / "first"
    config = PaperReproductionConfig(
        experiment_id="fixture",
        run_root=runs,
        output_dir=first_dir,
        specification=PROJECT_ROOT / "specification" / "benchmark-v1.json",
        require_complete_tables=False,
    )
    first = reproduce_paper(config, allow_incomplete=True)
    second = reproduce_paper(
        config,
        output_dir=tmp_path / "second",
        allow_incomplete=True,
    )

    assert first.status == "passed"
    assert first.output_hashes == second.output_hashes
    assert first.run_count == 2
    assert first.table_count == 4
    for name in (
        "benchmark-summary.json",
        "paper-tables.json",
        "paper-tables.csv",
        "paper-tables.md",
        "paper-tables.tex",
        "figure-input.json",
        "figure-input.csv",
        "traceability.json",
    ):
        assert (first_dir / name).is_file()
    tables = json.loads((first_dir / "paper-tables.json").read_text())
    ar_row = next(
        row for row in tables["Table 2"]["rows"] if row["model_id"] == "fixture-ar"
    )
    assert ar_row["cells"]["MMLU-Pro"]["value"] == 0.25
    assert ar_row["cells"]["MMLU-Pro"]["scaled_value"] == 25.0
    assert ar_row["cells"]["MMLU-Pro"]["source_run_ids"] == ["fixture-ar"]
