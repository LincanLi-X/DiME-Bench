from __future__ import annotations

import json
from pathlib import Path

import pytest

from dimebench.cli.main import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_cpu_quickstart_uses_mock_model_and_tiny_data(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    assert (
        main(
            [
                "run",
                "--config",
                str(PROJECT_ROOT / "configs/smoke.yaml"),
                "--set",
                "run_id=step16-ci-smoke",
                "--set",
                f"output_dir={output_root}",
                "--report-dir",
                str(report_root),
            ]
        )
        == 0
    )
    run_dir = output_root / "step16-ci-smoke"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert manifest["status"] == "completed"
    assert manifest["config"]["model"]["adapter"] == "mock"
    assert manifest["config"]["model"]["device"] == "cpu"
    assert len(predictions) == 4
    assert all(row["status"] == "success" for row in predictions)
    assert (report_root / "benchmark-summary.json").is_file()
