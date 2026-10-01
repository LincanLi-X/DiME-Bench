from __future__ import annotations

import json
from pathlib import Path

from dimebench.cli.main import build_parser, main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_required_step15_commands_are_registered() -> None:
    parser = build_parser()
    for arguments in (
        ["list", "models"],
        ["list", "tasks"],
        ["inspect", "prompt", "--task", "x", "--sample-id", "y"],
        ["run", "--config", "config.yaml"],
        ["smoke"],
        ["reproduce", "paper"],
        ["leaderboard"],
    ):
        parser.parse_args(arguments)


def test_unified_cli_completes_cpu_smoke_run(tmp_path: Path, capsys: object) -> None:
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    assert (
        main(
            [
                "run",
                "--config",
                str(PROJECT_ROOT / "configs" / "smoke.yaml"),
                "--set",
                "run_id=cli-smoke",
                "--set",
                f"output_dir={output_root}",
                "--report-dir",
                str(report_root),
            ]
        )
        == 0
    )
    run_dir = output_root / "cli-smoke"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert manifest["status"] == "completed"
    assert summary["status"] == "completed"
    assert set(summary["metrics"]) == {"boundary_consistency", "token_f1"}
    assert (report_root / "benchmark-summary.json").is_file()
    assert main(["evaluate", "--run-dir", str(run_dir)]) == 0
    assert main(["leaderboard", "--run-dir", str(run_dir)]) == 0
