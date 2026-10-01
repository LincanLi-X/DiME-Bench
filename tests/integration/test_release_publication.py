from __future__ import annotations

from pathlib import Path

from dimebench.artifacts.hashing import hash_file
from dimebench.publishing import build_leaderboard_site, validate_release
from dimebench.reporting import load_leaderboard_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "results/releases/v1.0.0"
STATIC_ROOT = PROJECT_ROOT / "leaderboard/static/v1.0.0"


def test_committed_v1_release_and_static_leaderboard_are_identical() -> None:
    manifest = validate_release(RELEASE_ROOT)
    leaderboard = load_leaderboard_summary(RELEASE_ROOT / "result-submission.json")
    assert manifest.release_version == "v1.0.0"
    assert len(leaderboard.entries) == 2
    assert hash_file(RELEASE_ROOT / "benchmark-summary.json") == hash_file(
        STATIC_ROOT / "benchmark-summary.json"
    )
    release_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file()
    )
    assert "/Users/" not in release_text
    assert "/home/" not in release_text
    assert "/blue/" not in release_text


def test_static_leaderboard_can_be_rebuilt_from_versioned_submission(
    tmp_path: Path,
) -> None:
    report = build_leaderboard_site(
        RELEASE_ROOT / "result-submission.json",
        tmp_path / "site",
    )
    assert report.status == "passed"
    assert report.summary_identical is True
    assert report.entry_count == 2
    assert (tmp_path / "site/index.html").is_file()
    assert (tmp_path / "site/leaderboard.json").is_file()
