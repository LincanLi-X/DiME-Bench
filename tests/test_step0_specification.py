from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specification" / "benchmark-v1.json"


class Step0SpecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_validator_succeeds(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_step0.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exactly_four_tracks_are_frozen(self) -> None:
        self.assertEqual(
            [track["id"] for track in self.spec["tracks"]],
            [
                "track1_general",
                "track2_infilling",
                "track3_editing",
                "track4_reasoning",
            ],
        )

    def test_all_paper_columns_resolve(self) -> None:
        tracks = {track["id"]: track for track in self.spec["tracks"]}
        special_scopes = {"__track__", "__dataset_row__"}
        for mapping in self.spec["paper_result_columns"]:
            track = tracks[mapping["track"]]
            self.assertIn(mapping["metric"], track["metric_ids"])
            task_ids = {task["id"] for task in track["tasks"]}
            self.assertIn(mapping["task"], task_ids | special_scopes)

    def test_required_documents_contain_core_sections(self) -> None:
        required_headings = {
            "benchmark-v1.md": [
                "## Track 1",
                "## Track 2",
                "## Track 3",
                "## Track 4",
            ],
            "model-protocol.md": [
                "## Shared comparison contract",
                "## dLLM decoding contract",
                "## AR decoding contract",
            ],
            "metric-definitions.md": [
                "## Standard metrics",
                "## Track 2 mechanism metrics",
                "## Track 3 mechanism metrics",
                "## Track 4 aggregates",
            ],
            "failure-policy.md": [
                "## Failure classes",
                "## Metric eligibility",
                "## Retry policy",
            ],
        }
        for filename, headings in required_headings.items():
            text = (ROOT / "docs" / "specification" / filename).read_text(
                encoding="utf-8"
            )
            for heading in headings:
                self.assertIn(heading, text, f"{filename} is missing {heading}")

    def test_controlled_comparison_defaults(self) -> None:
        defaults = self.spec["defaults"]
        self.assertTrue(defaults["deterministic"])
        self.assertEqual(defaults["dllm_denoising_steps"], 128)
        self.assertEqual(defaults["dllm_unmasking"], "confidence_based")
        self.assertEqual(defaults["ar_decoding"], "greedy")
        self.assertTrue(defaults["retain_failures"])


if __name__ == "__main__":
    unittest.main()
