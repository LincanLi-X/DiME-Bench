from __future__ import annotations

import subprocess
import sys
import unittest

import dimebench
from dimebench.cli.main import build_parser, main


class PackageImportTests(unittest.TestCase):
    def test_package_exports_version(self) -> None:
        self.assertEqual(dimebench.__version__, "1.0.0")
        self.assertEqual(dimebench.__all__, ["__version__"])

    def test_cli_parser_identity(self) -> None:
        self.assertEqual(build_parser().prog, "dimebench")

    def test_cli_main_without_arguments(self) -> None:
        self.assertEqual(main([]), 0)

    def test_python_module_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dimebench", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: dimebench", result.stdout)

    def test_python_module_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dimebench", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "dimebench 1.0.0")


if __name__ == "__main__":
    unittest.main()
