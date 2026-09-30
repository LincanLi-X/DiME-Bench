"""Single-completion pass@1 with an injectable code-execution backend."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from dimebench.datasets import GenerationRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.evaluators.standard._text import prediction_text
from dimebench.schemas.result import ResultSpec


@dataclass(frozen=True)
class CodeExecutionResult:
    """Minimal result returned by a versioned code executor."""

    passed: bool
    status: str
    stderr: str = ""


class CodeExecutor(Protocol):
    """Execution boundary for official or sandboxed code evaluation."""

    executor_id: str
    revision: str

    def execute(self, program: str, timeout_seconds: float) -> CodeExecutionResult:
        """Run a complete program and report whether every assertion passed."""


class LocalPythonExecutor:
    """Development-only isolated-process executor for trusted fixtures.

    This is not a security sandbox. Real benchmark submissions must inject a
    hardened executor (container, VM, or cluster sandbox).
    """

    executor_id = "local_python_subprocess"
    revision = "1.0.0"

    def execute(self, program: str, timeout_seconds: float) -> CodeExecutionResult:
        with tempfile.TemporaryDirectory(prefix="dimebench-code-") as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(program, encoding="utf-8")
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
            }
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-S", str(path)],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return CodeExecutionResult(False, "timeout")
        stderr = completed.stderr[-2000:]
        return CodeExecutionResult(
            completed.returncode == 0,
            "passed" if completed.returncode == 0 else "failed",
            stderr,
        )


class PassAt1Evaluator(StandardEvaluator):
    """Score one deterministic code completion against all frozen tests."""

    METRIC_ID = "pass_at_1"

    def __init__(
        self,
        executor: CodeExecutor | None = None,
        *,
        timeout_seconds: float = 3.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executor = executor or LocalPythonExecutor()
        self.timeout_seconds = timeout_seconds

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "k": 1,
            "executor_id": self.executor.executor_id,
            "executor_revision": self.executor.revision,
            "timeout_seconds": self.timeout_seconds,
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, GenerationRecord) or not sample.tests:
            raise EvaluationError("pass_at_1 requires a generation sample with tests")
        completion = prediction_text(result)
        if completion.lstrip().startswith(("def ", "async def ")):
            candidate = completion
        else:
            candidate = f"{sample.instruction.rstrip()}\n{completion}"
        program = candidate + "\n\n" + "\n".join(sample.tests) + "\n"
        execution = self.executor.execute(program, self.timeout_seconds)
        return float(execution.passed), {
            "execution_status": execution.status,
            "test_count": len(sample.tests),
            "stderr": execution.stderr,
        }


__all__ = [
    "CodeExecutionResult",
    "CodeExecutor",
    "LocalPythonExecutor",
    "PassAt1Evaluator",
]
