"""Implement and execute a deterministic custom sample-level metric."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import GenerationRecord, SampleRecord
from dimebench.evaluators import StandardEvaluator
from dimebench.schemas import ResultSpec


class ConcisionEvaluator(StandardEvaluator):
    METRIC_ID = "concision"

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        del sample
        text = str(result.parsed_output or "")
        token_count = len(text.split())
        return float(token_count <= 3), {"token_count": token_count, "limit": 3}


def main() -> None:
    sample = GenerationRecord(
        sample_id="example.test.item-1",
        dataset_id="example",
        split="test",
        source_id="item-1",
        instruction="Name the capital of France.",
        references=("Paris",),
    )
    result = ResultSpec(
        run_id="example",
        config_hash="a" * 64,
        sample_id=sample.sample_id,
        model_id="example-model",
        dataset_id=sample.dataset_id,
        task_id="example-task",
        status="success",
        raw_output="Paris",
        parsed_output="Paris",
    )
    outcome = ConcisionEvaluator().evaluate(result, sample)
    print(outcome.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
