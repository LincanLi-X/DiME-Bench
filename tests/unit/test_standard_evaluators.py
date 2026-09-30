from __future__ import annotations

import json
from pathlib import Path

import pytest

from dimebench.evaluators import EvaluationError, evaluate_sample
from dimebench.evaluators.standard import (
    AccuracyEvaluator,
    BERTScoreEvaluator,
    ExactMatchEvaluator,
    IFEvalEvaluator,
    OptionAccuracyEvaluator,
    PassAt1Evaluator,
    PathValidityEvaluator,
    RougeLEvaluator,
    TokenF1Evaluator,
    available_standard_evaluators,
    create_standard_evaluator,
)
from dimebench.evaluators.standard.rouge_l import rouge_l_f1
from dimebench.evaluators.standard.token_f1 import token_f1
from tests.standard_metric_fixtures import (
    generation_sample,
    infilling_sample,
    multiple_choice_sample,
    option_sample,
    path_sample,
    result_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FixedBERTScoreBackend:
    backend_id = "fixture"
    revision = "1"

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def score(self, prediction, reference, **kwargs):
        del prediction, reference, kwargs
        self.calls += 1
        return self.value


def test_standard_registry_contains_all_nine_metrics() -> None:
    assert set(available_standard_evaluators()) == {
        "accuracy",
        "normalized_exact_match",
        "pass_at_1",
        "instruction_following_rate",
        "token_f1",
        "rouge_l",
        "bertscore_f1",
        "option_accuracy",
        "path_validity",
    }
    assert create_standard_evaluator("accuracy").metric_id == "accuracy"
    with pytest.raises(EvaluationError, match="unknown standard metric"):
        create_standard_evaluator("not_a_metric")


def test_accuracy_and_option_accuracy() -> None:
    multiple_choice = multiple_choice_sample()
    assert (
        AccuracyEvaluator()
        .evaluate(result_for(multiple_choice, {"index": 1}), multiple_choice)
        .value
        == 1.0
    )
    assert (
        AccuracyEvaluator()
        .evaluate(result_for(multiple_choice, {"index": 0}), multiple_choice)
        .value
        == 0.0
    )

    option = option_sample()
    assert (
        OptionAccuracyEvaluator()
        .evaluate(result_for(option, {"index": 0}), option)
        .value
        == 1.0
    )


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("12.0", "12", 1.0),
        (r"\frac{1}{2}", "1/2", 1.0),
        ("Answer", " answer ", 1.0),
        ("13", "12", 0.0),
    ],
)
def test_normalized_exact_match(
    prediction: str, reference: str, expected: float
) -> None:
    sample = generation_sample(reference)
    outcome = ExactMatchEvaluator().evaluate(result_for(sample, prediction), sample)
    assert outcome.value == expected


def test_token_f1_matches_known_multiset_result() -> None:
    assert token_f1("newly comet", "newly discovered comet") == pytest.approx(0.8)
    assert token_f1("a a", "a") == pytest.approx(2 / 3)
    sample = infilling_sample("newly discovered comet")
    assert TokenF1Evaluator().evaluate(
        result_for(sample, "newly comet"), sample
    ).value == pytest.approx(0.8)


def test_rouge_l_matches_known_lcs_results() -> None:
    assert rouge_l_f1("a b c", "a x b c") == pytest.approx(6 / 7)
    assert rouge_l_f1("a c b", "a b c") == pytest.approx(2 / 3)
    sample = infilling_sample("a x b c")
    assert RougeLEvaluator().evaluate(
        result_for(sample, "a b c"), sample
    ).value == pytest.approx(6 / 7)


def test_ifeval_strict_prompt_aggregation() -> None:
    sample = generation_sample(
        "one two three",
        instruction="Reply with exactly three lowercase words.",
        metadata={"instruction_ids": ["exact_word_count", "lowercase"]},
    )
    evaluator = IFEvalEvaluator()
    assert evaluator.evaluate(result_for(sample, "one two three"), sample).value == 1
    assert evaluator.evaluate(result_for(sample, "One two three"), sample).value == 0


def test_pass_at_1_runs_all_trusted_fixture_tests() -> None:
    sample = generation_sample(
        "\n    return x + 1",
        instruction='def add_one(x):\n    """Return x plus one."""',
        tests=("assert add_one(2) == 3", "assert add_one(-1) == 0"),
    )
    evaluator = PassAt1Evaluator(timeout_seconds=2)
    assert (
        evaluator.evaluate(
            result_for(sample, {"code": "    return x + 1"}), sample
        ).value
        == 1
    )
    assert (
        evaluator.evaluate(
            result_for(sample, {"code": "    return x - 1"}), sample
        ).value
        == 0
    )


def test_bertscore_uses_injected_backend_and_records_freeze() -> None:
    backend = FixedBERTScoreBackend(0.75)
    sample = infilling_sample("reference")
    evaluator = BERTScoreEvaluator(
        backend,
        model_type="fixture-model",
        model_revision="abc123",
        tokenizer_revision="def456",
    )
    outcome = evaluator.evaluate(result_for(sample, "candidate"), sample)
    assert outcome.value == pytest.approx(0.75)
    assert backend.calls == 1
    assert evaluator.configuration()["model_revision"] == "abc123"


def test_bertscore_accepts_negative_rescaled_values() -> None:
    sample = infilling_sample("reference")
    outcome = BERTScoreEvaluator(FixedBERTScoreBackend(-0.05)).evaluate(
        result_for(sample, "candidate"), sample
    )
    assert outcome.value == pytest.approx(-0.05)


def test_path_validity_checks_endpoints_edges_and_obstacles() -> None:
    sample = path_sample()
    evaluator = PathValidityEvaluator()
    assert (
        evaluator.evaluate(
            result_for(sample, {"nodes": ["C", "B", "A", "D"]}), sample
        ).value
        == 1
    )
    invalid = evaluator.evaluate(result_for(sample, {"nodes": ["C", "A", "D"]}), sample)
    assert invalid.value == 0
    assert "non_adjacent" in invalid.details["violations"]

    obstructed = path_sample(obstacles=["A"])
    outcome = evaluator.evaluate(
        result_for(obstructed, {"nodes": ["C", "B", "A", "D"]}), obstructed
    )
    assert outcome.value == 0
    assert "obstacle" in outcome.details["violations"]


def test_failed_result_scores_zero_without_calling_expensive_backend() -> None:
    backend = FixedBERTScoreBackend(0.9)
    sample = infilling_sample("reference")
    outcome = BERTScoreEvaluator(backend).evaluate(
        result_for(sample, None, status="failure"), sample
    )
    assert outcome.value == 0
    assert outcome.details["route"] == "failure_zero"
    assert backend.calls == 0


def test_sample_metric_record_contains_values_and_evaluator_provenance() -> None:
    sample = infilling_sample("newly discovered comet")
    result = result_for(sample, "newly comet")
    record = evaluate_sample(
        result,
        sample,
        [TokenF1Evaluator(), RougeLEvaluator()],
    )
    assert record.prediction_hash == "b" * 64
    assert record.metrics["token_f1"] == pytest.approx(0.8)
    assert record.evaluator_metadata["token_f1.version"] == "1.0.0"
    assert len(record.evaluator_metadata["token_f1.config_hash"]) == 64
    assert json.loads(record.evaluator_metadata["token_f1.details"])


def test_committed_known_value_fixtures() -> None:
    cases = json.loads(
        (PROJECT_ROOT / "tests/fixtures/standard_metrics/golden.json").read_text()
    )
    for case in cases:
        if case["metric"] == "normalized_exact_match":
            sample = generation_sample(case["reference"])
            actual = (
                ExactMatchEvaluator()
                .evaluate(result_for(sample, case["prediction"]), sample)
                .value
            )
        elif case["metric"] == "token_f1":
            actual = token_f1(case["prediction"], case["reference"])
        else:
            actual = rouge_l_f1(case["prediction"], case["reference"])
        assert actual == pytest.approx(case["expected"]), case["name"]
