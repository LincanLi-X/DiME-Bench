from __future__ import annotations

from pathlib import Path

from dimebench.datasets.records import (
    EditingRecord,
    GenerationRecord,
    InfillingRecord,
    ReasoningRecord,
)
from dimebench.tasks import TaskCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG = TaskCatalog.from_roots(
    PROJECT_ROOT / "configs" / "tasks",
    PROJECT_ROOT / "dimebench" / "prompts",
)


def test_catalog_contains_all_v1_tasks() -> None:
    assert len(CATALOG) == 15
    assert CATALOG.ids[0] == "arxiv_abstracts"
    assert len(CATALOG.catalog_hash) == 64


def test_infilling_prompt_never_discloses_gold_middle() -> None:
    sample = InfillingRecord(
        sample_id="wikitext103.test.example",
        dataset_id="wikitext103",
        split="test",
        source_id="example",
        prefix="The fixed prefix has five words",
        middle="SECRET MIDDLE",
        suffix="the fixed suffix also has words",
        domain="test",
        span_length=2,
        prefix_length=6,
        suffix_length=6,
        span_length_bin="short",
        boundary_density="low",
    )
    task = CATALOG.get("wikitext103")
    for family in ("autoregressive", "diffusion"):
        prompt = task.render(sample, family)
        assert sample.middle not in prompt.text
        assert sample.prefix in prompt.text
        assert sample.suffix in prompt.text


def test_editing_prompt_never_discloses_reference_edit() -> None:
    sample = EditingRecord(
        sample_id="synthetic_repair.test.example",
        dataset_id="synthetic_repair",
        split="test",
        source_id="example",
        source_text="The value is wrong.",
        instruction="Repair the value.",
        reference_edit="UNIQUE GOLD REVISION",
        minimum_edit_distance=5,
        edit_type="repair",
        evidence=("The correct value is documented.",),
    )
    task = CATALOG.get("synthetic_repair")
    for family in ("autoregressive", "diffusion"):
        assert sample.reference_edit not in task.render(sample, family).text


def test_generation_and_reasoning_prompts_hide_references() -> None:
    generation = GenerationRecord(
        sample_id="gsm8k_general.test.example",
        dataset_id="gsm8k_general",
        split="test",
        source_id="example",
        instruction="What is one plus one?",
        references=("UNIQUE-GOLD-2",),
    )
    reasoning = ReasoningRecord(
        sample_id="gsm8k_chain.test.example",
        dataset_id="gsm8k_chain",
        split="test",
        source_id="example",
        problem="What is two plus two?",
        answer="UNIQUE-GOLD-4",
        reasoning_type="chain",
    )
    for family in ("autoregressive", "diffusion"):
        assert (
            generation.references[0]
            not in CATALOG.get("gsm8k_general").render(generation, family).text
        )
        assert (
            reasoning.answer
            not in CATALOG.get("gsm8k_chain").render(reasoning, family).text
        )


def test_paired_prompts_share_semantic_hash_but_not_final_hash() -> None:
    sample = ReasoningRecord(
        sample_id="math500_chain.test.example",
        dataset_id="math500_chain",
        split="test",
        source_id="example",
        problem="Solve x + 1 = 2.",
        answer="1",
        reasoning_type="chain",
    )
    task = CATALOG.get("math500_chain")
    ar = task.render(sample, "autoregressive")
    dllm = task.render(sample, "diffusion")
    assert ar.semantic_hash == dllm.semantic_hash
    assert ar.template_hash != dllm.template_hash
    assert ar.prompt_hash != dllm.prompt_hash
    assert ar.prompt_version == dllm.prompt_version == "1.0.0"
