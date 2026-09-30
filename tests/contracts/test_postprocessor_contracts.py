from __future__ import annotations

from dimebench.config import load_yaml
from dimebench.postprocessors import available_parsers, create_parser
from dimebench.schemas.task import TaskSpec
from tests.postprocessor_fixtures import PROJECT_ROOT


def test_every_v1_task_references_a_versioned_parser() -> None:
    registered = available_parsers()
    task_paths = sorted((PROJECT_ROOT / "configs" / "tasks").glob("*/*.yaml"))
    assert len(task_paths) == 15
    for path in task_paths:
        task = TaskSpec.model_validate(load_yaml(path))
        assert registered[task.parser_id] == task.parser_version
        parser = create_parser(task.parser_id, task.parser_version)
        assert task.parser_id in parser.parser_ids()


def test_v1_parser_registry_is_frozen() -> None:
    assert available_parsers() == {
        "code_extractor": "1.0.0",
        "full_text_edit": "1.0.0",
        "middle_only": "1.0.0",
        "numeric_answer": "1.0.0",
        "option_letter": "1.0.0",
        "option_letter_or_text": "1.0.0",
        "path_answer": "1.0.0",
        "raw_text": "1.0.0",
    }
