"""Path-Star start, goal, adjacency, node, and obstacle validation."""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from dimebench.datasets import ReasoningRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.schemas.result import ResultSpec


def _string_list(value: JsonValue | None, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"Path-Star metadata requires string list {field!r}")
    return tuple(cast(list[str], value))


def _reference_nodes(answer: str) -> tuple[str, ...]:
    nodes = tuple(part.strip() for part in answer.replace("→", "-").split("-"))
    if len(nodes) < 2 or any(not node for node in nodes):
        raise EvaluationError("Path-Star reference answer is not a valid path")
    return nodes


class PathValidityEvaluator(StandardEvaluator):
    """Validate a parsed path using the graph constraints in sample metadata."""

    METRIC_ID = "path_validity"

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "checks": ["start", "goal", "known_node", "adjacency", "obstacle"],
            "default_graph_direction": "undirected",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, ReasoningRecord):
            raise EvaluationError("path_validity requires a reasoning sample")
        parsed = result.parsed_output
        if not isinstance(parsed, dict):
            raise EvaluationError("path_validity requires parsed path output")
        nodes_value = parsed.get("nodes")
        if not isinstance(nodes_value, list) or not all(
            isinstance(node, str) for node in nodes_value
        ):
            raise EvaluationError("path_validity requires parsed_output.nodes")
        predicted = tuple(nodes_value)
        if len(predicted) < 2:
            return 0.0, {"violations": ["path_too_short"]}

        graph_nodes = set(_string_list(sample.metadata.get("nodes"), "nodes"))
        edge_labels = _string_list(sample.metadata.get("edges"), "edges")
        directed = sample.metadata.get("directed") is True
        edges: set[tuple[str, str]] = set()
        for label in edge_labels:
            separator = "->" if "->" in label else "-"
            parts = tuple(part.strip() for part in label.split(separator))
            if len(parts) != 2 or any(not part for part in parts):
                raise EvaluationError(f"invalid Path-Star edge {label!r}")
            edges.add((parts[0], parts[1]))
            if not directed:
                edges.add((parts[1], parts[0]))

        reference = _reference_nodes(sample.answer)
        start_value = sample.metadata.get("start")
        goal_value = sample.metadata.get("goal")
        start = start_value if isinstance(start_value, str) else reference[0]
        goal = goal_value if isinstance(goal_value, str) else reference[-1]
        raw_obstacles = sample.metadata.get("obstacles", [])
        obstacles = set(_string_list(raw_obstacles, "obstacles"))

        violations: list[str] = []
        if predicted[0] != start:
            violations.append("wrong_start")
        if predicted[-1] != goal:
            violations.append("wrong_goal")
        if any(node not in graph_nodes for node in predicted):
            violations.append("unknown_node")
        if any(node in obstacles for node in predicted):
            violations.append("obstacle")
        path_edges = zip(predicted, predicted[1:], strict=False)
        if any(edge not in edges for edge in path_edges):
            violations.append("non_adjacent")
        return float(not violations), {
            "start": start,
            "goal": goal,
            "predicted_nodes": cast(JsonValue, list(predicted)),
            "violations": cast(JsonValue, violations),
            "directed": directed,
        }


__all__ = ["PathValidityEvaluator"]
