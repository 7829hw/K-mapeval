"""Typed macro-template and question–validated-graph example retrieval."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from src.agent.concepts import FactorNode, GeoFlowGraph
from src.agent.templates import MACRO_TEMPLATES, MacroTemplate

ExampleEmbedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class QuestionGraphExample:
    example_id: str
    question: str
    graph: GeoFlowGraph
    split: str = "train"

    @property
    def question_hash(self) -> str:
        return hashlib.sha256(_question_key(self.question).encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "validated_geoflow_graph": self.graph.as_dict(),
            "split": self.split,
        }


class QuestionGraphExampleStore:
    def __init__(self, examples: Sequence[QuestionGraphExample] = ()) -> None:
        self._examples: list[QuestionGraphExample] = []
        for example in examples:
            self.add(example)

    def add(self, example: QuestionGraphExample) -> None:
        from src.agent.validation import validate_geoflow_graph

        validate_geoflow_graph(example.graph)
        if any(row.example_id == example.example_id for row in self._examples):
            raise ValueError(f"Duplicate example id: {example.example_id}")
        self._examples.append(example)

    def retrieve(
        self,
        question: str,
        *,
        embed: ExampleEmbedder,
        limit: int = 2,
        excluded_example_ids: frozenset[str] = frozenset(),
    ) -> list[QuestionGraphExample]:
        """Embedding cosine top-k with exact-question leakage prevention."""

        asked_hash = hashlib.sha256(_question_key(question).encode()).hexdigest()
        candidates = [
            row
            for row in self._examples
            if row.example_id not in excluded_example_ids and row.question_hash != asked_hash
        ]
        if not candidates or limit <= 0:
            return []
        vectors = embed([question, *(row.question for row in candidates)])
        if len(vectors) != len(candidates) + 1:
            raise ValueError("embedder must return one vector per input")
        asked = vectors[0]
        ranked = sorted(
            zip(candidates, vectors[1:], strict=True),
            key=lambda pair: (-cosine_similarity(asked, pair[1]), pair[0].example_id),
        )
        return [row for row, _ in ranked[:limit]]


def default_example_store() -> QuestionGraphExampleStore:
    """Authored training demonstrations; benchmark/test questions are never inserted here."""

    from src.agent.composition import compose_templates

    questions = {
        "filter_aggregate_measure": "기준 공간의 조건을 만족하는 객체 수를 계산한다.",
        "object_field_measure": "객체와 공간 필드 사이의 측정값을 구한다.",
        "route_optimize": "여러 위치를 잇는 네트워크 경로를 최적화한다.",
        "multi_route_compare": "여러 네트워크 경로의 측정값을 비교한다.",
        "multi_segment_aggregate": "연속 경로 구간들의 측정값을 합산한다.",
        "geocode_batch_compare": "질문이 이름을 댄 장소들을 위치로 바꾸고 직선 거리를 비교한다.",
        "location_bearing_classify": "기준 장소에서 대상 장소가 어느 방위 구역에 있는지 가른다.",
        "route_step_extract": "계산된 경로의 턴 단위 안내에서 한 수치를 읽는다.",
        "place_attribute_query": "찾은 장소에 저장된 속성을 조회한다.",
        "time_window_reverse": "이동 소요 시간으로 도착·출발 시각을 되짚는다.",
    }
    factors = {
        "filter_aggregate_measure": ("FILTER", FactorNode("example_radius", "radius_m", 500)),
        "object_field_measure": (
            "AGGREGATE",
            FactorNode("example_aggregate", "aggregate", "count"),
        ),
        "route_optimize": (
            "ROUTE_OPTIMIZE",
            FactorNode("example_budget", "time_budget_s", 10_800),
        ),
        "multi_route_compare": (
            "EXTREME_SELECT",
            FactorNode("example_extreme", "extreme", "min"),
        ),
        "multi_segment_aggregate": (
            "AGGREGATE",
            FactorNode("example_metric", "metric", "distance"),
        ),
        "geocode_batch_compare": (
            "EXTREME_SELECT",
            FactorNode("example_compare_extreme", "extreme", "min"),
        ),
        "location_bearing_classify": (
            "FILTER",
            FactorNode("example_direction", "direction", "east"),
        ),
        "route_step_extract": (
            "ROUTE_STEPS",
            FactorNode("example_step_key", "key", "left_turns"),
        ),
        "place_attribute_query": (
            "PLACE_DETAILS",
            FactorNode("example_attribute_key", "key", "category"),
        ),
        "time_window_reverse": (
            "SCHEDULE",
            FactorNode("example_schedule_measure", "measure", "arrival"),
        ),
    }
    rows = [
        QuestionGraphExample(
            example_id=f"appendix_e_{key}",
            question=questions[key],
            graph=_bind_example_factor(compose_templates([template]), *factors[key]),
        )
        for key, template in MACRO_TEMPLATES.items()
    ]
    return QuestionGraphExampleStore(rows)


def _bind_example_factor(
    graph: GeoFlowGraph, transformation: str, factor: FactorNode
) -> GeoFlowGraph:
    edges = tuple(
        replace(edge, factor_nodes=(*edge.factor_nodes, factor.id))
        if edge.transformation == transformation
        else edge
        for edge in graph.transformation_edges
    )
    return GeoFlowGraph(graph.concept_nodes, edges, (*graph.factor_nodes, factor), graph.metadata)


def retrieve_macro_templates(
    concepts: Sequence[dict[str, Any]],
    factors: Sequence[FactorNode],
    *,
    limit: int = 2,
) -> list[MacroTemplate]:
    """Rank reusable fragments from typed concepts/factors, never question literals or intent."""

    cores = {
        str(value.get("core_concept") or value.get("concept_type") or "object")
        for value in concepts
        if isinstance(value, dict)
    }
    roles = {
        str(value.get("functional_role") or value.get("role") or "support")
        for value in concepts
        if isinstance(value, dict)
    }
    factor_types = {factor.factor_type for factor in factors}
    scored: list[tuple[int, str, MacroTemplate]] = []
    for key, template in MACRO_TEMPLATES.items():
        accepted = set().union(*(port.core_concepts for port in template.input_ports))
        score = 2 * len(cores & accepted)
        score += 4 * len(factor_types & template.factor_affinity)
        if "network" in cores and "ROUTE" in template.name:
            score += 4
        if "proportion" in cores and "AGGREGATE" in template.name:
            score += 3
        if "measure" in roles:
            score += 1
        scored.append((score, key, template))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [template for _, _, template in scored[:limit]]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    scale = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / scale if scale else 0.0


def _question_key(question: str) -> str:
    return "".join(question.split()).casefold()
