"""Reusable Appendix-E-level macro templates.

Templates are typed graph-fragment priors. They do not classify a benchmark family and do not
impose required transformations on a graph produced by the planner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.agent.concepts import ConceptNode, TransformationEdge


@dataclass(frozen=True)
class TemplatePort:
    name: str
    core_concepts: frozenset[str]
    functional_roles: frozenset[str] = frozenset()
    multiple: bool = False

    def accepts(self, concept: ConceptNode) -> bool:
        return concept.core_concept in self.core_concepts and (
            not self.functional_roles or concept.functional_role in self.functional_roles
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "core_concepts": sorted(self.core_concepts),
            "functional_roles": sorted(self.functional_roles),
            "multiple": self.multiple,
        }


@dataclass(frozen=True)
class MacroTemplate:
    name: str
    input_ports: tuple[TemplatePort, ...]
    output_ports: tuple[TemplatePort, ...]
    concept_nodes: tuple[ConceptNode, ...]
    transformation_edges: tuple[TransformationEdge, ...]
    factor_affinity: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_ports": [port.as_dict() for port in self.input_ports],
            "output_ports": [port.as_dict() for port in self.output_ports],
            "concept_nodes": [node.as_dict() for node in self.concept_nodes],
            "transformation_edges": [edge.as_dict() for edge in self.transformation_edges],
            "factor_affinity": sorted(self.factor_affinity),
            "prior_only": True,
        }


_SPATIAL_INPUT = TemplatePort(
    "spatial_input",
    frozenset({"location", "object", "field", "network"}),
    frozenset({"extent", "temporal_extent", "condition", "support"}),
    True,
)
_MEASURE_OUTPUT = TemplatePort(
    "measure", frozenset({"amount", "object", "field", "proportion"}), frozenset({"measure"})
)


def _concept(node_id: str, core: str, role: str) -> ConceptNode:
    return ConceptNode(node_id, core, role, {"template_placeholder": True})


def _edge(edge_id: str, transform: str, inputs: tuple[str, ...], output: str) -> TransformationEdge:
    return TransformationEdge(edge_id, transform, inputs, (output,))


MACRO_TEMPLATES: dict[str, MacroTemplate] = {
    "filter_aggregate_measure": MacroTemplate(
        "FILTER-AGGREGATE-MEASURE",
        (_SPATIAL_INPUT,),
        (_MEASURE_OUTPUT,),
        (
            _concept("input", "object", "extent"),
            _concept("filtered", "object", "support"),
            _concept("measure", "amount", "measure"),
        ),
        (
            _edge("filter", "FILTER", ("input",), "filtered"),
            _edge("aggregate", "AGGREGATE", ("filtered",), "measure"),
        ),
        frozenset({"radius_m", "direction", "aggregate", "ordinal"}),
    ),
    "object_field_measure": MacroTemplate(
        "OBJECT-FIELD-MEASURE",
        (_SPATIAL_INPUT,),
        (_MEASURE_OUTPUT,),
        (
            _concept("object", "object", "extent"),
            _concept("field", "field", "support"),
            _concept("measure", "amount", "measure"),
        ),
        (
            _edge("field", "DISTANCE_MEASURE", ("object",), "field"),
            _edge("measure", "AGGREGATE", ("field",), "measure"),
        ),
        frozenset({"metric", "direction", "ordinal"}),
    ),
    "route_optimize": MacroTemplate(
        "ROUTE-OPTIMIZE",
        (_SPATIAL_INPUT,),
        (_MEASURE_OUTPUT,),
        (
            _concept("stops", "object", "extent"),
            _concept("network", "network", "support"),
            _concept("route", "field", "support"),
            _concept("measure", "object", "measure"),
        ),
        (
            _edge("routes", "ROUTE_MATRIX", ("stops", "network"), "route"),
            _edge("optimize", "ROUTE_OPTIMIZE", ("route",), "measure"),
        ),
        frozenset({"time_budget_s", "stay_duration_s", "fixed_order", "return_to_start", "metric"}),
    ),
    "multi_route_compare": MacroTemplate(
        "MULTI-ROUTE-COMPARE",
        (_SPATIAL_INPUT,),
        (_MEASURE_OUTPUT,),
        (
            _concept("places", "object", "extent"),
            _concept("routes", "field", "support"),
            _concept("measure", "object", "measure"),
        ),
        (
            _edge("routes", "ROUTE_MATRIX", ("places",), "routes"),
            _edge("compare", "EXTREME_SELECT", ("routes",), "measure"),
        ),
        frozenset({"metric", "direction", "extreme"}),
    ),
    "multi_segment_aggregate": MacroTemplate(
        "MULTI-SEGMENT-AGGREGATE",
        (_SPATIAL_INPUT,),
        (_MEASURE_OUTPUT,),
        (
            _concept("stops", "object", "extent"),
            _concept("routes", "field", "support"),
            _concept("legs", "field", "support"),
            _concept("measure", "amount", "measure"),
        ),
        (
            _edge("routes", "ROUTE_MATRIX", ("stops",), "routes"),
            _edge("legs", "SELECT_LEGS", ("routes",), "legs"),
            _edge("aggregate", "AGGREGATE", ("legs",), "measure"),
        ),
        frozenset({"metric", "fixed_order", "return_to_start"}),
    ),
}


def template_catalogue() -> list[dict[str, Any]]:
    return [template.as_dict() for template in MACRO_TEMPLATES.values()]
