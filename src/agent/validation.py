"""Strict validation of the paper-facing G1–G5 GeoFlow constraints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.agent.concepts import CONTEXTUAL_ROLES, GeoFlowGraph
from src.agent.semantics import TRANSFORMS

_ROLE_ORDER = {"sub_condition": 0, "condition": 1, "support": 2, "measure": 3}
_POLYMORPHIC_OUTPUTS = {
    "RESOLVE_PLACES": frozenset({"location", "object"}),
    "DISTANCE_MEASURE": frozenset({"amount", "field", "object"}),
    "FILTER": frozenset({"field", "object"}),
    "SORT": frozenset({"field", "object"}),
    "ORDINAL_SELECT": frozenset({"location", "object"}),
    "EXTREME_SELECT": frozenset({"amount", "location", "object"}),
    "ROUTE_COMPARE": frozenset({"field", "object"}),
    "ROUTE_OPTIMIZE": frozenset({"network", "object"}),
    "AGGREGATE": frozenset({"amount", "proportion"}),
    "MEASURE": frozenset({"amount", "event", "field", "network", "object", "proportion"}),
}
_TRANSFORM_INPUTS = {
    "RESOLVE_PLACES": frozenset({"location", "object"}),
    "PLACE_SEARCH": frozenset({"location", "object"}),
    "PLACE_DETAILS": frozenset({"location", "object"}),
    "DISTANCE_MEASURE": frozenset({"field", "location", "object"}),
    "ROUTE_MEASURE": frozenset({"location", "network", "object"}),
    "ROUTE_MATRIX": frozenset({"location", "network", "object"}),
    "SELECT_LEGS": frozenset({"field", "network"}),
    "ROUTE_EXTRACT": frozenset({"field"}),
    "ROUTE_STEPS": frozenset({"field"}),
    "ROUTE_COMPARE": frozenset({"field", "object"}),
    "ROUTE_OPTIMIZE": frozenset({"field", "network", "object"}),
    "SCHEDULE": frozenset({"amount", "event", "field", "network"}),
    "FILTER": frozenset({"field", "location", "object"}),
    "SORT": frozenset({"amount", "field", "object"}),
    "ORDINAL_SELECT": frozenset({"field", "object"}),
    "EXTREME_SELECT": frozenset({"amount", "field", "object"}),
    "AGGREGATE": frozenset({"amount", "field", "object", "proportion"}),
    "MATCH_OPTIONS": frozenset({"amount", "event", "field", "network", "object", "proportion"}),
    "MEASURE": frozenset(
        {"amount", "event", "field", "location", "network", "object", "proportion"}
    ),
}


@dataclass(frozen=True)
class ValidationResult:
    topological_order: tuple[str, ...]
    constraints: dict[str, bool]


class G1G5ValidationError(ValueError):
    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule


def validate_geoflow_graph(graph: GeoFlowGraph) -> ValidationResult:
    """Validate G1–G5 without template-specific or lenient fallback rules."""

    concepts = graph.concepts_by_id
    factors = graph.factors_by_id
    producer: dict[str, str] = {}
    for edge in graph.transformation_edges:
        for concept_id in edge.output_concepts:
            if concept_id not in concepts:
                raise G1G5ValidationError(
                    "G4", f"transformation {edge.id} outputs unknown concept {concept_id}"
                )
            if concept_id in producer:
                raise G1G5ValidationError("G4", f"concept {concept_id} has multiple producers")
            producer[concept_id] = edge.id
        unknown_inputs = [value for value in edge.input_concepts if value not in concepts]
        if unknown_inputs:
            raise G1G5ValidationError(
                "G4", f"transformation {edge.id} reads unknown concept {unknown_inputs[0]}"
            )
        unknown_factors = [value for value in edge.factor_nodes if value not in factors]
        if unknown_factors:
            raise G1G5ValidationError(
                "G4", f"transformation {edge.id} reads unknown factor {unknown_factors[0]}"
            )
    for factor in graph.factor_nodes:
        unknown_sources = [value for value in factor.source_concepts if value not in concepts]
        if unknown_sources:
            raise G1G5ValidationError(
                "G4", f"factor {factor.id} cites unknown concept {unknown_sources[0]}"
            )

    dependencies = {edge.id: set() for edge in graph.transformation_edges}
    consumers = {edge.id: set() for edge in graph.transformation_edges}
    edge_by_id = {edge.id: edge for edge in graph.transformation_edges}
    for edge in graph.transformation_edges:
        for concept_id in edge.input_concepts:
            parent = producer.get(concept_id)
            if parent and parent != edge.id:
                dependencies[edge.id].add(parent)
                consumers[parent].add(edge.id)

    order = _topological_order(dependencies, consumers)
    if len(order) != len(graph.transformation_edges):
        raise G1G5ValidationError("G1", "transformation graph contains a cycle")

    for edge in graph.transformation_edges:
        if edge.transformation.upper() == "MATCH_OPTIONS":
            raise G1G5ValidationError(
                "G3", "MATCH_OPTIONS belongs to MCQAdapter, outside the GeoFlow reasoning core"
            )
        transform = TRANSFORMS.get(edge.transformation.upper())
        if transform is None:
            raise G1G5ValidationError("G3", f"unknown spatial transformation {edge.transformation}")
        output_types = {concepts[value].core_concept for value in edge.output_concepts}
        accepted_outputs = _POLYMORPHIC_OUTPUTS.get(
            edge.transformation.upper(), frozenset({transform.output_type})
        )
        if not output_types <= accepted_outputs:
            raise G1G5ValidationError(
                "G3",
                f"{edge.id} ({edge.transformation}) produces {transform.output_type}, "
                f"not {', '.join(sorted(output_types))}",
            )
        accepted_inputs = _TRANSFORM_INPUTS.get(edge.transformation.upper())
        if accepted_inputs:
            incompatible = [
                value
                for value in edge.input_concepts
                if concepts[value].core_concept not in accepted_inputs
            ]
            if incompatible:
                concept = concepts[incompatible[0]]
                raise G1G5ValidationError(
                    "G3",
                    f"{edge.id} ({edge.transformation}) cannot consume "
                    f"{concept.core_concept} concept {concept.id}",
                )
        for source_id in edge.input_concepts:
            source = concepts[source_id]
            for target_id in edge.output_concepts:
                target = concepts[target_id]
                if _role_inversion(source.functional_role, target.functional_role):
                    raise G1G5ValidationError(
                        "G2",
                        f"role ordering {source.functional_role} -> {target.functional_role} "
                        f"on {edge.id}",
                    )

    source_edges = {
        edge.id
        for edge in graph.transformation_edges
        if not dependencies[edge.id]
        and (
            not edge.input_concepts
            or any(
                concepts[value].functional_role in CONTEXTUAL_ROLES for value in edge.input_concepts
            )
            or any(
                concepts[value].functional_role in CONTEXTUAL_ROLES
                for value in edge.output_concepts
            )
        )
    }
    measure_edges = {
        edge.id
        for edge in graph.transformation_edges
        if any(concepts[value].functional_role == "measure" for value in edge.output_concepts)
    }
    if not source_edges:
        raise G1G5ValidationError("G5", "graph has no contextual input")
    if not measure_edges:
        raise G1G5ValidationError("G5", "graph has no measure output")

    from_context = _closure(source_edges, consumers)
    to_measure = _reverse_closure(measure_edges, dependencies)
    all_edges = set(edge_by_id)
    unavailable = all_edges - from_context
    if unavailable:
        raise G1G5ValidationError(
            "G4", f"transformation {sorted(unavailable)[0]} has no available contextual data"
        )
    disconnected = all_edges - to_measure
    if disconnected:
        raise G1G5ValidationError(
            "G5", f"transformation {sorted(disconnected)[0]} does not contribute to a measure"
        )

    return ValidationResult(
        tuple(order),
        {
            "G1_acyclicity": True,
            "G2_role_ordering": True,
            "G3_type_compatibility": True,
            "G4_data_availability": True,
            "G5_connectivity": True,
        },
    )


def _role_inversion(source: str, target: str) -> bool:
    if source in CONTEXTUAL_ROLES or target in CONTEXTUAL_ROLES:
        return False
    return (
        source in _ROLE_ORDER
        and target in _ROLE_ORDER
        and _ROLE_ORDER[source] > _ROLE_ORDER[target]
    )


def _topological_order(
    dependencies: dict[str, set[str]], consumers: dict[str, set[str]]
) -> list[str]:
    incoming = {node: len(values) for node, values in dependencies.items()}
    ready = sorted(node for node, count in incoming.items() if count == 0)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for child in sorted(consumers[node]):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
                ready.sort()
    return ordered


def _closure(starts: Iterable[str], adjacency: dict[str, set[str]]) -> set[str]:
    found = set(starts)
    pending = list(starts)
    while pending:
        for value in adjacency[pending.pop()]:
            if value not in found:
                found.add(value)
                pending.append(value)
    return found


def _reverse_closure(starts: Iterable[str], dependencies: dict[str, set[str]]) -> set[str]:
    return _closure(starts, dependencies)
