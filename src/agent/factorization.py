"""Adapters around deterministic semantic factorization (G -> G')."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from src.agent.canonicalization import canonicalize_ir_payload
from src.agent.concepts import (
    ConceptNode,
    FactorNode,
    GeoFlowGraph,
    TransformationEdge,
    factor_nodes_from_concepts,
)
from src.agent.semantics import (
    SEMANTIC_FACTORS,
    TRANSFORMS,
    SemanticFactorization,
    factorize_semantic_graph,
)
from src.agent.validation import ValidationResult, validate_geoflow_graph


@dataclass(frozen=True)
class FactorizedPlan:
    geoflow: GeoFlowGraph
    validation: ValidationResult
    semantic: SemanticFactorization
    operator_hyperedges: tuple[dict[str, Any], ...]


def plan_to_geoflow(analysis: Mapping[str, Any], payload: Mapping[str, Any]) -> GeoFlowGraph:
    """Normalize either the Concept/Edge IR or the previous step-shaped wire format."""

    if payload.get("transformation_edges") is not None:
        return canonicalize_ir_payload(analysis, payload)
    nested = payload.get("geoflow")
    if isinstance(nested, Mapping):
        return canonicalize_ir_payload(analysis, nested)

    raw_steps = payload.get("graph") if payload.get("graph") is not None else payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("GeoFlow response does not contain a non-empty graph")
    concepts = [
        ConceptNode.from_dict(value, fallback_id=f"c{index + 1}")
        for index, value in enumerate(analysis.get("concepts") or ())
        if isinstance(value, Mapping)
    ]
    by_id = {node.id: node for node in concepts}
    factors = list(factor_nodes_from_concepts(concepts))
    factor_by_key = {(factor.factor_type, repr(factor.value)): factor.id for factor in factors}
    output_by_step: dict[str, tuple[str, ...]] = {}
    edges: list[TransformationEdge] = []

    for position, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError(f"GeoFlow node {position} is not an object")
        edge_id = str(raw.get("id") or f"t{position + 1}")
        transform = str(raw.get("transformation") or raw.get("transform") or "").upper()
        if not transform:
            raise ValueError(f"GeoFlow node {edge_id} names no transformation")
        if transform == "MATCH_OPTIONS":
            # Migration of recorded/old planner replies. MCQ matching is now performed after
            # GroundedAnswer generation and is never represented in the paper-facing graph.
            declared = raw.get("inputs") or raw.get("depends_on") or ()
            for value in declared:
                upstream = str(value).lstrip("$").split(".", 1)[0]
                for concept_id in output_by_step.get(upstream, ()):
                    node = by_id[concept_id]
                    promoted = replace(node, functional_role="measure")
                    by_id[concept_id] = promoted
                    concepts[concepts.index(node)] = promoted
            continue
        declared = raw.get("input_concepts") or raw.get("inputs") or raw.get("depends_on") or ()
        inputs: list[str] = []
        for value in declared:
            key = str(value).lstrip("$").split(".", 1)[0]
            if key in output_by_step:
                inputs.extend(output_by_step[key])
            elif key in by_id:
                inputs.append(key)

        explicit_outputs = raw.get("output_concepts")
        if explicit_outputs:
            outputs = tuple(str(value) for value in explicit_outputs)
        else:
            concept_ids = [
                str(value) for value in raw.get("concept_ids") or () if str(value) in by_id
            ]
            outputs = tuple(concept_ids)
        if not outputs:
            output_id = f"result_{edge_id}"
            default_role = "measure" if position == len(raw_steps) - 1 else "support"
            role = str(raw.get("role") or default_role)
            output_type = TRANSFORMS[transform].output_type if transform in TRANSFORMS else "object"
            node = ConceptNode(
                output_id,
                output_type,
                role,
                {"source": "transformation_output", "transformation_id": edge_id},
                f"{transform} result",
                True,
            )
            concepts.append(node)
            by_id[node.id] = node
            outputs = (node.id,)
        else:
            for output in outputs:
                if output not in by_id:
                    output_type = (
                        TRANSFORMS[transform].output_type if transform in TRANSFORMS else "object"
                    )
                    default_role = "measure" if position == len(raw_steps) - 1 else "support"
                    node = ConceptNode(
                        output,
                        output_type,
                        str(raw.get("role") or default_role),
                        {"source": "transformation_output"},
                        implicit=True,
                    )
                    concepts.append(node)
                    by_id[output] = node

        factor_ids: list[str] = []
        raw_factors = raw.get("factors") if isinstance(raw.get("factors"), Mapping) else {}
        for key, value in raw_factors.items():
            lookup = (str(key), repr(value))
            factor_id = factor_by_key.get(lookup)
            if factor_id is None:
                factor_id = f"factor_{edge_id}_{key}"
                factors.append(FactorNode(factor_id, str(key), value, {"source": "planner"}))
                factor_by_key[lookup] = factor_id
            factor_ids.append(factor_id)
        factor_ids.extend(
            factor.id
            for factor in factors
            if factor.id not in factor_ids and _factor_applies(factor.factor_type, transform)
        )
        edges.append(
            TransformationEdge(
                edge_id,
                transform,
                tuple(dict.fromkeys(inputs)),
                outputs,
                tuple(dict.fromkeys(factor_ids)),
                {
                    "role": raw.get("role"),
                    "via": list(raw.get("via") or ()),
                    "legacy_inputs": list(raw.get("inputs") or raw.get("depends_on") or ()),
                    "concept_ids": list(raw.get("concept_ids") or ()),
                },
            )
        )
        output_by_step[edge_id] = outputs
    return GeoFlowGraph(tuple(concepts), tuple(edges), tuple(factors)).with_implicit_concepts()


def geoflow_to_semantic_steps(graph: GeoFlowGraph) -> list[dict[str, Any]]:
    producer = {
        concept_id: edge.id
        for edge in graph.transformation_edges
        for concept_id in edge.output_concepts
    }
    factors = graph.factors_by_id
    steps: list[dict[str, Any]] = []
    for edge in graph.transformation_edges:
        dependencies = list(
            dict.fromkeys(
                producer[value]
                for value in edge.input_concepts
                if value in producer and producer[value] != edge.id
            )
        )
        semantic_factors = {
            factors[value].factor_type: factors[value].value
            for value in edge.factor_nodes
            if factors[value].factor_type in SEMANTIC_FACTORS
        }
        output_roles = [
            graph.concepts_by_id[value].functional_role for value in edge.output_concepts
        ]
        concept_ids = [
            value for value in edge.output_concepts if not graph.concepts_by_id[value].implicit
        ]
        # The producing nodes first, then the concepts this edge reads that no node stands for.
        # `_resolve_endpoints` and `_resolve_via` are built to read concept ids out of `inputs`
        # -- that is how the step-shaped wire format said *which* pair a measure is between --
        # and lowering the IR dropped them, so three `directions` nodes reading one geocode
        # batch all measured from its first place: a detour question answered 18345 m for a
        # gold of 11.4 km with every stage reporting success. Node ids are filtered out of
        # `inputs` by `produced_by` membership downstream, so appending concepts changes no
        # dependency.
        named_concepts = [
            value for value in edge.input_concepts if value not in dependencies
        ]
        steps.append(
            {
                "id": edge.id,
                "transform": edge.transformation,
                "inputs": [*dependencies, *named_concepts],
                "concept_ids": concept_ids,
                "role": (
                    "measure"
                    if "measure" in output_roles
                    else str(edge.attributes.get("role") or "support")
                ),
                **({"factors": semantic_factors} if semantic_factors else {}),
                **({"via": list(edge.attributes["via"])} if edge.attributes.get("via") else {}),
            }
        )
    return steps


def factorize_plan(
    analysis: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    options: Sequence[str],
    facts: Any,
    available: frozenset[str],
    strict_types: bool = True,
) -> FactorizedPlan:
    geoflow = plan_to_geoflow(analysis, payload)
    validation = validate_geoflow_graph(geoflow)
    semantic = factorize_semantic_graph(
        geoflow_to_semantic_steps(geoflow),
        concepts=[node.as_legacy_dict() for node in geoflow.concept_nodes],
        options=list(options),
        facts=facts,
        available=available,
        strict_types=strict_types,
    )
    factor_inputs = {edge.id: list(edge.factor_nodes) for edge in geoflow.transformation_edges}
    edges_by_id = {edge.id: edge for edge in geoflow.transformation_edges}
    hyperedges = tuple(
        {
            "operator_id": step["id"],
            "operator": step["operator"],
            "input_concepts": list(edges_by_id[step["id"]].input_concepts),
            "factor_nodes": factor_inputs.get(step["id"], []),
            "output_concepts": list(edges_by_id[step["id"]].output_concepts),
        }
        for step in semantic.graph
        if any(edge.id == step["id"] for edge in geoflow.transformation_edges)
    )
    return FactorizedPlan(geoflow, validation, semantic, hyperedges)


def route_through_the_stated_stop(graph: GeoFlowGraph, facts: Any) -> GeoFlowGraph:
    """Send a lone route through the stop the question states, before G1-G5 reads the graph.

    Grounding already binds `facts.via_place` onto a route that has no waypoints, but grounding
    runs after validation, and these graphs do not survive that long: the planner resolves the
    stop in its own node and then measures a route that ignores it, so the resolve node
    contributes to nothing and G5 refuses the whole graph. Binding it here connects the branch
    and sends the drive where the question said, in one move.

    Only with exactly one ROUTE_MEASURE in the graph -- a detour question measures two and only
    one goes through the stop, which is the graph's own structure to say -- and only onto a route
    that declared no `via` of its own. The stop is matched to a concept by the text the Analysis
    stage copied, never by where an input sat.
    """

    stated = str(getattr(facts, "via_place", "") or "").strip()
    if not stated:
        return graph
    routes = [
        edge
        for edge in graph.transformation_edges
        if edge.transformation.upper() == "ROUTE_MEASURE"
    ]
    if len(routes) != 1 or routes[0].attributes.get("via"):
        return graph
    stop = next(
        (node.id for node in graph.concept_nodes if str(node.text or "").strip() == stated), None
    )
    if stop is None or stop in routes[0].attributes.get("via", ()):
        return graph
    edges = tuple(
        replace(
            edge,
            # Kept among the inputs as well as recorded as `via`: G5 reads connectivity off the
            # concept graph, where `via` lives in attributes, so a stop named only there still
            # leaves its resolve node contributing to nothing.
            input_concepts=tuple(dict.fromkeys((*edge.input_concepts, stop))),
            attributes={**edge.attributes, "via": [stop]},
        )
        if edge is routes[0]
        else edge
        for edge in graph.transformation_edges
    )
    return GeoFlowGraph(graph.concept_nodes, edges, graph.factor_nodes, graph.metadata)


def grounding_factor_nodes(facts: Any) -> tuple[FactorNode, ...]:
    """The question's typed facts as explicit factor vertices.

    Shared with macro-template retrieval, which had been ranking templates against the factors
    derivable from the Analysis stage's concept attributes alone -- an empty list on every
    itinerary question, so `ROUTE-OPTIMIZE`, whose whole affinity is the time budget and the
    stays, was never retrieved for the questions that state exactly those.
    """

    values = {
        "radius_m": getattr(facts, "radius_m", None),
        "direction": getattr(facts, "direction", None),
        "time_budget_s": getattr(facts, "time_budget_s", None),
        "stays": getattr(facts, "stays", None),
        "metric": getattr(facts, "route_objective", None),
        "return_to_start": getattr(facts, "returns_to_start", None),
        "fixed_order": getattr(facts, "stated_order", None),
    }
    return tuple(
        FactorNode(
            id=f"factor_fact_{name}",
            factor_type=name,
            value=value,
            attributes={"source": "typed_question_fact"},
        )
        for name, value in values.items()
        if value not in (None, "", (), [], False)
    )


def attach_grounding_factors(graph: GeoFlowGraph, facts: Any) -> GeoFlowGraph:
    """Promote deterministic typed facts to FactorNodes and connect them to hyperedges."""

    graph = route_through_the_stated_stop(graph, facts)
    existing = {factor.factor_type for factor in graph.factor_nodes}
    added = [
        factor for factor in grounding_factor_nodes(facts) if factor.factor_type not in existing
    ]
    if not added:
        return graph
    edges = tuple(
        replace(
            edge,
            factor_nodes=tuple(
                dict.fromkeys(
                    [
                        *edge.factor_nodes,
                        *(
                            factor.id
                            for factor in added
                            if _factor_applies(factor.factor_type, edge.transformation)
                        ),
                    ]
                )
            ),
        )
        for edge in graph.transformation_edges
    )
    return GeoFlowGraph(
        graph.concept_nodes,
        edges,
        (*graph.factor_nodes, *added),
        graph.metadata,
    )


def _factor_applies(factor: str, transform: str) -> bool:
    groups = {
        "FILTER": {"radius_m", "direction", "target_subtype"},
        "ORDINAL_SELECT": {"ordinal"},
        "ROUTE_MEASURE": {"metric", "measure"},
        "ROUTE_MATRIX": {"metric", "measure", "fixed_order", "return_to_start"},
        "ROUTE_OPTIMIZE": {
            "time_budget_s",
            "stay_duration_s",
            "stays",
            "metric",
            "measure",
            "fixed_order",
            "return_to_start",
        },
        "SCHEDULE": {"time_budget_s", "stay_duration_s", "stays"},
        "AGGREGATE": {"metric", "measure", "aggregate"},
    }
    return factor in groups.get(transform, set())
