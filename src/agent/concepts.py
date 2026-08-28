"""Paper-facing GeoFlow intermediate representation.

GeoFlow's vertices are spatial concepts and its directed hyperedges are transformations.  Factors
are first-class vertices in the factorized graph, rather than literals hidden in operator calls.
The executable operator graph is deliberately a later representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.agent.semantics import accepted_input_types

CORE_CONCEPTS = frozenset(
    {"location", "object", "field", "event", "network", "amount", "proportion"}
)
FUNCTIONAL_ROLES = frozenset(
    {"extent", "temporal_extent", "sub_condition", "condition", "support", "measure"}
)
CONTEXTUAL_ROLES = frozenset({"extent", "temporal_extent"})


@dataclass(frozen=True)
class ConceptNode:
    id: str
    core_concept: str
    functional_role: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    implicit: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ConceptNode.id must not be empty")
        if self.core_concept not in CORE_CONCEPTS:
            raise ValueError(f"Unknown core concept: {self.core_concept}")
        if self.functional_role not in FUNCTIONAL_ROLES:
            raise ValueError(f"Unknown functional role: {self.functional_role}")
        object.__setattr__(self, "attributes", dict(self.attributes))

    @property
    def concept_type(self) -> str:
        """Compatibility name used by the deterministic operator contracts."""

        return self.core_concept

    @property
    def role(self) -> str:
        """Compatibility name used by the existing executor."""

        return self.functional_role

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "core_concept": self.core_concept,
            "functional_role": self.functional_role,
            "attributes": dict(self.attributes),
            "implicit": self.implicit,
        }

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "concept_type": self.core_concept,
            "role": self.functional_role,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, fallback_id: str) -> ConceptNode:
        concept = str(
            value.get("core_concept") or value.get("concept_type") or value.get("type") or "object"
        ).lower()
        role = str(value.get("functional_role") or value.get("role") or "support").lower()
        return cls(
            id=str(value.get("id") or fallback_id),
            text=str(value.get("text") or value.get("name") or ""),
            core_concept=concept if concept in CORE_CONCEPTS else "object",
            functional_role=role if role in FUNCTIONAL_ROLES else "support",
            attributes=value.get("attributes") if isinstance(value.get("attributes"), dict) else {},
            implicit=bool(value.get("implicit") or value.get("derived")),
        )


@dataclass(frozen=True)
class FactorNode:
    id: str
    factor_type: str
    value: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source_concepts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.factor_type:
            raise ValueError("FactorNode requires id and factor_type")
        object.__setattr__(self, "attributes", dict(self.attributes))
        object.__setattr__(self, "source_concepts", tuple(self.source_concepts))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "factor_type": self.factor_type,
            "value": self.value,
            "attributes": dict(self.attributes),
            "source_concepts": list(self.source_concepts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, fallback_id: str) -> FactorNode:
        return cls(
            id=str(value.get("id") or fallback_id),
            factor_type=str(value.get("factor_type") or value.get("type") or "parameter"),
            value=value.get("value"),
            attributes=value.get("attributes") if isinstance(value.get("attributes"), dict) else {},
            source_concepts=tuple(str(item) for item in value.get("source_concepts") or ()),
        )


@dataclass(frozen=True)
class TransformationEdge:
    id: str
    transformation: str
    input_concepts: tuple[str, ...]
    output_concepts: tuple[str, ...]
    factor_nodes: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.transformation:
            raise ValueError("TransformationEdge requires id and transformation")
        if not self.output_concepts:
            raise ValueError(f"TransformationEdge {self.id} has no output concept")
        object.__setattr__(self, "input_concepts", tuple(self.input_concepts))
        object.__setattr__(self, "output_concepts", tuple(self.output_concepts))
        object.__setattr__(self, "factor_nodes", tuple(self.factor_nodes))
        object.__setattr__(self, "attributes", dict(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transformation": self.transformation,
            "input_concepts": list(self.input_concepts),
            "output_concepts": list(self.output_concepts),
            "factor_nodes": list(self.factor_nodes),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, fallback_id: str) -> TransformationEdge:
        return cls(
            id=str(value.get("id") or fallback_id),
            transformation=str(value.get("transformation") or value.get("transform") or ""),
            input_concepts=tuple(str(item) for item in value.get("input_concepts") or ()),
            output_concepts=tuple(str(item) for item in value.get("output_concepts") or ()),
            factor_nodes=tuple(str(item) for item in value.get("factor_nodes") or ()),
            attributes=value.get("attributes") if isinstance(value.get("attributes"), dict) else {},
        )


@dataclass(frozen=True)
class GeoFlowGraph:
    concept_nodes: tuple[ConceptNode, ...]
    transformation_edges: tuple[TransformationEdge, ...]
    factor_nodes: tuple[FactorNode, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "concept_nodes", tuple(self.concept_nodes))
        object.__setattr__(self, "transformation_edges", tuple(self.transformation_edges))
        object.__setattr__(self, "factor_nodes", tuple(self.factor_nodes))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _unique_ids(self.concept_nodes, "concept")
        _unique_ids(self.factor_nodes, "factor")
        _unique_ids(self.transformation_edges, "transformation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_nodes": [node.as_dict() for node in self.concept_nodes],
            "factor_nodes": [node.as_dict() for node in self.factor_nodes],
            "transformation_edges": [edge.as_dict() for edge in self.transformation_edges],
            "metadata": dict(self.metadata),
        }

    @property
    def concepts_by_id(self) -> dict[str, ConceptNode]:
        return {node.id: node for node in self.concept_nodes}

    @property
    def factors_by_id(self) -> dict[str, FactorNode]:
        return {node.id: node for node in self.factor_nodes}

    def with_implicit_concepts(self) -> GeoFlowGraph:
        """Complete concepts implied by transformations, independently of benchmark labels."""

        concepts = list(self.concept_nodes)
        edges = list(self.transformation_edges)
        by_id = {node.id: node for node in concepts}

        def add(node: ConceptNode) -> None:
            if node.id not in by_id:
                concepts.append(node)
                by_id[node.id] = node

        transforms = {edge.transformation.upper() for edge in self.transformation_edges}
        if transforms & {"ROUTE_MEASURE", "ROUTE_MATRIX", "ROUTE_OPTIMIZE", "SELECT_LEGS"}:
            add(
                ConceptNode(
                    "implicit_network",
                    "network",
                    "support",
                    {"source": "implicit_completion"},
                    "road network",
                    True,
                )
            )
            first_route = next(
                (
                    index
                    for index, edge in enumerate(edges)
                    if edge.transformation.upper()
                    in {"ROUTE_MEASURE", "ROUTE_MATRIX", "ROUTE_OPTIMIZE", "SELECT_LEGS"}
                ),
                None,
            )
            if first_route is not None:
                edge = edges[first_route]
                edges[first_route] = TransformationEdge(
                    edge.id,
                    edge.transformation,
                    tuple(dict.fromkeys((*edge.input_concepts, "implicit_network"))),
                    tuple(dict.fromkeys((*edge.output_concepts, "implicit_route"))),
                    edge.factor_nodes,
                    edge.attributes,
                )
                for index in range(first_route + 1, len(edges)):
                    edge = edges[index]
                    # Only where the transformation actually consumes a route field. Handing
                    # `implicit_route` to one that does not is this file completing a graph the
                    # validator then refuses -- `ROUTE_MEASURE` and `ROUTE_MATRIX` take a place
                    # and produce a route, they do not read one, and that contradiction alone
                    # cost 36 questions in one hundred.
                    accepted = accepted_input_types(edge.transformation)
                    if (
                        edge.transformation.upper()
                        in {
                            "ROUTE_OPTIMIZE",
                            "SELECT_LEGS",
                            "ROUTE_EXTRACT",
                            "ROUTE_COMPARE",
                            "AGGREGATE",
                        }
                        and (accepted is None or "field" in accepted)
                    ):
                        edges[index] = TransformationEdge(
                            edge.id,
                            edge.transformation,
                            tuple(dict.fromkeys((*edge.input_concepts, "implicit_route"))),
                            edge.output_concepts,
                            edge.factor_nodes,
                            edge.attributes,
                        )
            add(
                ConceptNode(
                    "implicit_route",
                    "field",
                    "support",
                    {"source": "implicit_completion"},
                    "route field",
                    True,
                )
            )
        if transforms & {"DISTANCE_MEASURE", "SORT", "FILTER"}:
            add(
                ConceptNode(
                    "implicit_spatial_field",
                    "field",
                    "support",
                    {"source": "implicit_completion"},
                    "spatial field",
                    True,
                )
            )
            first_measure = next(
                (
                    index
                    for index, edge in enumerate(edges)
                    if edge.transformation.upper() == "DISTANCE_MEASURE"
                ),
                None,
            )
            if first_measure is not None:
                edge = edges[first_measure]
                edges[first_measure] = TransformationEdge(
                    edge.id,
                    edge.transformation,
                    edge.input_concepts,
                    tuple(dict.fromkeys((*edge.output_concepts, "implicit_spatial_field"))),
                    edge.factor_nodes,
                    edge.attributes,
                )
        return GeoFlowGraph(tuple(concepts), tuple(edges), self.factor_nodes, self.metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GeoFlowGraph:
        concepts = value.get("concept_nodes") or value.get("concepts") or ()
        factors = value.get("factor_nodes") or value.get("factors") or ()
        edges = value.get("transformation_edges") or value.get("edges") or ()
        if not all(isinstance(item, Mapping) for item in (*concepts, *factors, *edges)):
            raise ValueError("GeoFlow graph members must be objects")
        return cls(
            tuple(
                ConceptNode.from_dict(item, fallback_id=f"c{index + 1}")
                for index, item in enumerate(concepts)
            ),
            tuple(
                TransformationEdge.from_dict(item, fallback_id=f"t{index + 1}")
                for index, item in enumerate(edges)
            ),
            tuple(
                FactorNode.from_dict(item, fallback_id=f"f{index + 1}")
                for index, item in enumerate(factors)
            ),
            value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
        )


def factor_nodes_from_concepts(concepts: Sequence[ConceptNode]) -> tuple[FactorNode, ...]:
    """Promote typed constraint attributes to explicit factor vertices."""

    factor_keys = {
        "radius_m",
        "ordinal",
        "direction",
        "time_budget_s",
        "stay_duration_s",
        "stays",
        "metric",
        "measure",
        "return_to_start",
        "fixed_order",
    }
    factors: list[FactorNode] = []
    for concept in concepts:
        for key, value in concept.attributes.items():
            if key in factor_keys and value not in (None, "", [], ()):
                factors.append(
                    FactorNode(
                        id=f"factor_{concept.id}_{key}",
                        factor_type=key,
                        value=value,
                        attributes={"source": "concept_attribute"},
                        source_concepts=(concept.id,),
                    )
                )
    return tuple(factors)


def _unique_ids(values: Iterable[Any], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value.id in seen:
            raise ValueError(f"Duplicate {label} id: {value.id}")
        seen.add(value.id)
