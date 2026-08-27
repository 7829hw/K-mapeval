"""Type-compatible composition of macro-template graph fragments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from src.agent.concepts import ConceptNode, GeoFlowGraph, TransformationEdge
from src.agent.templates import MacroTemplate


def compatible_ports(output_concept: ConceptNode, template: MacroTemplate) -> list[str]:
    return [port.name for port in template.input_ports if port.accepts(output_concept)]


def compose_templates(
    templates: Sequence[MacroTemplate],
    *,
    bindings: Mapping[tuple[str, str], str] | None = None,
) -> GeoFlowGraph:
    """Compose fragments by explicit, type-compatible port bindings.

    No template is a hard constraint: this function supplies a construction prior that a planner
    may copy, adapt, or ignore. Missing bindings leave a placeholder concept as a graph input.
    """

    bindings = bindings or {}
    concepts: list[ConceptNode] = []
    edges: list[TransformationEdge] = []
    known: dict[str, ConceptNode] = {}
    for position, template in enumerate(templates):
        prefix = f"m{position + 1}_"
        rename: dict[str, str] = {}
        for node in template.concept_nodes:
            bound = bindings.get((template.name, node.id))
            if bound is not None:
                if bound not in known:
                    raise ValueError(f"Unknown composition binding: {bound}")
                if not any(port.accepts(known[bound]) for port in template.input_ports):
                    raise ValueError(
                        f"Type-incompatible port binding: {bound} -> {template.name}.{node.id}"
                    )
                rename[node.id] = bound
                continue
            renamed = prefix + node.id
            rename[node.id] = renamed
            clone = replace(node, id=renamed)
            concepts.append(clone)
            known[renamed] = clone
        for edge in template.transformation_edges:
            edges.append(
                replace(
                    edge,
                    id=prefix + edge.id,
                    input_concepts=tuple(rename[value] for value in edge.input_concepts),
                    output_concepts=tuple(rename[value] for value in edge.output_concepts),
                )
            )
    return GeoFlowGraph(tuple(concepts), tuple(edges), metadata={"prior_only": True})
