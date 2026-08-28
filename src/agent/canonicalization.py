"""Deterministic canonicalization of the planner's drafted GeoFlow, before strict G1-G5.

The Graph Construction stage answers in the Concept/Edge IR: `concept_nodes`, `factor_nodes` and
`transformation_edges`.  Asking a model to restate the Analysis stage's typed concepts verbatim,
in a second reply, is asking it to copy identifiers -- and when it paraphrases one, every
reference to it is a concept the graph does not contain.  Measured on a hundred questions, that
is what 82 of them died of: 18 drafts passed G1-G5 and the rest named concepts, factors or edges
that were not in their own reply.

The step-shaped wire format never had this problem, because `plan_to_geoflow` completed it:
concepts came from the Analysis stage, factors were derived from their attributes, an unresolved
reference was dropped rather than fatal, and a missing output was synthesized with the type its
transformation declares.  The IR path skipped all of it.  This module is that same completion,
written once, for the representation the planner actually answers in.

Two rules keep it from becoming a way to launder a bad graph into a good one:

* every repair is *additive and last-resort* -- it fires only where the graph would otherwise be
  refused, so a draft that already passes canonicalizes to itself; and
* types come from the vocabulary, never from the planner.  A produced concept is typed by the
  transformation that produces it, which is the same table G3 validates against.

What it deliberately does not do is invent transformations.  A graph that measures a route and
calls the result an AMOUNT is retyped, not given the ROUTE_EXTRACT it skipped: the planner's
structure is the planner's, and a wrong structure has to fail loudly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.agent.concepts import (
    CONTEXTUAL_ROLES,
    ConceptNode,
    FactorNode,
    GeoFlowGraph,
    TransformationEdge,
    factor_nodes_from_concepts,
)
from src.agent.semantics import TRANSFORMS
from src.agent.validation import accepted_output_types

#: Procedural precedence, as G2 orders it. Contextual roles sit outside it by design.
_ROLE_ORDER = ("sub_condition", "condition", "support", "measure")


def canonicalize_ir_payload(
    analysis: Mapping[str, Any], payload: Mapping[str, Any]
) -> GeoFlowGraph:
    """Complete a drafted Concept/Edge IR graph into one strict G1-G5 can decide."""

    builder = _Canonicalizer(analysis, payload)
    return builder.build()


class _Canonicalizer:
    def __init__(self, analysis: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        self._analysis = analysis
        self._payload = payload
        self._concepts: dict[str, ConceptNode] = {}
        self._factors: dict[str, FactorNode] = {}
        self._edges: list[TransformationEdge] = []
        self._outputs_by_edge: dict[str, tuple[str, ...]] = {}
        self._producer: dict[str, str] = {}
        #: What a written concept id denotes from here on. A second transformation claiming a
        #: concept the first already produced is a redefinition, so later readers get the later
        #: value -- the same rule a program's second assignment follows.
        self._alias: dict[str, str] = {}

    # -- concept and factor pools ------------------------------------------------------------

    def _add_concept(self, node: ConceptNode) -> str:
        """Register a concept, keeping the first spelling of any id."""

        self._concepts.setdefault(node.id, node)
        return node.id

    def _add_factor(self, node: FactorNode) -> str:
        self._factors.setdefault(node.id, node)
        return node.id

    def _seed(self) -> None:
        """The planner's own nodes first, then the Analysis stage's, then derived factors.

        The planner wins every id it declares. Seeding it first is what makes this completion
        additive: a draft that named all of its own concepts is untouched by the Analysis
        overlay, and only the ids it referenced without declaring are filled in.
        """

        declared = self._payload.get("concept_nodes") or self._payload.get("concepts") or ()
        for index, raw in enumerate(declared):
            if isinstance(raw, Mapping):
                self._add_concept(ConceptNode.from_dict(raw, fallback_id=f"c{index + 1}"))
        analysed = [
            ConceptNode.from_dict(raw, fallback_id=f"a{index + 1}")
            for index, raw in enumerate(self._analysis.get("concepts") or ())
            if isinstance(raw, Mapping)
        ]
        for node in analysed:
            self._add_concept(node)

        planner_factors = self._payload.get("factor_nodes") or self._payload.get("factors") or ()
        if isinstance(planner_factors, Mapping):
            planner_factors = [
                {"id": f"factor_{key}", "factor_type": key, "value": value}
                for key, value in planner_factors.items()
            ]
        for index, raw in enumerate(planner_factors):
            if isinstance(raw, Mapping):
                self._add_factor(FactorNode.from_dict(raw, fallback_id=f"f{index + 1}"))
        for factor in factor_nodes_from_concepts(tuple(self._concepts.values())):
            self._add_factor(factor)

    # -- reference resolution ----------------------------------------------------------------

    def _resolve(self, value: Any, *, edge_id: str) -> tuple[str, ...]:
        """What one written reference denotes: concepts, or nothing.

        A reference may be an id, a `$node.path` expression, the id of an earlier edge -- the
        planner routinely wires edges to edges rather than to their concepts -- or an inline
        object. Anything that resolves to none of those is dropped, which is what the
        step-shaped path has always done with a dangling input.
        """

        if isinstance(value, Mapping):
            if value.get("factor_type"):
                return ()
            return (self._add_concept(ConceptNode.from_dict(value, fallback_id=f"{edge_id}_in")),)
        key = str(value).lstrip("$").split(".", 1)[0]
        if key in self._concepts:
            return (self._alias.get(key, key),)
        if key in self._outputs_by_edge:
            return self._outputs_by_edge[key]
        return ()

    def _resolve_factor(self, value: Any, *, edge_id: str, position: int) -> tuple[str, ...]:
        """A factor reference, or an inline factor object the planner wrote in place of one."""

        if isinstance(value, Mapping):
            return (
                self._add_factor(
                    FactorNode.from_dict(value, fallback_id=f"factor_{edge_id}_{position}")
                ),
            )
        key = str(value).lstrip("$").split(".", 1)[0]
        if key in self._factors:
            return (key,)
        return ()

    # -- edges -------------------------------------------------------------------------------

    def _raw_edges(self) -> list[Mapping[str, Any]]:
        raw = self._payload.get("transformation_edges") or self._payload.get("edges") or ()
        return [item for item in raw if isinstance(item, Mapping)]

    def _synthesize(self, concept_id: str, transform: str, *, role: str) -> ConceptNode:
        """A concept the graph refers to but never declared, typed by what produces it.

        Its text is deliberately empty. A concept's text is where grounding reads place names
        from, so describing this one as `RESOLVE_PLACES result` put that phrase into a geocoder
        and came back `PlaceNotFoundError: No place matched 'RESOLVE_PLACES result'`. A concept
        the planner never named has no name, and saying so is what keeps it out of a lookup.
        """

        declared = TRANSFORMS.get(transform.upper())
        return ConceptNode(
            concept_id,
            declared.output_type if declared is not None else "object",
            role,
            {"source": "transformation_output", "transformation": transform},
            "",
            True,
        )

    def _build_edges(self) -> None:
        raw_edges = self._raw_edges()
        promote_to_measure: set[str] = set()
        for position, raw in enumerate(raw_edges):
            edge_id = str(raw.get("id") or f"t{position + 1}")
            transform = str(raw.get("transformation") or raw.get("transform") or "").upper()
            if not transform:
                raise ValueError(f"GeoFlow node {edge_id} names no transformation")

            declared_inputs = (
                raw.get("input_concepts") or raw.get("inputs") or raw.get("depends_on") or ()
            )
            inputs: list[str] = []
            for value in declared_inputs:
                inputs.extend(self._resolve(value, edge_id=edge_id))

            if transform == "MATCH_OPTIONS":
                # MCQ matching left the reasoning core when `MCQAdapter` was split out. A planner
                # that still writes it is naming the measure it wanted, so keep that and drop the
                # edge, exactly as the step-shaped path does.
                promote_to_measure.update(inputs)
                continue

            factor_ids: list[str] = []
            declared_factors = raw.get("factor_nodes") or raw.get("factors") or ()
            if isinstance(declared_factors, Mapping):
                declared_factors = [
                    {"id": f"factor_{edge_id}_{key}", "factor_type": key, "value": value}
                    for key, value in declared_factors.items()
                ]
            for index, value in enumerate(declared_factors):
                factor_ids.extend(self._resolve_factor(value, edge_id=edge_id, position=index))
            # A factor the planner wrote among the input concepts is a factor, not a concept.
            for index, value in enumerate(declared_inputs):
                if isinstance(value, Mapping) and value.get("factor_type"):
                    factor_ids.extend(
                        self._resolve_factor(value, edge_id=edge_id, position=index)
                    )
                    continue
                if isinstance(value, str):
                    key = value.lstrip("$").split(".", 1)[0]
                    if key not in self._concepts and key in self._factors:
                        factor_ids.append(key)

            is_last = position == len(raw_edges) - 1
            outputs = self._edge_outputs(
                raw, edge_id, transform, is_last=is_last, inputs=tuple(dict.fromkeys(inputs))
            )

            self._edges.append(
                TransformationEdge(
                    edge_id,
                    transform,
                    tuple(dict.fromkeys(inputs)),
                    outputs,
                    tuple(dict.fromkeys(factor_ids)),
                    raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {},
                )
            )
            self._outputs_by_edge[edge_id] = outputs
            for concept_id in outputs:
                self._producer[concept_id] = edge_id

        for concept_id in promote_to_measure:
            self._reroll(concept_id, role="measure")

    def _edge_outputs(
        self,
        raw: Mapping[str, Any],
        edge_id: str,
        transform: str,
        *,
        is_last: bool,
        inputs: tuple[str, ...],
    ) -> tuple[str, ...]:
        declared = raw.get("output_concepts") or raw.get("concept_ids") or ()
        if isinstance(declared, str):
            declared = [declared]
        role = "measure" if is_last else "support"
        outputs: list[str] = []
        for value in declared:
            if isinstance(value, Mapping):
                concept_id = self._add_concept(
                    ConceptNode.from_dict(value, fallback_id=f"{edge_id}_out")
                )
            else:
                concept_id = str(value).lstrip("$").split(".", 1)[0]
            if concept_id not in self._concepts:
                self._add_concept(self._synthesize(concept_id, transform, role=role))
            if concept_id in self._producer:
                # Two transformations claiming one concept is a G4 refusal. Neither claim is
                # discarded: the second producer gets its own concept, typed by the
                # transformation that produces it, and every later reader is pointed at it. A
                # planner that writes this is refining the first result, and refinement is what
                # a redefinition means.
                renamed = self._add_concept(
                    self._synthesize(f"{concept_id}__{edge_id}", transform, role=role)
                )
                self._alias[concept_id] = renamed
                concept_id = renamed
            outputs.append(concept_id)
        if not outputs:
            outputs.append(
                self._add_concept(self._synthesize(f"result_{edge_id}", transform, role=role))
            )
        return tuple(dict.fromkeys(outputs))

    # -- last-resort completions -------------------------------------------------------------

    def _reroll(self, concept_id: str, *, role: str | None = None, core: str | None = None) -> None:
        node = self._concepts.get(concept_id)
        if node is None:
            return
        self._concepts[concept_id] = replace(
            node,
            functional_role=role or node.functional_role,
            core_concept=core or node.core_concept,
            attributes={
                **node.attributes,
                **({"role_completed_from": node.functional_role} if role else {}),
                **({"type_completed_from": node.core_concept} if core else {}),
            },
        )

    def _retype_produced_concepts(self) -> None:
        """A produced concept is typed by what produces it, not by what the planner guessed."""

        for edge in self._edges:
            accepted = accepted_output_types(edge.transformation)
            if not accepted:
                continue
            for concept_id in edge.output_concepts:
                node = self._concepts[concept_id]
                if node.core_concept in accepted:
                    continue
                declared = TRANSFORMS[edge.transformation.upper()].output_type
                self._reroll(concept_id, core=declared)

    def _propagate_roles(self) -> None:
        """No concept may be less advanced than the concept it was computed from (G2)."""

        rank = {role: index for index, role in enumerate(_ROLE_ORDER)}
        for edge in self._edges:
            highest = max(
                (
                    rank[self._concepts[value].functional_role]
                    for value in edge.input_concepts
                    if self._concepts[value].functional_role in rank
                ),
                default=-1,
            )
            if highest < 0:
                continue
            for concept_id in edge.output_concepts:
                role = self._concepts[concept_id].functional_role
                if role in rank and rank[role] < highest:
                    self._reroll(concept_id, role=_ROLE_ORDER[highest])

    def _root_the_leaves_in_context(self) -> None:
        """A concept no transformation produces is data the question gave. That is context.

        G4 asks whether every transformation's data is available and answers it by reachability
        from an edge rooted in a contextual role, so a leaf the Analysis stage happened to label
        `support` strands its whole branch. The question supplied `target1` exactly as it
        supplied `anchor`; only the label differed, and 43 of one run's refusals were that label.
        Promoting every leaf -- rather than one, and rather than only when no context exists at
        all -- is what makes the rule uniform instead of first-one-wins.
        """

        for edge in self._edges:
            for value in edge.input_concepts:
                # A concept produced by the very edge that reads it -- `RESOLVE_PLACES` over
                # `target1` into `target1`, the planner saying "resolve it in place" -- is not
                # supplied by the graph. The question supplied it, so it is a leaf. Renaming it
                # instead was tried and is wrong: the executor's references are built on these
                # ids, and 28 graphs then failed to resolve a `$node.place` path.
                if self._producer.get(value) not in (None, edge.id):
                    continue
                if self._concepts[value].functional_role in CONTEXTUAL_ROLES:
                    continue
                self._reroll(value, role="extent")

    def _ensure_measure(self) -> None:
        """G5 needs an answer. The sinks are what the graph computed last."""

        if any(
            self._concepts[value].functional_role == "measure"
            for edge in self._edges
            for value in edge.output_concepts
        ):
            return
        consumed = {value for edge in self._edges for value in edge.input_concepts}
        for edge in reversed(self._edges):
            unconsumed = [value for value in edge.output_concepts if value not in consumed]
            if unconsumed:
                self._reroll(unconsumed[0], role="measure")
                return
        if self._edges:
            self._reroll(self._edges[-1].output_concepts[0], role="measure")

    # -- assembly ----------------------------------------------------------------------------

    def build(self) -> GeoFlowGraph:
        self._seed()
        self._build_edges()
        if not self._edges:
            raise ValueError("GeoFlow response does not contain a non-empty graph")
        self._retype_produced_concepts()
        self._propagate_roles()
        self._root_the_leaves_in_context()
        self._ensure_measure()
        metadata = self._payload.get("metadata")
        graph = GeoFlowGraph(
            tuple(self._concepts.values()),
            tuple(self._edges),
            tuple(self._factors.values()),
            metadata if isinstance(metadata, dict) else {},
        )
        return graph.with_implicit_concepts()
