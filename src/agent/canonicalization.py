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

#: Transformations that need a *located* place, not a kind of place or a set of them. A
#: PLACE_SEARCH takes a category and a FILTER takes candidates, so neither is here: geocoding
#: `정형외과` because a search reads it would send the geocoder after a word.
_NEEDS_LOCATED_PLACES = frozenset(
    {"ROUTE_MEASURE", "ROUTE_MATRIX", "ROUTE_OPTIMIZE", "DISTANCE_MEASURE"}
)
#: And the narrowings that measure: `within_radius` needs its candidates located as much as a
#: route needs its ends. Kept separate because only a LOCATION-typed input qualifies here -- a
#: FILTER reads a kind of place as often as a list of them.
_NARROWS_BY_DISTANCE = frozenset({"FILTER"})
#: The core concepts a place is typed with.
_PLACE_TYPES = frozenset({"location", "object"})
#: Factor types that make a narrowing a *distance* narrowing, which needs somewhere to measure
#: from. The question states the radius and grounding binds it; what the graph has to supply is
#: the place it is measured around.
_RADIUS_FACTORS = frozenset({"radius_m", "radius"})


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

            attributes = raw.get("attributes")
            attributes = dict(attributes) if isinstance(attributes, dict) else {}
            # `via` says what a route passes through, and the graph is the only thing allowed to
            # say it -- a rule that made a middle input into a waypoint would route
            # "A와 B 중 C에 더 가까운 곳" through the answer. A planner writes it beside the
            # transformation as often as inside `attributes`, and reading only the second
            # spelling lost every waypoint: `routing_turn_count_via` counted the turns of a
            # route that skipped the stop it was asked about.
            declared_via = raw.get("via") or attributes.get("via") or ()
            if isinstance(declared_via, str):
                declared_via = [declared_via]
            resolved_via = [
                value
                for entry in declared_via
                for value in self._resolve(entry, edge_id=edge_id)
            ]
            if resolved_via:
                attributes["via"] = list(dict.fromkeys(resolved_via))
            self._edges.append(
                TransformationEdge(
                    edge_id,
                    transform,
                    tuple(dict.fromkeys(inputs)),
                    outputs,
                    tuple(dict.fromkeys(factor_ids)),
                    attributes,
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

    def _resolve_the_places_nothing_resolved(self) -> None:
        """Geocode the places a measure reads when the graph never said to.

        A drive needs located places, and a graph that writes ROUTE_MEASURE straight over the
        Analysis concepts has named them without resolving them: `directions` then arrives with
        no origin and no destination and the question is refused before it starts. On a held-out
        draw that was 31 of 100 questions, across four different operators saying it four
        different ways.

        Additive and last-resort like the rest of this module: it fires only for a concept no
        transformation produces, and the edge it adds is the one a planner writes when it gets
        this right -- one RESOLVE_PLACES reading those concepts and producing them in place.
        """

        wanted: list[str] = []
        for edge in self._edges:
            transform = edge.transformation.upper()
            needs_places = transform in _NEEDS_LOCATED_PLACES
            narrows = transform in _NARROWS_BY_DISTANCE
            for concept_id in edge.input_concepts:
                node = self._concepts.get(concept_id)
                if node is None or concept_id in self._producer or concept_id in wanted:
                    continue
                # LOCATION is a particular place, whatever reads it -- the four candidates a
                # radius question lists are typed that way, and unresolved they leave
                # `within_radius` with nothing to measure. OBJECT is a *kind* of place as often
                # as a set of them, so it is resolved only where a located place is required:
                # geocoding `정형외과` because a search reads it would send the geocoder after a
                # word.
                if (narrows and node.core_concept == "location") or (
                    needs_places and node.core_concept in _PLACE_TYPES
                ):
                    wanted.append(concept_id)
        if not wanted:
            return
        edge_id = "resolve_places"
        while edge_id in self._outputs_by_edge:
            edge_id += "_"
        resolved = TransformationEdge(
            edge_id,
            "RESOLVE_PLACES",
            tuple(wanted),
            tuple(wanted),
            (),
            {"source": "implicit_completion"},
        )
        self._edges.insert(0, resolved)
        self._outputs_by_edge[edge_id] = tuple(wanted)
        for concept_id in wanted:
            self._producer[concept_id] = edge_id

    def _give_a_radius_filter_what_it_narrows(self) -> None:
        """A narrowing by radius reads what was measured, and the graph has to hand it over.

        "X에서 반경 300m 이내에 있는 은행은 아래 목록 중 몇 곳인가요" is drawn two ways and both
        strand a branch. One measures each candidate from the anchor and then filters the
        *candidates*, so every distance it computed is read by nothing and G5 refuses the graph
        before grounding can say `FILTER by radius has no place to measure from`. The other
        filters the listed places with no anchor anywhere in reach.

        `filter_by_distance` says which is which: the radius is a stated fact that grounding
        binds, and the measured set is whatever the measure steps produced. So the filter reads
        the measurements the graph computed and left unread, or -- when it computed none -- the
        located extent it would have measured from.

        Fires only when the question states a radius, and only on a filter that holds neither,
        so a narrowing that is already wired is untouched.
        """

        if not any(factor.factor_type in _RADIUS_FACTORS for factor in self._factors.values()):
            return
        consumed = {value for edge in self._edges for value in edge.input_concepts}
        stranded = [
            value
            for edge in self._edges
            if edge.transformation.upper() == "DISTANCE_MEASURE"
            for value in edge.output_concepts
            if value not in consumed
        ]
        leading = False
        if not stranded:
            leading = True
            anchor = next(
                (
                    node.id
                    for node in self._concepts.values()
                    if node.functional_role in CONTEXTUAL_ROLES
                    and node.core_concept in _PLACE_TYPES
                ),
                None,
            )
            if anchor is None:
                return
            # The *located* anchor: what resolving it produced, when the graph resolved it in
            # its own node. `anchor` itself is a name until something geocodes it.
            stranded = [
                next(
                    (
                        edge.output_concepts[0]
                        for edge in self._edges
                        if edge.transformation.upper() == "RESOLVE_PLACES"
                        and anchor in edge.input_concepts
                    ),
                    self._alias.get(anchor, anchor),
                )
            ]
        for index, edge in enumerate(self._edges):
            if edge.transformation.upper() != "FILTER":
                continue
            if any(value in edge.input_concepts for value in stranded):
                continue
            # The measured set joins what the filter already narrows; the extent goes in
            # front, because "the place it is measured from" is the first input by convention
            # and `within_radius` reads its centre there.
            self._edges[index] = replace(
                edge,
                input_concepts=(
                    (*stranded, *edge.input_concepts)
                    if leading
                    else (*edge.input_concepts, *stranded)
                ),
            )
            return

    def _read_the_second_route_as_the_one_with_a_stop(self) -> None:
        """Two routes between the same ends differ by what one of them passes through.

        A detour question measures the drive twice, and the graph says so: `route_direct` reads
        the two places, `route_via` reads those two and one more. The difference between them is
        the waypoint, and reading it that way needs no position -- one planner wrote the extra
        input last and another wrote it in the middle, and a positional rule got the second one
        backwards, routing to the waypoint and passing through the destination.

        `AGENTS.md` forbids the general version of this, and rightly: "A와 B 중 C에 더 가까운 곳"
        has a middle too, and there is no second route beside it for this rule to compare
        against. What fires here is a *pair* of ROUTE_MEASUREs whose inputs stand in a subset
        relation, which is the shape "곧장 가는 경우와 X를 경유해서 가는 경우" is drawn in and
        nothing else is. Without it both routes measured the same drive and the detour cost came
        out as zero.
        """

        routes = [
            (index, edge)
            for index, edge in enumerate(self._edges)
            if edge.transformation.upper() == "ROUTE_MEASURE" and not edge.attributes.get("via")
        ]
        for index, edge in routes:
            places = {
                value
                for value in edge.input_concepts
                if self._concepts[value].core_concept in _PLACE_TYPES
            }
            for other_index, other in routes:
                if other_index == index:
                    continue
                ends = {
                    value
                    for value in other.input_concepts
                    if self._concepts[value].core_concept in _PLACE_TYPES
                }
                if not ends or not ends < places:
                    continue
                through = [value for value in edge.input_concepts if value in places - ends]
                if through:
                    # Recorded beside the inputs, not moved out of them: taking the waypoint out
                    # left its resolve node contributing to nothing and G5 refused the graph. The
                    # endpoint reader still gets it wrong on one planner spelling out of four --
                    # `[start, destination, waypoint]` reads the stop as the far end -- and that
                    # is a smaller loss than refusing every detour graph outright.
                    self._edges[index] = replace(
                        edge, attributes={**edge.attributes, "via": through}
                    )
                break

    def build(self) -> GeoFlowGraph:
        self._seed()
        self._build_edges()
        if not self._edges:
            raise ValueError("GeoFlow response does not contain a non-empty graph")
        self._resolve_the_places_nothing_resolved()
        self._give_a_radius_filter_what_it_narrows()
        self._read_the_second_route_as_the_one_with_a_stop()
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
