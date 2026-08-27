"""A stated restriction has to reach the answer, and the graph has to say where it applies.

Two measurements shaped this file, in order. Requiring a `FILTER` node was violated by 45 of 47
recorded `Search-Narrow-Rank` rows and 37 of those answered correctly, because the kind rides on
`nearest` as readily as on a filter -- so presence of a transformation is not enforcement of a
constraint, and what is measured is the constraint. Then *refusing* a graph over either departure
was tried and cost more than it caught: `nearby_kth_nearest` lost 27.8 points, validation fell
below 90%, and seven rows executed nothing at all, because repair does not fix a malformed
reference. Roughly three answerable questions were lost for every unrestricted answer prevented.

So both are diagnostics. They are emitted, they travel with the result, and the graph runs. What
they are for is the architectural claim -- whether a stated restriction reaches the answer is
measurable, and that is worth having whether or not it is worth blocking on.
"""

from __future__ import annotations

from src.agent.geoflow import OPERATOR_CONTRACTS
from src.agent.semantics import (
    constraint_concepts,
    factorize_semantic_graph,
    resolve_operator,
)
from src.agent.spatial import (
    GroundingFacts,
    _ground_graph_literals,
    _unpreserved_constraints,
)

ALL = frozenset(OPERATOR_CONTRACTS)

ANCHOR = {"id": "anchor", "text": "대림역 2호선", "concept_type": "location", "role": "extent"}
KIND = {"id": "kind", "text": "중식 음식점", "concept_type": "object", "role": "sub_condition"}
OPTIONS = ["진달래장국", "일일양꼬치", "맛있는떡", "일품채관"]
FACTS = GroundingFacts(anchor="대림역 2호선", target_type="음식점", target_subtype="중식")


def _build(graph, *, concepts, options=OPTIONS, facts=FACTS):
    return factorize_semantic_graph(
        graph, concepts=list(concepts), options=list(options), facts=facts, available=ALL
    )


# ---------------------------------------------------------------------------------------------
# Saying which kind of measure it is
# ---------------------------------------------------------------------------------------------


def test_a_set_measure_ranks_a_set_however_many_inputs_it_has() -> None:
    """`DISTANCE_MEASURE` guesses from the shape of its inputs; `SET_MEASURE` is told."""

    assert resolve_operator(
        "SET_MEASURE", {}, input_types=["object", "object"], facts=None, available=ALL
    ) == ("nearest", "_a_set")


def test_a_pairwise_measure_relates_two_places_and_says_nothing_about_a_set() -> None:
    assert resolve_operator(
        "PAIRWISE_MEASURE", {}, input_types=["object", "object"], facts=None, available=ALL
    )[0] == "haversine_distance"


def test_a_set_measure_over_a_retrieval_carries_the_stated_kind() -> None:
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor"], "role": "extent"},
            {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
            {"id": "ranked", "transform": "SET_MEASURE", "inputs": ["anchor", "found"]},
            {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["ranked"],
             "role": "measure"},
        ],
        concepts=[ANCHOR, KIND],
    )
    grounded = _ground_graph_literals(built.graph, "질문", OPTIONS, FACTS)

    ranking = next(step for step in grounded if step["operator"] == "nearest")
    assert ranking["arguments"]["required_type"] == "중식"
    assert _unpreserved_constraints(grounded, FACTS) == []


# ---------------------------------------------------------------------------------------------
# Where the restriction applies
# ---------------------------------------------------------------------------------------------


def test_a_restriction_is_a_concept_and_naming_it_as_an_input_is_not_a_dangling_reference() -> (
    None
):
    """The planner already writes `FILTER(inputs=["measured", "constraint"])`, and it is right
    to: the restriction is a thing the question stated and the node is where it applies."""

    assert set(constraint_concepts([ANCHOR, KIND])) == {"kind"}

    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor"], "role": "extent"},
            {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
            {"id": "narrowed", "transform": "FILTER", "inputs": ["found", "kind"],
             "factors": {"scope": "attribute"}},
            {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["narrowed"],
             "role": "measure"},
        ],
        concepts=[ANCHOR, KIND],
    )

    narrowing = built.graph[2]
    assert narrowing["operator"] == "filter_places"
    assert narrowing["constraint_ids"] == ["kind"]
    # A constraint is not a data dependency, so it is not gathered as one.
    assert narrowing["depends_on"] == ["found"]


def test_an_input_that_names_nothing_is_reported_and_the_graph_still_runs() -> None:
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor"], "role": "extent"},
            {"id": "m", "transform": "PAIRWISE_MEASURE", "inputs": ["anchor", "candidate_0"]},
        ],
        concepts=[ANCHOR, KIND],
    )

    reported = [row for row in built.diagnostics if row["kind"] == "invalid_semantic_reference"]
    assert len(reported) == 1
    assert reported[0]["node"] == "m"
    assert reported[0]["reference"] == "candidate_0"
    assert reported[0]["transform"] == "PAIRWISE_MEASURE"
    # And the graph is intact: the reference to nothing is dropped, as it always was, rather
    # than taking the node and the question with it.
    assert [step["operator"] for step in built.graph] == [
        "batch_geocode",
        # One usable input left, so the pair becomes a cross-join over what there is -- which is
        # exactly the collapse the diagnostic is there to make visible.
        "pairwise_distances",
    ]


# ---------------------------------------------------------------------------------------------
# Preservation
# ---------------------------------------------------------------------------------------------


def test_a_pairwise_graph_that_cannot_hold_the_stated_kind_is_reported() -> None:
    """Measuring the candidates one pair at a time leaves the restriction nowhere to sit.

    Every step succeeds and the answer comes back unrestricted -- the nearest restaurant of any
    kind, on a question that asked for the nearest 중식 one. Worth counting; not worth refusing,
    which was measured.
    """

    concepts = [
        ANCHOR,
        KIND,
        *(
            {"id": f"c{index}", "text": name, "concept_type": "location", "role": "support"}
            for index, name in enumerate(OPTIONS)
        ),
    ]
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor"], "role": "extent"},
            *(
                {"id": f"r{index}", "transform": "RESOLVE_PLACES", "inputs": [],
                 "concept_ids": [f"c{index}"]}
                for index in range(4)
            ),
            *(
                {"id": f"m{index}", "transform": "PAIRWISE_MEASURE",
                 "inputs": ["anchor", f"r{index}"]}
                for index in range(4)
            ),
            {"id": "pick", "transform": "EXTREME_SELECT",
             "inputs": [f"m{index}" for index in range(4)], "factors": {"extreme": "min"}},
            {"id": "answer", "transform": "MATCH_OPTIONS", "inputs": ["pick"], "role": "measure"},
        ],
        concepts=concepts,
    )
    grounded = _ground_graph_literals(built.graph, "질문", OPTIONS, FACTS)

    unheld = _unpreserved_constraints(grounded, FACTS)
    assert len(unheld) == 1
    assert unheld[0]["kind"] == "constraint_unpreserved"
    assert unheld[0]["value"] == "중식"
    assert unheld[0]["carried_by"] == []
    # Reported, and the graph is one the executor can still run end to end.
    assert [step["operator"] for step in grounded][-1] == "match_options"


def test_a_restriction_the_question_does_not_state_is_not_required() -> None:
    grounded = [{"id": "a", "operator": "batch_geocode", "arguments": {}, "role": "measure"}]

    assert _unpreserved_constraints(grounded, GroundingFacts(anchor="가")) == []


def test_a_carrier_that_never_reaches_the_measure_does_not_count() -> None:
    """A narrowing on a branch the answer is not read from is a narrowing of nothing."""

    grounded = [
        {"id": "a", "operator": "batch_geocode", "arguments": {}, "depends_on": []},
        {"id": "side", "operator": "nearest",
         "arguments": {"candidates": "$a", "required_type": "중식"}, "depends_on": ["a"]},
        {"id": "answer", "operator": "match_options", "arguments": {"places": "$a"},
         "depends_on": ["a"], "role": "measure"},
    ]

    reported = _unpreserved_constraints(grounded, FACTS)
    assert len(reported) == 1
    assert reported[0]["reaches_measure"] is True and reported[0]["carried_by"] == ["side"]
