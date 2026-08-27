"""Which input is the anchor is a property of what it produces, not of where it was written.

`nearest` was wired positionally -- first input the anchor, the rest the candidates. Under
`DISTANCE_MEASURE` that was survivable: the guard only chose `nearest` when the first input was
already a single place. `SET_MEASURE` means "a set measured against an anchor" and selects
`nearest` outright, so a planner writing the two the other way round -- which it does -- got

    nearest(anchor=$filtered_candidates.0.place, candidates=<the anchor>)

the anchor measured against itself, and a four-candidate set arriving at the narrowing as one
place. A correctly extracted, correctly attached, correctly enforced subtype then narrowed a set
of one. It cost about 3.3 points on a 99-row subset and every stage reported success.
"""

from __future__ import annotations

from src.agent.geoflow import OPERATOR_CONTRACTS
from src.agent.semantics import factorize_semantic_graph

ALL = frozenset(OPERATOR_CONTRACTS)

ANCHOR = {"id": "anchor", "text": "종로3가역 1호선", "concept_type": "location", "role": "extent"}
CANDIDATES = [
    {"id": f"c{index}", "text": name, "concept_type": "location", "role": "support"}
    for index, name in enumerate(["종로스시야", "송사부수제쌀고로케", "한일식당"])
]
OPTIONS = ["종로스시야", "송사부수제쌀고로케", "한일식당", "알 수 없음"]


def _build(graph, *, concepts, options=OPTIONS, facts=None):
    return factorize_semantic_graph(
        graph, concepts=list(concepts), options=list(options), facts=facts, available=ALL
    )


def _resolvers():
    return [
        {"id": "anchor_res", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["anchor"], "role": "extent"},
        {"id": "candidates_res", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": [concept["id"] for concept in CANDIDATES], "role": "support"},
    ]


def _measure(inputs, transform="SET_MEASURE"):
    built = _build(
        [*_resolvers(), {"id": "measured", "transform": transform, "inputs": inputs}],
        concepts=[ANCHOR, *CANDIDATES],
    )
    return next(step for step in built.graph if step["id"] == "measured")


# ---------------------------------------------------------------------------------------------
# Order invariance
# ---------------------------------------------------------------------------------------------


def test_a_set_measure_binds_the_same_either_way_round() -> None:
    forwards = _measure(["anchor_res", "candidates_res"])
    backwards = _measure(["candidates_res", "anchor_res"])

    assert forwards["operator"] == backwards["operator"] == "nearest"
    assert forwards["arguments"] == backwards["arguments"]
    assert forwards["arguments"] == {
        "anchor": "$anchor_res.0.place",
        "candidates": "$candidates_res",
    }


def test_the_observed_failure_is_pinned() -> None:
    """`SET_MEASURE(inputs=['filtered_candidates', 'anchor'])`, exactly as recorded."""

    built = _build(
        [
            *_resolvers(),
            {"id": "filtered_candidates", "transform": "FILTER", "inputs": ["candidates_res"]},
            {"id": "measured", "transform": "SET_MEASURE",
             "inputs": ["filtered_candidates", "anchor_res"]},
        ],
        concepts=[ANCHOR, *CANDIDATES],
    )

    measured = next(step for step in built.graph if step["id"] == "measured")
    assert measured["arguments"] == {
        "anchor": "$anchor_res.0.place",
        "candidates": "$filtered_candidates",
    }


def test_the_anchor_is_the_singleton_even_among_several_candidate_sets() -> None:
    built = _build(
        [
            *_resolvers(),
            {"id": "more", "transform": "PLACE_SEARCH", "inputs": ["anchor_res"]},
            {"id": "measured", "transform": "SET_MEASURE",
             "inputs": ["candidates_res", "more", "anchor_res"]},
        ],
        concepts=[ANCHOR, *CANDIDATES],
    )

    measured = next(step for step in built.graph if step["id"] == "measured")
    assert measured["arguments"]["anchor"] == "$anchor_res.0.place"
    assert measured["arguments"]["candidates"] == ["$candidates_res", "$more"]


# ---------------------------------------------------------------------------------------------
# Ambiguity is reported, not guessed at silently
# ---------------------------------------------------------------------------------------------


def test_two_singletons_and_no_set_is_reported() -> None:
    built = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["anchor"],
             "role": "extent"},
            {"id": "b", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["c0"]},
            {"id": "measured", "transform": "SET_MEASURE", "inputs": ["a", "b"]},
        ],
        concepts=[ANCHOR, *CANDIDATES],
    )

    reported = [
        row for row in built.diagnostics if row["kind"] == "set_measure_binding_ambiguous"
    ]
    assert len(reported) == 1
    assert reported[0]["node"] == "measured"
    assert reported[0]["collections"] == []
    # Reported, and still wired: the fallback is positional and documented, not silent.
    measured = next(step for step in built.graph if step["id"] == "measured")
    assert measured["arguments"]["anchor"] == "$a.0.place"


def test_an_unambiguous_binding_reports_nothing() -> None:
    built = _build(
        [*_resolvers(), {"id": "measured", "transform": "SET_MEASURE",
                         "inputs": ["candidates_res", "anchor_res"]}],
        concepts=[ANCHOR, *CANDIDATES],
    )

    assert [row for row in built.diagnostics if "binding" in row["kind"]] == []


# ---------------------------------------------------------------------------------------------
# A pair is not a set
# ---------------------------------------------------------------------------------------------


def test_a_pairwise_measure_keeps_both_places_whichever_order_they_are_in() -> None:
    """No anchor and no candidates: two places, and the pair is what it says it is."""

    concepts = [ANCHOR, CANDIDATES[0]]
    forwards = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["anchor"],
             "role": "extent"},
            {"id": "b", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["c0"]},
            {"id": "span", "transform": "PAIRWISE_MEASURE", "inputs": ["a", "b"]},
        ],
        concepts=concepts,
    )
    backwards = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["anchor"],
             "role": "extent"},
            {"id": "b", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["c0"]},
            {"id": "span", "transform": "PAIRWISE_MEASURE", "inputs": ["b", "a"]},
        ],
        concepts=concepts,
    )

    first = next(step for step in forwards.graph if step["id"] == "span")
    second = next(step for step in backwards.graph if step["id"] == "span")
    assert first["operator"] == second["operator"] == "haversine_distance"
    # The pair is the same pair; which end is `place_a` follows the order the graph wrote, and
    # a separation is symmetric, so both are correct measurements of the same thing.
    assert set(first["arguments"].values()) == set(second["arguments"].values())
    assert set(first["arguments"].values()) == {"$a.0.place", "$b.0.place"}


def test_a_pairwise_measure_is_not_given_an_anchor_and_a_set() -> None:
    span = _measure(["anchor_res", "candidates_res"], transform="PAIRWISE_MEASURE")

    assert span["operator"] == "haversine_distance"
    assert "candidates" not in span["arguments"]
