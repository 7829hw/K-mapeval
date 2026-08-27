"""Measurement shape is inferred from the resolved graph, and an ablation says why.

The planner was given `SET_MEASURE` and `PAIRWISE_MEASURE` so it could state which it meant
instead of the factorizer inferring it. It went worse, twice, and the ablation is worth keeping
because the reasoning that motivated it is the kind that recurs:

    explicit set-vs-pair vocabulary   overall 80.4 -> 75.8, `nearby_kth_nearest` 80.6 -> 44.4
    binding ambiguity                 43 of 297 rows gave no way to tell anchor from candidates,
                                      23 of them in `nearby_kth_nearest` alone
    order-independent binding          candidate collapse 25 -> 14, accuracy 77.1 -> 75.8

So the mechanical defect the second fix targeted was real and fixing it did not restore
correctness. What made the old mapping work was not that it guessed well but that its guard is a
precondition: `nearest` is reached only when the first input already resolved to a single place
and a later one to a set. `SET_MEASURE` chose `nearest` on the planner's word and removed that
precondition; binding by what each input produces did not put it back.

Do not reintroduce an explicit set/pair distinction over positional `inputs`. If it returns, it
needs typed roles -- `{"anchor": ..., "candidates": ...}` -- so there is nothing to infer.
"""

from __future__ import annotations

from src.agent.geoflow import OPERATOR_CONTRACTS, SKELETONS
from src.agent.semantics import TRANSFORMS, factorize_semantic_graph

ALL = frozenset(OPERATOR_CONTRACTS)

ANCHOR = {"id": "anchor", "text": "종로3가역 1호선", "concept_type": "location", "role": "extent"}
CANDIDATES = [
    {"id": f"c{index}", "text": name, "concept_type": "location", "role": "support"}
    for index, name in enumerate(["종로스시야", "송사부수제쌀고로케", "한일식당"])
]


def _build(graph, *, concepts, options=("가", "나"), facts=None):
    return factorize_semantic_graph(
        graph, concepts=list(concepts), options=list(options), facts=facts, available=ALL
    )


def test_the_ablated_vocabulary_is_gone() -> None:
    assert "SET_MEASURE" not in TRANSFORMS
    assert "PAIRWISE_MEASURE" not in TRANSFORMS
    assert "DISTANCE_MEASURE" in TRANSFORMS


def test_no_skeleton_or_pattern_asks_for_it() -> None:
    import json

    from src.agent.geoflow import TEMPLATES

    shown = json.dumps(SKELETONS, ensure_ascii=False) + json.dumps(
        [template["pattern"] for template in TEMPLATES.values()], ensure_ascii=False
    )

    assert "SET_MEASURE" not in shown
    assert "PAIRWISE_MEASURE" not in shown


def test_an_anchor_against_a_set_is_a_ranking() -> None:
    """Inferred from what resolved, which is the mapping the ablation kept."""

    built = _build(
        [
            {"id": "anchor_res", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor"], "role": "extent"},
            {"id": "candidates_res", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": [concept["id"] for concept in CANDIDATES]},
            {"id": "measured", "transform": "DISTANCE_MEASURE",
             "inputs": ["anchor_res", "candidates_res"]},
        ],
        concepts=[ANCHOR, *CANDIDATES],
    )

    measured = built.graph[2]
    assert measured["operator"] == "nearest"
    assert measured["arguments"] == {
        "anchor": "$anchor_res.0.place",
        "candidates": "$candidates_res",
    }


def test_two_single_places_are_a_pair() -> None:
    built = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["anchor"],
             "role": "extent"},
            {"id": "b", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["c0"]},
            {"id": "span", "transform": "DISTANCE_MEASURE", "inputs": ["a", "b"]},
        ],
        concepts=[ANCHOR, *CANDIDATES],
    )

    assert built.graph[2]["operator"] == "haversine_distance"
    assert built.graph[2]["arguments"] == {"place_a": "$a.0.place", "place_b": "$b.0.place"}


def test_a_set_written_first_is_not_read_as_an_anchor() -> None:
    """The guard is a precondition, not a guess: it does not reach `nearest` at all here.

    This is the shape that broke under `SET_MEASURE`, which chose `nearest` regardless and then
    measured the anchor against itself. Inferring the shape declines the ranking instead.
    """

    built = _build(
        [
            {"id": "anchor_res", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor"], "role": "extent"},
            {"id": "candidates_res", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": [concept["id"] for concept in CANDIDATES]},
            {"id": "measured", "transform": "DISTANCE_MEASURE",
             "inputs": ["candidates_res", "anchor_res"]},
        ],
        concepts=[ANCHOR, *CANDIDATES],
    )

    assert built.graph[2]["operator"] != "nearest"
