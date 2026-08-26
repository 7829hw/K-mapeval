"""The LLM says what transformation the question needs; this decides which operator performs it.

`GRAPH_PROMPT` used to hand the planner 47 operator contracts and Kakao's category codes and ask
for `nearby_places(center, category_code="PM9", radius_m=600)`. That put a mechanical question --
which tool computes a spatial relation, given the concept types -- to a language model, once per
question. The paper puts a Concept Transformation stage there instead.

Two properties carry the whole claim, and both are tested here: factorization never sees the
question, and it is deterministic. A third is what earns the deletion of `Retrieve-Rank-Ordinal`:
lifting each macro-template's worked example into transformations and factorizing it back returns
the operators it started from.
"""

from __future__ import annotations

import inspect

import pytest

from src.agent.geoflow import OPERATOR_CONTRACTS, TEMPLATES
from src.agent.semantics import (
    TRANSFORMS,
    factorize_semantic_graph,
    is_semantic_graph,
    lift_to_semantic,
    resolve_operator,
)

ALL = frozenset(OPERATOR_CONTRACTS)


def _build(graph, *, concepts=(), options=("A", "B"), facts=None, available=ALL):
    return factorize_semantic_graph(
        graph, concepts=list(concepts), options=list(options), facts=facts, available=available
    )


def test_factorization_cannot_see_the_question() -> None:
    """Not a convention -- there is no parameter for it."""

    parameters = set(inspect.signature(factorize_semantic_graph).parameters)
    assert "question" not in parameters
    assert parameters == {"steps", "concepts", "options", "facts", "available"}
    assert "question" not in set(inspect.signature(resolve_operator).parameters)


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_macro_template_round_trips_through_the_vocabulary(key: str) -> None:
    """Lift the worked example to transformations, factorize it back, get the same operators."""

    original = TEMPLATES[key]["example"]["graph"]
    rebuilt = _build(lift_to_semantic(original))

    assert [step["operator"] for step in rebuilt.graph] == [
        step["operator"] for step in original
    ]


def test_the_same_graph_always_factorizes_the_same_way() -> None:
    graph = lift_to_semantic(TEMPLATES["route_optimize"]["example"]["graph"])

    assert _build(graph).as_dict() == _build(graph).as_dict()


def test_an_ordinal_is_a_factor_and_the_superlative_is_the_same_graph() -> None:
    """What `Retrieve-Rank-Ordinal` used to be a whole template for."""

    def graph(position: int):
        return [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["c"]},
            {"id": "f", "transform": "PLACE_SEARCH", "inputs": ["a"]},
            {"id": "s", "transform": "SORT", "inputs": ["a", "f"]},
            {"id": "k", "transform": "ORDINAL_SELECT", "inputs": ["s"],
             "factors": {"ordinal": position}},
            {"id": "m", "transform": "MATCH_OPTIONS", "inputs": ["k"], "role": "measure"},
        ]

    concepts = [{"id": "c", "text": "서울역"}]
    second = _build(graph(2), concepts=concepts)
    first = _build(graph(1), concepts=concepts)

    assert [s["operator"] for s in second.graph] == [s["operator"] for s in first.graph]
    assert second.graph[3]["arguments"]["index"] == 1
    assert first.graph[3]["arguments"]["index"] == 0
    # `select_by_index` reads a `nearest` node's ordering, not the node itself.
    assert second.graph[3]["arguments"]["items"] == "$s.ranked"


def test_place_names_come_from_the_concept_graph_not_the_planner() -> None:
    """A planner that retypes a name truncates one. The analysis already extracted them."""

    graph = [
        {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": ["one", "two"], "place_names": ["틀린 이름"]},
    ]
    concepts = [
        {"id": "one", "text": "백련산꿈마을숲정이"},
        {"id": "two", "text": "가좌동 마을극장"},
    ]

    built = _build(graph, concepts=concepts)

    assert built.graph[0]["arguments"]["place_names"] == [
        "백련산꿈마을숲정이",
        "가좌동 마을극장",
    ]


def test_the_candidate_texts_are_asked_for_by_scope_not_retyped() -> None:
    built = _build(
        [{"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "factors": {"scope": "options"}}],
        options=["동묘파출소", "안임지구대"],
    )

    assert built.graph[0]["arguments"]["place_names"] == ["동묘파출소", "안임지구대"]


@pytest.mark.parametrize(
    ("transform", "factors", "arity", "expected"),
    [
        ("PLACE_SEARCH", {}, 1, "nearby_places"),
        ("PLACE_SEARCH", {}, 0, "place_search"),
        ("PLACE_DETAILS", {"scope": "one"}, 1, "place_details"),
        ("PLACE_DETAILS", {}, 1, "batch_place_details"),
        ("DISTANCE_MEASURE", {}, 2, "haversine_distance"),
        ("DISTANCE_MEASURE", {}, 1, "pairwise_distances"),
        ("ROUTE_MEASURE", {"measure": "duration"}, 2, "travel_time"),
        ("ROUTE_MEASURE", {"measure": "distance"}, 2, "directions"),
        ("ROUTE_EXTRACT", {"measure": "duration"}, 1, "extract_duration"),
        ("ROUTE_EXTRACT", {"measure": "distance"}, 1, "extract_distance"),
        ("EXTREME_SELECT", {"extreme": "max"}, 2, "select_max"),
        ("EXTREME_SELECT", {"extreme": "min"}, 2, "select_min"),
        ("AGGREGATE", {"aggregate": "difference"}, 2, "difference"),
        ("AGGREGATE", {"aggregate": "proportion"}, 2, "calculate_proportion"),
        ("AGGREGATE", {"scope": "groups"}, 2, "aggregate_route_groups"),
        ("AGGREGATE", {}, 2, "sum_amounts"),
        ("SCHEDULE", {"measure": "start"}, 1, "calculate_start_time"),
        ("SCHEDULE", {}, 1, "calculate_finish_time"),
    ],
)
def test_precedence_is_explicit_and_stated(
    transform: str, factors: dict, arity: int, expected: str
) -> None:
    operator, _rule = resolve_operator(
        transform, factors, input_types=["object"] * arity, facts=None, available=ALL
    )

    assert operator == expected


def test_a_stated_radius_and_a_stated_sector_choose_the_filter() -> None:
    """The literal comes from the analysis, so the *choice* it drives is deterministic too."""

    from src.agent.spatial import GroundingFacts

    radius = GroundingFacts(radius_m=600)
    sector = GroundingFacts(direction="북동쪽")
    plain = GroundingFacts()

    assert resolve_operator("FILTER", {}, input_types=["object"], facts=radius,
                            available=ALL)[0] == "within_radius"
    assert resolve_operator("FILTER", {}, input_types=["object"], facts=sector,
                            available=ALL)[0] == "filter_by_direction"
    assert resolve_operator("FILTER", {}, input_types=["object"], facts=plain,
                            available=ALL)[0] == "filter_places"


def test_an_operator_the_registry_cannot_run_is_not_chosen() -> None:
    without = ALL - {"travel_time"}
    operator, rule = resolve_operator(
        "ROUTE_MEASURE", {"measure": "duration"}, input_types=["object", "object"],
        facts=None, available=without
    )

    assert operator == "directions"
    assert rule == "fallback_first_runnable"


def test_a_transformation_no_operator_can_perform_is_a_graph_failure() -> None:
    with pytest.raises(ValueError, match="No executable operator"):
        resolve_operator("ROUTE_OPTIMIZE", {}, input_types=[], facts=None,
                         available=frozenset({"identity_measure"}))


def test_an_unknown_transformation_says_what_the_vocabulary_is() -> None:
    with pytest.raises(ValueError, match="Unknown semantic transformation"):
        resolve_operator("TELEPORT", {}, input_types=[], facts=None, available=ALL)


def test_a_planner_that_names_an_operator_is_carried_and_counted() -> None:
    """Losing a graph that would have executed to a vocabulary preference is the worse trade."""

    built = _build(
        [
            {"id": "a", "operator": "batch_geocode", "arguments": {"place_names": ["가"]}},
            {"id": "m", "transform": "MEASURE", "inputs": ["a"], "role": "measure"},
        ]
    )

    assert built.concrete_nodes == ("a",)
    assert [row["rule"] for row in built.decisions][0] == "planner_named_the_operator"
    assert built.graph[1]["operator"] == "identity_measure"


def test_a_node_that_names_neither_is_refused() -> None:
    with pytest.raises(ValueError, match="neither a transformation nor an operator"):
        _build([{"id": "a", "inputs": []}])


def test_the_vocabulary_and_the_prompt_cannot_drift_apart() -> None:
    from src.agent.semantics import transform_catalogue

    catalogue = transform_catalogue()
    for name in TRANSFORMS:
        assert name in catalogue


def test_a_semantic_graph_is_told_from_a_concrete_one() -> None:
    assert is_semantic_graph([{"id": "a", "transform": "MEASURE"}])
    assert not is_semantic_graph([{"id": "a", "operator": "identity_measure"}])
    assert not is_semantic_graph([])
