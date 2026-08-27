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

    assert [step["operator"] for step in rebuilt.graph] == [step["operator"] for step in original]


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
            {
                "id": "k",
                "transform": "ORDINAL_SELECT",
                "inputs": ["s"],
                "factors": {"ordinal": position},
            },
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
        {
            "id": "a",
            "transform": "RESOLVE_PLACES",
            "inputs": [],
            "concept_ids": ["one", "two"],
            "place_names": ["틀린 이름"],
        },
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

    assert (
        resolve_operator("FILTER", {}, input_types=["object"], facts=radius, available=ALL)[0]
        == "within_radius"
    )
    assert (
        resolve_operator("FILTER", {}, input_types=["object"], facts=sector, available=ALL)[0]
        == "filter_by_direction"
    )
    assert (
        resolve_operator("FILTER", {}, input_types=["object"], facts=plain, available=ALL)[0]
        == "filter_places"
    )


def test_an_operator_the_registry_cannot_run_is_not_chosen() -> None:
    without = ALL - {"travel_time"}
    operator, rule = resolve_operator(
        "ROUTE_MEASURE",
        {"measure": "duration"},
        input_types=["object", "object"],
        facts=None,
        available=without,
    )

    assert operator == "directions"
    assert rule == "fallback_first_runnable"


def test_a_transformation_no_operator_can_perform_is_a_graph_failure() -> None:
    with pytest.raises(ValueError, match="No executable operator"):
        resolve_operator(
            "ROUTE_OPTIMIZE",
            {},
            input_types=[],
            facts=None,
            available=frozenset({"identity_measure"}),
        )


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


# ---------------------------------------------------------------------------------------------
# What the first live run of this stage found.
#
# The vocabulary took immediately -- 200 planner nodes, none of them naming an operator -- and
# accuracy fell to 40.3% anyway, with 47 questions producing no valid graph at all. Every cause
# was in this module, and each is pinned below. The loop that found them replays the semantic
# graphs that run recorded, so none of it needed another benchmark pass.
# ---------------------------------------------------------------------------------------------


def test_a_geocode_always_gets_names_to_resolve() -> None:
    """122 of 642 recorded RESOLVE_PLACES nodes named a concept the analysis did not have.

    `Option 0`, `candidate_options`, a slug of the anchor -- usually meaning the candidate texts,
    sometimes because the Analysis stage had returned nothing but `question_context`. An empty
    `batch_geocode` geocodes nothing and the graph fails four nodes later, so the lookup falls
    through an explicit chain instead.
    """

    concepts = [{"id": "anchor", "text": "서울역", "concept_type": "location", "role": "extent"}]

    invented_options = _build(
        [
            {
                "id": "a",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["Option 0", "Option 1"],
            }
        ],
        concepts=concepts,
        options=["가게 A", "가게 B"],
    )
    assert invented_options.graph[0]["arguments"]["place_names"] == ["가게 A", "가게 B"]

    invented_anchor = _build(
        [
            {
                "id": "a",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["seoul_station_slug"],
            }
        ],
        concepts=concepts,
        options=["가게 A"],
    )
    assert invented_anchor.graph[0]["arguments"]["place_names"] == ["서울역"]

    nothing = _build(
        [{"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["x"]}],
        concepts=[],
        options=["가게 A", "가게 B"],
    )
    assert nothing.graph[0]["arguments"]["place_names"] == ["가게 A", "가게 B"]


def test_one_place_against_many_is_a_ranking_not_a_pair() -> None:
    """190 recorded graphs measured an anchor against a candidate *set*.

    Wired as a pair it becomes `haversine_distance(place_a=anchor, place_b=$candidates.0.place)`
    -- the first candidate, with the rest silently discarded, and every node downstream ranking
    a scalar.
    """

    concepts = [
        {"id": "anchor", "text": "서울역"},
        {"id": "c1", "text": "A"},
        {"id": "c2", "text": "B"},
    ]
    built = _build(
        [
            {
                "id": "anchor",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["anchor"],
            },
            {
                "id": "cands",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["c1", "c2"],
            },
            {"id": "d", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "cands"]},
        ],
        concepts=concepts,
    )

    assert built.graph[2]["operator"] == "nearest"
    assert built.graph[2]["arguments"] == {
        "anchor": "$anchor.0.place",
        "candidates": "$cands",
    }
    # Two single places is still a pair.
    pair = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["anchor"]},
            {"id": "b", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["c1"]},
            {"id": "d", "transform": "DISTANCE_MEASURE", "inputs": ["a", "b"]},
        ],
        concepts=concepts,
    )
    assert pair.graph[2]["operator"] == "haversine_distance"


def test_a_sort_over_an_existing_ordering_is_folded_away() -> None:
    """`nearest` already ranks. A second node to re-sort it asks for a field it does not carry."""

    built = _build(
        [
            {"id": "a", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["x"]},
            {"id": "c", "transform": "PLACE_SEARCH", "inputs": ["a"]},
            {"id": "d", "transform": "DISTANCE_MEASURE", "inputs": ["a", "c"]},
            {"id": "s", "transform": "SORT", "inputs": ["d"]},
            {"id": "k", "transform": "ORDINAL_SELECT", "inputs": ["s"], "factors": {"ordinal": 2}},
        ],
        concepts=[{"id": "x", "text": "서울역"}],
    )

    assert [step["id"] for step in built.graph] == ["a", "c", "d", "k"]
    # The ordinal still reads the ordering the fused node produced.
    assert built.graph[-1]["arguments"]["items"] == "$d.ranked"
    assert any(row["rule"] == "fused_into_existing_ordering" for row in built.decisions)


def test_a_tour_gets_the_cost_matrix_it_needs() -> None:
    """9 recorded graphs asked to order an itinerary without asking for the costs first."""

    built = _build(
        [
            {"id": "stops", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["a", "b"]},
            {"id": "tour", "transform": "ROUTE_OPTIMIZE", "inputs": ["stops"], "role": "measure"},
        ],
        concepts=[{"id": "a", "text": "가"}, {"id": "b", "text": "나"}],
    )

    assert [step["operator"] for step in built.graph] == [
        "batch_geocode",
        "distance_matrix",
        "tsp_tw",
    ]
    assert built.graph[2]["arguments"]["distance_matrix"] == "$tour_matrix"
    assert any(row["rule"] == "composed_for_route_optimize" for row in built.decisions)


def test_an_input_written_as_a_field_path_still_names_its_node() -> None:
    """A planner wrote `inputs: ["R1.start", "R1.end"]`. The node is the dependency."""

    built = _build(
        [
            {"id": "R1", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["a", "b"]},
            {"id": "M1", "transform": "ROUTE_MEASURE", "inputs": ["R1.start", "R1.end"]},
        ],
        concepts=[{"id": "a", "text": "가"}, {"id": "b", "text": "나"}],
    )

    assert built.graph[1]["depends_on"] == ["R1"]
    assert built.graph[1]["arguments"] == {
        "origin": "$R1.0.place",
        "destination": "$R1.1.place",
    }


def test_a_total_over_a_matrix_is_not_an_extract_from_a_route() -> None:
    """15 recorded graphs asked ROUTE_EXTRACT of a matrix, which returns a dict, not a route."""

    built = _build(
        [
            {"id": "p", "transform": "RESOLVE_PLACES", "inputs": [], "concept_ids": ["a", "b"]},
            {"id": "m", "transform": "ROUTE_MATRIX", "inputs": ["p"]},
            {
                "id": "e",
                "transform": "ROUTE_EXTRACT",
                "inputs": ["m"],
                "factors": {"measure": "distance"},
            },
        ],
        concepts=[{"id": "a", "text": "가"}, {"id": "b", "text": "나"}],
    )

    # A square matrix holds every pair; a trip drives the consecutive ones. The selection is
    # composed in as its own node so the grouping stays visible in the graph, and the total is
    # taken over what it selected rather than over the whole grid.
    assert built.graph[2]["operator"] == "select_legs"
    assert built.graph[2]["arguments"] == {"routes": "$m"}
    assert built.graph[3]["operator"] == "sum_route_metrics"
    assert built.graph[3]["arguments"] == {"routes": "$e_legs"}


# ---------------------------------------------------------------------------------------------
# Semantic skeletons carry the question-shape knowledge.
#
# Deleting the 163-line planner prompt deleted operator documentation *and* graph-shape guidance
# together, and only the first was redundant. The second cost 31 points: a
# "네 번째로 가까운 은행" question retrieved `Geocode-Batch-Compare` -- pattern "resolve the
# anchor and the candidates, then rank them" -- as its only guidance and copied it, ranking the
# four answer texts. `SKELETONS` is that knowledge as transformation structure.
# ---------------------------------------------------------------------------------------------


def test_nothing_the_planner_is_shown_names_an_operator() -> None:
    """0% concrete operator leakage, over the prompt, the patterns and the skeletons.

    `nearest` and `difference` are ordinary English words that happen to collide with operator
    names; they appear only in prose and in a factor value. The names that could only be an
    instruction to call a tool are the compound ones, and none of those appears at all.
    """

    import json
    import re

    from src.agent.geoflow import SKELETONS, TEMPLATES
    from src.agent.semantics import transform_catalogue
    from src.agent.spatial import GRAPH_PROMPT

    shown = (
        GRAPH_PROMPT.replace("{transform_catalogue}", transform_catalogue())
        + json.dumps([t["pattern"] for t in TEMPLATES.values()], ensure_ascii=False)
        + json.dumps(SKELETONS, ensure_ascii=False)
    )

    assert not [name for name in OPERATOR_CONTRACTS if "_" in name and name in shown]
    assert not re.search(r"\b(?:MT1|CS2|PS3|SC4|AC5|PK6|OL7|SW8|BK9|FD6|CE7|HP8|PM9)\b", shown)
    assert not [node["id"] for sk in SKELETONS.values() for node in sk if "operator" in node]
    # Every skeleton node names a transformation from the vocabulary and nothing else.
    for skeleton in SKELETONS.values():
        for node in skeleton:
            assert node["transform"] in TRANSFORMS


def test_the_ordinal_skeleton_searches_rather_than_ranking_the_answers() -> None:
    """The shape the 31 points were lost to, stated as a skeleton.

    Only the anchor is resolved. The candidate texts are answers: `nearby_kth_nearest` draws its
    gold as rank k of the whole neighbourhood and its decoys from ranks 1 to 6, so a graph that
    resolves the four options and ranks them is answering "which of these four is closest".
    """

    from src.agent.geoflow import SKELETONS

    shape = [node["transform"] for node in SKELETONS["search_rank_ordinal"]]
    assert shape == [
        "RESOLVE_PLACES",
        "PLACE_SEARCH",
        "DISTANCE_MEASURE",
        "ORDINAL_SELECT",
        "MATCH_OPTIONS",
    ]
    # One RESOLVE_PLACES, and it is the anchor.
    assert sum(t == "RESOLVE_PLACES" for t in shape) == 1
    # k is a factor on the selection, which is the whole difference from the superlative.
    ordinal = next(
        node for node in SKELETONS["search_rank_ordinal"] if node["transform"] == "ORDINAL_SELECT"
    )
    assert "ordinal" in ordinal["factors"]


def test_a_stated_radius_reaches_the_filter_that_applies_it() -> None:
    """`within_radius` never received its radius, because no planner had ever written one.

    Grounding bound `radius_m` onto the retrieval and onto option recovery only. The semantic
    layer composes FILTER over a stated radius, so a radius question now produces a filter node
    on every one -- and it was being validated as missing its only argument.
    `nearby_within_radius_count` read 8.3%.
    """

    from src.agent.geoflow import SKELETONS
    from src.agent.spatial import GroundingFacts, _ground_graph_literals

    facts = GroundingFacts(anchor="기준점", target_type="약국", radius_m=600)
    built = _build(SKELETONS["radius"], options=["가", "나"], facts=facts)
    grounded = _ground_graph_literals(built.graph, "질문", ["가", "나"], facts)

    # The shape now measures before it narrows, so the operator that applies the radius is the
    # one that reads a measurement rather than the one that recomputes it. The literal is bound
    # the same way either way, which is the property under test.
    filtering = next(
        step for step in grounded if step["operator"] in {"within_radius", "filter_by_distance"}
    )
    stated = filtering["arguments"].get("radius_m") or filtering["arguments"].get("max_distance_m")
    assert stated == 600


# ---------------------------------------------------------------------------------------------
# Concept Analysis is load-bearing now, and it was not before.
#
# Measured over the same benchmark rows, the Analysis stage is exactly as weak at `af51e93` as
# here: 24% vs 19% of questions get no usable concepts, 43% vs 44% name a kind of place and get
# `target_type: null`. That is not a regression -- it is a long-standing weakness the old
# architecture did not depend on, because the planner copied place names out of the question
# itself. The semantic architecture routes place identity through the concept graph, so the same
# analysis that supported 82.1% supports 50.5%.
# ---------------------------------------------------------------------------------------------


def test_the_fallback_concept_graph_is_built_from_what_the_question_states() -> None:
    """It used to be the whole question, as one concept, typed as a place."""

    from src.agent.geoflow import normalize_analysis
    from src.agent.spatial import extract_facts

    question = (
        "지금 세라존에 있습니다. 계단에서 발을 헛디뎌 발목이 심하게 부었습니다. "
        "여기서 두 번째로 가까운 정형외과는 다음 중 어디인가요?"
    )
    empty = {"intent": "nearby", "concepts": [], "measure": "ranking"}
    analysis = normalize_analysis(empty, question, facts=extract_facts(empty, question))

    texts = [concept["text"] for concept in analysis["concepts"]]
    assert "세라존" in texts
    assert "정형외과" in texts
    assert question not in texts
    # And the kind reaches the field template retrieval reads.
    assert analysis["target_type"] == "정형외과"


def test_a_synthetic_placeholder_is_never_geocoded() -> None:
    """The role completion inserts one so the graph has an extent. It names no place.

    Naming it in `concept_ids` sent `batch_geocode` after a sentence -- which is what happened on
    every question whose analysis produced no extent at all.
    """

    concepts = [
        {
            "id": "question_context",
            "text": "지금 어딘가에 있습니다. 무엇이 가장 가까운가요?",
            "concept_type": "object",
            "role": "extent",
            "attributes": {"synthetic": True},
        },
        {"id": "real", "text": "서울역", "concept_type": "location", "role": "extent"},
    ]
    built = _build(
        [
            {
                "id": "a",
                "transform": "RESOLVE_PLACES",
                "inputs": [],
                "concept_ids": ["question_context", "real"],
            }
        ],
        concepts=concepts,
        options=["가", "나"],
    )

    assert built.graph[0]["arguments"]["place_names"] == ["서울역"]


def test_a_question_stating_nothing_resolvable_falls_back_to_a_measure_alone() -> None:
    """Honest rather than invented: it says what is asked for and nothing about where."""

    from src.agent.geoflow import normalize_analysis
    from src.agent.spatial import GroundingFacts

    analysis = normalize_analysis({"concepts": []}, "무엇이 가장 좋은가요?", facts=GroundingFacts())
    places = [c for c in analysis["concepts"] if c["concept_type"] in {"location", "object"}]

    assert all((c.get("attributes") or {}).get("synthetic") for c in places)
    assert any(c["role"] == "measure" for c in analysis["concepts"])
