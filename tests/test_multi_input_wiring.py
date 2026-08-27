"""Every input a semantic node declares has to reach the operator that runs it.

This is the shape of every wiring defect this module has shipped, and none of them failed
loudly. `distance_matrix` over four resolved stops read `inputs[0]` and built a 1x1 grid.
`tsp_tw` took its node list from whichever input was object-typed and got one place beside a
six-place matrix. A via-route read the first two inputs and drove to the waypoint. Two
`DISTANCE_MEASURE` nodes over one resolved list factorized identically, so their difference was
zero. Each returned a confident number computed over less evidence than the graph had gathered.

So the invariant is checked rather than remembered, and these are the shapes it covers.
"""

from __future__ import annotations

import pytest

from src.agent.geoflow import OPERATOR_CONTRACTS, SKELETONS
from src.agent.semantics import (
    _PARTIAL_CONSUMERS,
    consumed_inputs,
    factorize_semantic_graph,
    unconsumed_inputs,
)
from src.agent.spatial import GroundingFacts

ALL = frozenset(OPERATOR_CONTRACTS)


def _places(count: int, prefix: str = "p"):
    return [{"id": f"{prefix}{index}", "text": f"장소{index}"} for index in range(count)]


def _build(graph, *, concepts=(), options=("가", "나", "다", "라"), facts=None, available=ALL):
    return factorize_semantic_graph(
        graph, concepts=list(concepts), options=list(options), facts=facts, available=available
    )


def _resolve_each(concepts):
    return [
        {"id": f"r{index}", "transform": "RESOLVE_PLACES", "inputs": [],
         "concept_ids": [concept["id"]], "role": "extent"}
        for index, concept in enumerate(concepts)
    ]


# ---------------------------------------------------------------------------------------------
# The invariant itself
# ---------------------------------------------------------------------------------------------


def test_consumed_inputs_finds_a_reference_however_deeply_it_is_nested() -> None:
    arguments = {
        "origins": ["$a.0.place", {"origin": "$b"}],
        "nodes": "$c.ranked",
        "index": 2,
        "key": "distance_m",
    }

    assert consumed_inputs(arguments) == {"a", "b", "c"}


def test_an_input_the_wiring_never_reads_is_reported_by_name() -> None:
    assert unconsumed_inputs(["a", "b", "c"], {"place_a": "$a", "place_b": "$c"}) == ["b"]


#: One test per exemption, named here so the list cannot grow without one. An exemption is a
#: claim that the input reaches the operator by another route, and a claim needs a test.
_EXEMPTION_TESTS = {
    "tsp_tw": "test_a_tour_reaches_its_stops_through_the_matrix_they_were_priced_in",
    "select_legs": "test_a_leg_selection_reaches_its_stops_through_the_matrix",
    "batch_geocode": "test_a_geocode_resolves_concepts_rather_than_upstream_output",
}


def test_every_operator_that_reads_fewer_inputs_than_it_depends_on_says_why() -> None:
    """The exemption list is small, each entry is a relation, and each one is tested.

    An exemption says the input is consumed indirectly. That is a claim about the graph, not a
    way to quiet the check, so every entry names the test that demonstrates the indirect route
    and this test fails if one is added without it.
    """

    assert set(_PARTIAL_CONSUMERS) <= set(OPERATOR_CONTRACTS)
    assert all(reason.strip() for reason in _PARTIAL_CONSUMERS.values())
    assert set(_PARTIAL_CONSUMERS) == set(_EXEMPTION_TESTS)
    defined = set(globals())
    assert not [name for name in _EXEMPTION_TESTS.values() if name not in defined]


def test_a_tour_reaches_its_stops_through_the_matrix_they_were_priced_in() -> None:
    """`tsp_tw` indexes into its cost matrix, so its nodes are that matrix's own stop list."""

    concepts = _places(3)
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "legs", "transform": "ROUTE_MATRIX",
             "inputs": [f"r{index}" for index in range(3)]},
            {"id": "tour", "transform": "ROUTE_OPTIMIZE",
             "inputs": [f"r{index}" for index in range(3)] + ["legs"]},
        ],
        concepts=concepts,
    )

    stops = ["$r0.0.place", "$r1.0.place", "$r2.0.place"]
    tour = built.graph[4]
    assert tour["operator"] == "tsp_tw"
    assert set(tour["depends_on"]) == {"r0", "r1", "r2", "legs"}
    # Not read as arguments of their own, but every one of them is in the list the tour indexes.
    assert tour["arguments"]["nodes"] == stops
    assert consumed_inputs(tour["arguments"]) == {"r0", "r1", "r2", "legs"}


def test_a_leg_selection_reaches_its_stops_through_the_matrix() -> None:
    concepts = _places(3)
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "legs", "transform": "ROUTE_MATRIX",
             "inputs": [f"r{index}" for index in range(3)]},
            {"id": "path", "transform": "SELECT_LEGS",
             "inputs": [f"r{index}" for index in range(3)] + ["legs"]},
        ],
        concepts=concepts,
    )

    selection = built.graph[4]
    assert selection["operator"] == "select_legs"
    assert selection["arguments"] == {"routes": "$legs"}
    # The stops are the matrix's own origins, so selecting from it selects among them.
    assert built.graph[3]["arguments"]["origins"] == [
        "$r0.0.place", "$r1.0.place", "$r2.0.place"
    ]


def test_a_radius_filter_needs_no_exemption_because_it_reads_the_input_it_declares() -> None:
    """It was on the exemption list and never needed to be: it reads its candidates."""

    facts = GroundingFacts(anchor="기준점", target_type="약국", radius_m=400)
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["a"], "role": "extent"},
            {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
            {"id": "inside", "transform": "FILTER", "inputs": ["found"]},
        ],
        concepts=[{"id": "a", "text": "기준점"}],
        facts=facts,
    )

    retrieval, filtering = built.graph[1], built.graph[2]
    assert filtering["operator"] == "within_radius"
    # The one input it declares is read; the centre comes from the retrieval's own arguments,
    # which is the anchor that retrieval already searched around.
    assert filtering["arguments"]["candidates"] == "$found"
    assert filtering["arguments"]["center"] == retrieval["arguments"]["center"]


def test_a_geocode_resolves_concepts_rather_than_upstream_output() -> None:
    """The names come from the concept graph; an upstream node is context, not content."""

    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["a"], "role": "extent"},
            {"id": "more", "transform": "RESOLVE_PLACES", "inputs": ["anchor"],
             "concept_ids": ["b"], "role": "support"},
        ],
        concepts=[{"id": "a", "text": "기준점"}, {"id": "b", "text": "다른곳"}],
    )

    assert built.graph[1]["operator"] == "batch_geocode"
    assert built.graph[1]["arguments"] == {"place_names": ["다른곳"]}
    assert built.graph[1]["depends_on"] == ["anchor"]


def test_a_retrieval_needs_no_exemption_because_it_reads_its_centre() -> None:
    """`place_search` was on the exemption list; given any input it resolves to `nearby_places`,
    which reads that input as the centre. The exemption was unreachable and is gone."""

    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["a"], "role": "extent"},
            {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
        ],
        concepts=[{"id": "a", "text": "기준점"}],
    )

    assert built.graph[1]["operator"] == "nearby_places"
    assert built.graph[1]["arguments"] == {"center": "$anchor.0.place"}


@pytest.mark.parametrize("key", sorted(SKELETONS))
def test_no_skeleton_gathers_evidence_it_then_ignores(key: str) -> None:
    facts = GroundingFacts(
        anchor="기준점",
        target_type="약국",
        radius_m=600,
        listed_places=("가게1", "가게2", "가게3"),
    )
    built = _build(SKELETONS[key], facts=facts)

    for node in built.graph:
        if node["operator"] in _PARTIAL_CONSUMERS:
            continue
        assert not unconsumed_inputs(node["depends_on"], node["arguments"]), (
            f"{key}/{node['id']} declares {node['depends_on']} and reads "
            f"{sorted(consumed_inputs(node['arguments']))}"
        )


# ---------------------------------------------------------------------------------------------
# The shapes the instructions name
# ---------------------------------------------------------------------------------------------


def test_a_pairwise_distance_reads_both_of_its_places() -> None:
    concepts = _places(2)
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "span", "transform": "DISTANCE_MEASURE", "inputs": ["r0", "r1"]},
        ],
        concepts=concepts,
    )

    assert built.graph[2]["operator"] == "haversine_distance"
    assert built.graph[2]["arguments"] == {"place_a": "$r0.0.place", "place_b": "$r1.0.place"}


def test_two_measures_over_one_resolved_list_measure_the_pairs_they_name() -> None:
    """Both halves of a difference came out identical, so the difference was zero."""

    concepts = [
        {"id": "anchor", "text": "종각역 1호선"},
        {"id": "target1", "text": "서울백제어린이박물관"},
        {"id": "target2", "text": "광진교8번가"},
    ]
    built = _build(
        [
            {"id": "R1", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor", "target1", "target2"], "role": "extent"},
            {"id": "D1", "transform": "DISTANCE_MEASURE", "inputs": ["R1"],
             "concept_ids": ["anchor", "target1"]},
            {"id": "D2", "transform": "DISTANCE_MEASURE", "inputs": ["R1"],
             "concept_ids": ["anchor", "target2"]},
            {"id": "gap", "transform": "AGGREGATE", "inputs": ["D1", "D2"],
             "factors": {"aggregate": "difference"}},
        ],
        concepts=concepts,
    )

    first, second = built.graph[1]["arguments"], built.graph[2]["arguments"]
    assert first == {"place_a": "$R1.0.place", "place_b": "$R1.1.place"}
    assert second == {"place_a": "$R1.0.place", "place_b": "$R1.2.place"}
    assert first != second
    assert built.graph[3]["arguments"] == {"minuend": "$D1", "subtrahend": "$D2"}


def test_a_pair_named_in_inputs_rather_than_in_concept_ids_still_resolves() -> None:
    """The planner writes concept ids where node ids belong; they were silently dropped."""

    concepts = [
        {"id": "anchor", "text": "가"},
        {"id": "target1", "text": "나"},
        {"id": "target2", "text": "다"},
    ]
    built = _build(
        [
            {"id": "resolve_all", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["anchor", "target1", "target2"], "role": "extent"},
            {"id": "dist1", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "target1"],
             "concept_ids": ["distance_1"]},
        ],
        concepts=concepts,
    )

    assert built.graph[1]["arguments"] == {
        "place_a": "$resolve_all.0.place",
        "place_b": "$resolve_all.1.place",
    }
    assert built.graph[1]["depends_on"] == ["resolve_all"]


def test_a_pair_the_graph_wired_as_two_nodes_is_not_overridden_by_its_concept_ids() -> None:
    concepts = _places(2)
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "span", "transform": "DISTANCE_MEASURE", "inputs": ["r0", "r1"],
             "concept_ids": ["p0", "p1"]},
        ],
        concepts=concepts,
    )

    assert built.graph[2]["arguments"] == {"place_a": "$r0.0.place", "place_b": "$r1.0.place"}


def test_filtering_many_candidates_keeps_every_candidate_it_was_given() -> None:
    facts = GroundingFacts(anchor="기준점", target_type="약국", radius_m=400)
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["a"], "role": "extent"},
            {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
            {"id": "measured", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "found"]},
            {"id": "inside", "transform": "FILTER", "inputs": ["measured"]},
        ],
        concepts=[{"id": "a", "text": "기준점"}],
        facts=facts,
    )

    assert built.graph[2]["operator"] == "nearest"
    assert built.graph[3]["operator"] == "filter_by_distance"
    assert built.graph[3]["arguments"] == {"items": "$measured.ranked"}


def test_a_radius_filter_over_a_retrieval_measures_from_the_centre_that_retrieval_used() -> None:
    """Given only the candidate set, the filter was handed the candidate set as its centre."""

    facts = GroundingFacts(anchor="기준점", target_type="약국", radius_m=400)
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["a"], "role": "extent"},
            {"id": "found", "transform": "PLACE_SEARCH", "inputs": ["anchor"]},
            {"id": "inside", "transform": "FILTER", "inputs": ["found"]},
        ],
        concepts=[{"id": "a", "text": "기준점"}],
        facts=facts,
    )

    assert built.graph[2]["operator"] == "within_radius"
    assert built.graph[2]["arguments"] == {
        "center": "$anchor.0.place",
        "candidates": "$found",
    }


def test_a_radius_filter_with_nothing_to_measure_from_is_refused_rather_than_answered() -> None:
    facts = GroundingFacts(radius_m=400)

    with pytest.raises(ValueError, match="no place to measure from"):
        _build(
            [
                {"id": "listed", "transform": "RESOLVE_PLACES", "inputs": [],
                 "factors": {"scope": "options"}, "role": "extent"},
                {"id": "inside", "transform": "FILTER", "inputs": ["listed"]},
            ],
            facts=facts,
        )


def test_counting_a_filtered_set_counts_the_set_and_not_its_coordinates() -> None:
    facts = GroundingFacts(anchor="기준점", radius_m=300, listed_places=("가", "나", "다", "라"))
    built = _build(
        [
            {"id": "anchor", "transform": "RESOLVE_PLACES", "inputs": [],
             "concept_ids": ["a"], "role": "extent"},
            {"id": "listed", "transform": "RESOLVE_PLACES", "inputs": [],
             "factors": {"scope": "listed"}, "role": "extent"},
            {"id": "measured", "transform": "DISTANCE_MEASURE", "inputs": ["anchor", "listed"]},
            {"id": "inside", "transform": "FILTER", "inputs": ["measured"]},
            {"id": "count", "transform": "AGGREGATE", "inputs": ["inside"],
             "factors": {"aggregate": "count"}},
        ],
        concepts=[{"id": "a", "text": "기준점"}],
        facts=facts,
    )

    assert built.graph[1]["arguments"] == {"place_names": ["가", "나", "다", "라"]}
    assert built.graph[4]["operator"] == "count_items"
    assert built.graph[4]["arguments"] == {"items": "$inside"}


def test_a_matrix_and_a_tour_agree_on_which_stops_they_are_over() -> None:
    concepts = _places(4)
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "legs", "transform": "ROUTE_MATRIX",
             "inputs": [f"r{index}" for index in range(4)]},
            {"id": "tour", "transform": "ROUTE_OPTIMIZE",
             "inputs": [f"r{index}" for index in range(4)] + ["legs"]},
        ],
        concepts=concepts,
    )

    stops = ["$r0.0.place", "$r1.0.place", "$r2.0.place", "$r3.0.place"]
    assert built.graph[4]["arguments"] == {"origins": stops, "destinations": stops}
    assert built.graph[5]["arguments"] == {"nodes": stops, "distance_matrix": "$legs"}


def test_a_route_reads_its_origin_its_destination_and_every_waypoint() -> None:
    concepts = _places(3)
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "route", "transform": "ROUTE_MEASURE", "inputs": ["r0", "r1", "r2"],
             "via": ["r1"]},
        ],
        concepts=concepts,
    )

    assert built.graph[3]["arguments"] == {
        "origin": "$r0.0.place",
        "destination": "$r2.0.place",
        "waypoints": ["$r1.0.place"],
    }
    assert consumed_inputs(built.graph[3]["arguments"]) == {"r0", "r1", "r2"}


def test_a_selection_across_several_measures_reads_all_of_them() -> None:
    """`select_max({"items": "$dist1"})` is a maximum over one item, and it reported it."""

    concepts = _places(4, prefix="t")
    built = _build(
        [
            *_resolve_each(concepts),
            {"id": "d1", "transform": "DISTANCE_MEASURE", "inputs": ["r0", "r1"]},
            {"id": "d2", "transform": "DISTANCE_MEASURE", "inputs": ["r0", "r2"]},
            {"id": "d3", "transform": "DISTANCE_MEASURE", "inputs": ["r0", "r3"]},
            {"id": "farthest", "transform": "EXTREME_SELECT", "inputs": ["d1", "d2", "d3"],
             "factors": {"extreme": "max"}},
        ],
        concepts=concepts,
    )

    assert built.graph[7]["arguments"] == {
        "items": ["$d1", "$d2", "$d3"],
        "key": "distance_m",
    }


def test_an_operator_gathering_several_branches_flattens_them() -> None:
    from src.tools.spatial import SpatialOperatorRegistry as ops

    branches = [[{"name": "가", "distance_m": 900}], [{"name": "나", "distance_m": 200}]]

    assert ops.select_min(branches, "distance_m")["name"] == "나"
    assert [item["name"] for item in ops.sort_by(branches, "distance_m")] == ["나", "가"]


def test_a_count_reads_a_count_a_node_already_reported() -> None:
    """Wrapping a tour in a collection of one and calling that a count answers "one"."""

    from src.tools.spatial import SpatialOperatorRegistry as ops

    assert ops.count_items({"visited_count": 3, "order": [0, 2, 1]})["count"] == 3
    assert ops.count_items([{"name": "가"}, {"name": "나"}])["count"] == 2


def test_a_stated_subtype_that_matches_nothing_is_evidence_and_not_a_lexicon_gap() -> None:
    """The filter used to pass every candidate through, and the ranking answered from those."""

    from src.tools.spatial import SpatialOperatorRegistry as ops

    restaurants = [
        {"name": "태백식당", "category": "음식점 > 한식", "latitude": 37.5, "longitude": 127.0},
        {"name": "김밥천국", "category": "음식점 > 분식", "latitude": 37.5, "longitude": 127.0},
    ]

    assert ops.filter_places(restaurants, required_types=["중식"]) == restaurants
    assert ops.filter_places(restaurants, required_types=["중식"], types_are_required=True) == []
    kept = ops.filter_places(restaurants, required_types=["분식"], types_are_required=True)
    assert [place["name"] for place in kept] == ["김밥천국"]


def test_a_subtype_over_candidates_with_no_category_at_all_is_still_a_gap() -> None:
    """Nothing to test the constraint against is a lexicon gap, and the filter is dropped."""

    from src.tools.spatial import SpatialOperatorRegistry as ops

    bare = [{"name": "가게", "latitude": 37.5, "longitude": 127.0}]

    assert ops.filter_places(bare, required_types=["중식"], types_are_required=True) == bare


def test_a_listed_candidate_count_measures_filters_and_counts_the_names_offered() -> None:
    """The shape end to end, from the facts the analysis read to the option it points at."""

    from src.agent.geoflow import SKELETONS
    from src.agent.spatial import _ground_graph_literals

    skeleton = [dict(node) for node in SKELETONS["listed_candidates_count"]]
    skeleton[0]["concept_ids"] = ["anchor"]
    facts = GroundingFacts(
        anchor="왕십리곱창거리",
        target_type="은행",
        radius_m=300,
        listed_places=("하나은행365", "우리은행 365코너", "우리은행 서울동부", "KB국민은행ATM"),
    )
    options = ["한 곳", "두 곳", "세 곳", "네 곳"]
    built = _build(
        skeleton,
        concepts=[{"id": "anchor", "text": "왕십리곱창거리", "concept_type": "location"}],
        options=options,
        facts=facts,
    )
    grounded = _ground_graph_literals(built.graph, "질문", options, facts)
    by_id = {node["id"]: node for node in grounded}

    assert by_id["listed"]["arguments"]["place_names"] == list(facts.listed_places)
    assert by_id["measured"]["operator"] == "nearest"
    assert by_id["inside"]["operator"] == "filter_by_distance"
    assert by_id["inside"]["arguments"]["max_distance_m"] == 300
    assert by_id["count"]["operator"] == "count_items"
    # And it ends at the options rather than at an identity measure, so the graph names an
    # answer instead of leaving the response stage to read "1" and pick "한 곳" itself.
    assert by_id["answer"]["operator"] == "match_count_options"


def test_a_count_matcher_is_chosen_only_when_the_options_count_something() -> None:
    """Visiting orders carry digits inside place names and distances carry units; neither counts."""

    from src.tools.spatial import options_state_counts

    assert options_state_counts(["한 곳", "두 곳", "세 곳", "네 곳"])
    assert options_state_counts(["한 곳", "주어진 지도 정보로는 알 수 없음", "세 곳", "네 곳"])
    assert not options_state_counts(["약 2.4km", "약 3.3km", "약 2.6km", "약 3.1km"])
    assert not options_state_counts(["외대앞역 1호선 → 롯데시네마", "A → B", "C → D", "E → F"])
    assert not options_state_counts(["진달래장국", "일일양꼬치", "맛있는떡", "일품채관"])


def test_a_count_matches_the_option_that_states_it_and_nothing_else() -> None:
    from src.tools.spatial import SpatialOperatorRegistry as ops

    options = ["한 곳", "두 곳", "세 곳", "네 곳"]

    assert ops.match_count_options({"count": 1}, options)["best_option"]["option_index"] == 0
    four = ops.match_count_options({"visited_count": 4}, options)
    assert four["best_option"]["option_index"] == 3
    # A count no option states is not "about" the nearest one: a count is exact or it is unmet.
    assert ops.match_count_options({"count": 9}, options)["best_option"] is None
