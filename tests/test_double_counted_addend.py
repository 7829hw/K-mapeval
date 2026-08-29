"""A total must not add a measurement to the thing it was measured from.

Measured on `dataset/seoul_kmapeval_v9_mcq_300.jsonl`: all seven `trip_total_distance` questions
that ended with the grounded answer matching no option overshot the gold by *exactly their own
first leg*, and the plan says why -- `sum_amounts(amounts=["$t4", "$t7", "$t10", "$t3"])` where
`$t4` is `extract_distance(route="$t3")`. The raw route and the distance taken off it are one
measurement written twice, so a three-leg drive was totalled from four addends. `sum_amounts`
cannot see it: by the time it runs both are dicts carrying `distance_m`. The executor can.
"""

from __future__ import annotations

from src.agent.spatial import _drop_subsumed_addends, _step_sources

STEPS = [
    {"id": "t1", "operator": "batch_geocode", "arguments": {"place_names": ["풀문"]}},
    {"id": "t2", "operator": "batch_geocode", "arguments": {"place_names": ["인사동거리"]}},
    {
        "id": "t3",
        "operator": "directions",
        "arguments": {"origin": "$t1.0.place", "destination": "$t2.0.place"},
    },
    {"id": "t4", "operator": "extract_distance", "arguments": {"route": "$t3"}},
    {
        "id": "t6",
        "operator": "directions",
        "arguments": {"origin": "$t2.0.place", "destination": "$t1.0.place"},
    },
    {"id": "t7", "operator": "extract_distance", "arguments": {"route": "$t6"}},
]


def test_a_leg_is_not_added_to_the_route_it_came_off() -> None:
    sources = _step_sources(STEPS)
    trimmed = _drop_subsumed_addends({"amounts": ["$t4", "$t7", "$t3"]}, sources)
    assert trimmed["amounts"] == ["$t4", "$t7"]


def test_the_same_step_is_not_added_twice() -> None:
    sources = _step_sources(STEPS)
    trimmed = _drop_subsumed_addends({"amounts": ["$t4", "$t4", "$t7"]}, sources)
    assert trimmed["amounts"] == ["$t4", "$t7"]


def test_independent_legs_all_survive() -> None:
    """The rule must only ever drop a source of another addend, never a measurement of its own."""

    sources = _step_sources(STEPS)
    assert _drop_subsumed_addends({"amounts": ["$t4", "$t7"]}, sources)["amounts"] == [
        "$t4",
        "$t7",
    ]
    # Two raw routes with nothing extracted off either is a legitimate total of two legs.
    assert _drop_subsumed_addends({"amounts": ["$t3", "$t6"]}, sources)["amounts"] == [
        "$t3",
        "$t6",
    ]


def test_a_literal_beside_a_reference_is_left_alone() -> None:
    sources = _step_sources(STEPS)
    trimmed = _drop_subsumed_addends({"amounts": ["$t4", 1200, "$t7"]}, sources)
    assert trimmed["amounts"] == ["$t4", 1200, "$t7"]


def test_an_indirect_source_is_still_a_source() -> None:
    """`$t3` is two steps back from a distance taken off a route taken off it."""

    steps = [
        *STEPS,
        {"id": "t5", "operator": "identity_measure", "arguments": {"value": "$t4"}},
    ]
    sources = _step_sources(steps)
    trimmed = _drop_subsumed_addends({"amounts": ["$t5", "$t3", "$t7"]}, sources)
    assert trimmed["amounts"] == ["$t5", "$t7"]
