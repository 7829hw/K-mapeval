"""Plans this port refused for a spelling, replayed from the runs that lost them.

Seven of the ten `agent_reasoning_failure` rows across the two 300-question draws were
`GeoFlow node ... is missing arguments: ...`, and `kmapeval_223` failed that way on all three
passes of the v7a run — a repeat across passes is a defect, not a draw. The graphs below are the
planner's own, copied out of `logs/`, so a change that makes them validate is measured against
what the planner actually wrote rather than what it might write.
"""

from __future__ import annotations

import pytest

from src.agent.geoflow import normalize_and_validate_graph
from src.tools.spatial import SpatialOperatorRegistry

MAX_STEPS = 15


def _steps(graph: list[dict]) -> dict[str, dict]:
    steps, _ = normalize_and_validate_graph({"graph": graph}, max_steps=MAX_STEPS)
    return {step["id"]: step for step in steps}


def test_a_lone_dependency_fills_a_lone_missing_argument() -> None:
    """`kmapeval_223`, pass 3: every input implied by `depends_on`, `arguments` left empty.

    The planner said where each value comes from and each operator has one slot to put it in, so
    there is one binding consistent with the plan. Refused, it cost the question — and the repair
    round filled `extract_distance` and left `sum_amounts` empty, which is why the repair is not
    the answer here.
    """

    steps = _steps(
        [
            {
                "id": "locations",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["더스테이모텔", "양재천 벚꽃길"]},
                "depends_on": [],
                "role": "extent",
            },
            {
                "id": "route_legs",
                "operator": "distance_matrix",
                "arguments": {
                    "pairs": [
                        {
                            "origin": "$locations.0.place",
                            "destination": "$locations.1.place",
                        }
                    ]
                },
                "depends_on": ["locations"],
                "role": "support",
            },
            {
                "id": "leg_distances",
                "operator": "extract_distance",
                "arguments": {},
                "depends_on": ["route_legs"],
                "role": "support",
            },
            {
                "id": "total_distance",
                "operator": "sum_amounts",
                "arguments": {},
                "depends_on": ["leg_distances"],
                "role": "measure",
            },
        ]
    )
    assert steps["leg_distances"]["arguments"] == {"route": "$route_legs"}
    assert steps["total_distance"]["arguments"] == {"amounts": "$leg_distances"}


def test_two_missing_arguments_are_still_a_guess() -> None:
    """The narrowness is the point: one slot and one source, or nothing is filled."""

    with pytest.raises(ValueError, match="missing arguments: anchor, candidates"):
        _steps(
            [
                {
                    "id": "places",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["가", "나"]},
                    "role": "extent",
                },
                {
                    "id": "ranking",
                    "operator": "nearest",
                    "arguments": {},
                    "depends_on": ["places"],
                    "role": "measure",
                },
            ]
        )


def test_two_dependencies_are_still_a_guess() -> None:
    with pytest.raises(ValueError, match="missing arguments: route"):
        _steps(
            [
                {
                    "id": "a",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["가"]},
                    "role": "extent",
                },
                {
                    "id": "b",
                    "operator": "batch_geocode",
                    "arguments": {"place_names": ["나"]},
                    "role": "extent",
                },
                {
                    "id": "legs",
                    "operator": "extract_distance",
                    "arguments": {},
                    "depends_on": ["a", "b"],
                    "role": "measure",
                },
            ]
        )


def test_nearest_accepts_the_name_nearby_places_uses_for_the_same_point() -> None:
    """`kmapeval_015` and `kmapeval_026`: `nearby_places(center=...)` then `nearest(center=...)`."""

    steps = _steps(
        [
            {
                "id": "anchor",
                "operator": "batch_geocode",
                "arguments": {"place_names": ["쌍문역사산책길"]},
                "role": "extent",
            },
            {
                "id": "target_pois",
                "operator": "nearby_places",
                "arguments": {"center": "$anchor.0.location", "query": "내과", "limit": 15},
                "depends_on": ["anchor"],
                "role": "support",
            },
            {
                "id": "ranking",
                "operator": "nearest",
                "arguments": {"center": "$anchor.0.location", "candidates": "$target_pois"},
                "depends_on": ["target_pois"],
                "role": "measure",
            },
        ]
    )
    assert "center" in steps["ranking"]["arguments"]


def test_the_registry_takes_center_as_the_anchor_it_ranks_from() -> None:
    """Validating it is not enough: the operator has to run on the spelling that was let through."""

    registry = SpatialOperatorRegistry()
    anchor = {"place_id": "a", "name": "기준", "latitude": 37.5, "longitude": 127.0}
    far = {"place_id": "b", "name": "먼 곳", "latitude": 37.55, "longitude": 127.0}
    near = {"place_id": "c", "name": "가까운 곳", "latitude": 37.501, "longitude": 127.0}
    result = registry.invoke("nearest", {"center": anchor, "candidates": [far, near]})
    assert result["nearest"]["place_id"] == "c"


def test_extract_distance_measures_every_leg_when_handed_the_list() -> None:
    """`kmapeval_223`, pass 2: `extract_distance(routes="$segments.routes")` over three legs."""

    registry = SpatialOperatorRegistry()
    routes = [{"distance_m": 7508, "duration_s": 900}, {"distance_m": 10058, "duration_s": 1200}]
    assert registry.invoke("extract_distance", {"routes": routes}) == [
        {"distance_m": 7508.0},
        {"distance_m": 10058.0},
    ]
    # And the sum the next node takes still adds up.
    legs = registry.invoke("extract_distance", {"routes": routes})
    total = registry.invoke("sum_amounts", {"amounts": legs})
    assert total["distance_m"] == 17566.0


def test_a_reference_that_never_resolved_still_fails_loudly() -> None:
    """The leniency is about shape, never about inventing a measurement that was not made."""

    registry = SpatialOperatorRegistry()
    with pytest.raises(ValueError, match="measures nothing"):
        registry.invoke("extract_distance", {"route": "$legs_matrix.routes[0]"})
