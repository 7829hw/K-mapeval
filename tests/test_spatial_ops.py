from __future__ import annotations

import pytest

from src.tools.spatial import SpatialOperatorRegistry


def test_haversine_and_bearing() -> None:
    ops = SpatialOperatorRegistry()
    seoul_station = {"latitude": 37.5547, "longitude": 126.9707}
    tower = {"latitude": 37.5512, "longitude": 126.9882}
    distance = ops.haversine_distance(seoul_station, tower)
    direction = ops.bearing_to_direction(seoul_station, tower)
    assert 1500 < distance["distance_m"] < 1700
    assert direction["direction"] in {"E", "SE"}
    assert direction["cardinal_direction"] == "E"
    assert direction["cardinal_direction_ko"] == "동쪽"


def test_filter_by_direction_returns_only_sector_matches_nearest_first() -> None:
    ops = SpatialOperatorRegistry()
    center = {"name": "기준", "latitude": 37.5, "longitude": 127.0}
    places = [
        {"name": "먼 북쪽", "latitude": 37.52, "longitude": 127.0},
        {"name": "동쪽", "latitude": 37.5, "longitude": 127.01},
        {"name": "가까운 북쪽", "latitude": 37.51, "longitude": 127.0},
    ]
    matches = ops.invoke(
        "filter_by_direction",
        {"center": center, "places": places, "direction": "북쪽"},
    )
    assert [place["name"] for place in matches] == ["가까운 북쪽", "먼 북쪽"]
    assert all(place["cardinal_direction_ko"] == "북쪽" for place in matches)


def test_place_shaped_operator_inputs_are_normalized_before_computation() -> None:
    ops = SpatialOperatorRegistry()
    geocoded = {
        "query": "기준",
        "place": {"name": "기준", "latitude": 37.5, "longitude": 127.0},
        "candidates": [],
    }
    target = {"place": {"name": "북쪽", "lat": 37.52, "lng": 127.0}}
    other = {"place": {"name": "동쪽", "lat": 37.5, "lng": 127.02}}

    distance = ops.invoke("haversine_distance", {"place_a": geocoded, "place_b": target})
    matches = ops.invoke(
        "filter_by_direction",
        {"center": geocoded, "places": [other, target], "direction": "북쪽"},
    )

    assert 2000 < distance["distance_m"] < 2500
    assert [place["name"] for place in matches] == ["북쪽"]


def test_unresolved_place_input_fails_as_an_explicit_place_error() -> None:
    ops = SpatialOperatorRegistry()
    center = {"name": "기준", "latitude": 37.5, "longitude": 127.0}

    with pytest.raises(ValueError, match="PlaceNotFoundError"):
        ops.invoke("haversine_distance", {"place_a": center, "place_b": None})

    resolved = {"name": "북", "latitude": 37.51, "longitude": 127.0}
    nearest = ops.invoke("nearest", {"anchor": center, "candidates": [None, resolved]})
    assert nearest["nearest"]["name"] == "북"
    assert nearest["nearest"]["candidate_index"] == 1


def test_match_distance_options_accepts_numeric_options_and_measured_records() -> None:
    result = SpatialOperatorRegistry().invoke(
        "match_distance_options",
        {"distance": {"distance_km": 1.058}, "options": [1036, 1061, 1200]},
    )

    assert result["best_option"] == 1
    assert result["computed_distance_m"] == 1058


def test_route_comparison_and_sum() -> None:
    ops = SpatialOperatorRegistry()
    routes = [
        {"distance_m": 3000, "duration_s": 900},
        {"distance_m": 2000, "duration_s": 1200},
    ]
    assert ops.compare_routes(routes)["best_index"] == 1
    assert ops.compare_routes(routes, metric="duration_s")["best_index"] == 0
    assert ops.sum_route_metrics(routes) == {"distance_m": 5000, "duration_s": 2100}


def test_planner_argument_aliases_from_recent_logs_are_supported() -> None:
    ops = SpatialOperatorRegistry()
    distance = ops.invoke(
        "haversine_distance",
        {"lat1": 37.5547, "lng1": 126.9707, "lat2": 37.5512, "lng2": 126.9882},
    )
    assert 1500 < distance["distance_m"] < 1700

    selected = ops.invoke("select_min", {"values": [300, 100, 200]})
    assert selected == {"index": 1, "value": 100}

    summed = ops.invoke(
        "sum_route_metrics",
        {"legs": [1000, 2500], "metric": "distance_m"},
    )
    assert summed == {"distance_m": 3500, "duration_s": 0}

    candidate = ops.invoke(
        "select_min",
        {
            "candidates": {
                "Option 1": {"distance_m": 3500, "duration_s": 0},
                "Option 2": {"distance_m": 5000, "duration_s": 0},
            }
        },
    )
    assert candidate["candidate"] == "Option 1"


def test_a_place_is_not_its_own_nearest_neighbour() -> None:
    """The anchor sits among the candidates often enough that 0.0 m would always win.

    A nearest-convenience-store question lists the convenience store it starts from among its
    options, and a stored retrieval heads its own block; ranked by distance the anchor answers its
    own question, which is never what "가장 가까운" asks.
    """

    ops = SpatialOperatorRegistry()
    spot = {"latitude": 37.542619, "longitude": 126.847355}
    anchor = {"place_id": "a", "name": "GS25 화곡초교점", **spot}
    twin = {"place_id": "elsewhere", "name": "GS25 화곡초교점", **spot}
    neighbour = {
        "place_id": "b",
        "name": "CU 화곡본동점",
        "latitude": 37.543215,
        "longitude": 126.848,
    }

    ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": [anchor, neighbour]})
    assert ranked["nearest"]["name"] == "CU 화곡본동점"

    # Same place under another id, because the context minted one per block entry.
    ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": [twin, neighbour]})
    assert ranked["nearest"]["name"] == "CU 화곡본동점"

    # An empty ranking answers nothing, so the self-match stays when it is all there is.
    ranked = ops.invoke("nearest", {"anchor": anchor, "candidates": [anchor]})
    assert ranked["nearest"]["name"] == "GS25 화곡초교점"


def test_a_direction_filter_drops_the_centre_it_measures_from() -> None:
    ops = SpatialOperatorRegistry()
    centre = {"place_id": "a", "name": "안도로메다", "latitude": 37.5620, "longitude": 126.9881}
    south = {
        "place_id": "b",
        "name": "Seoul Namsan Elementary School",
        "latitude": 37.5570,
        "longitude": 126.9880,
    }

    matches = ops.invoke(
        "filter_by_direction", {"center": centre, "places": [centre, south], "direction": "남쪽"}
    )
    assert [place["name"] for place in matches] == ["Seoul Namsan Elementary School"]
