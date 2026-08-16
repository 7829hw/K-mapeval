from __future__ import annotations

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
