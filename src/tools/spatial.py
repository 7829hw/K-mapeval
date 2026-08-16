from __future__ import annotations

import math
from typing import Any


class SpatialOperatorRegistry:
    """Deterministic operators; these never spend Kakao API calls."""

    names = (
        "haversine_distance",
        "bearing_to_direction",
        "filter_by_direction",
        "select_min",
        "select_max",
        "sort_by",
        "compare_routes",
        "sum_route_metrics",
    )

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        method = getattr(self, name, None)
        if name not in self.names or method is None:
            raise ValueError(f"Unknown spatial operator: {name}")
        return method(**_normalize_arguments(name, arguments))

    @staticmethod
    def haversine_distance(place_a: dict[str, Any], place_b: dict[str, Any]) -> dict[str, float]:
        lat1, lon1 = float(place_a["latitude"]), float(place_a["longitude"])
        lat2, lon2 = float(place_b["latitude"]), float(place_b["longitude"])
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        hav = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        distance_m = 2 * 6_371_008.8 * math.asin(math.sqrt(hav))
        return {"distance_m": distance_m, "distance_km": distance_m / 1000}

    @staticmethod
    def bearing_to_direction(place_a: dict[str, Any], place_b: dict[str, Any]) -> dict[str, Any]:
        lat1 = math.radians(float(place_a["latitude"]))
        lat2 = math.radians(float(place_b["latitude"]))
        delta_lon = math.radians(float(place_b["longitude"]) - float(place_a["longitude"]))
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
        directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        direction = directions[round(bearing / 45) % 8]
        direction_ko = {
            "N": "북쪽",
            "NE": "북동쪽",
            "E": "동쪽",
            "SE": "남동쪽",
            "S": "남쪽",
            "SW": "남서쪽",
            "W": "서쪽",
            "NW": "북서쪽",
        }[direction]
        cardinal_directions = ("N", "E", "S", "W")
        cardinal_direction = cardinal_directions[round(bearing / 90) % 4]
        return {
            "bearing_degrees": bearing,
            "direction": direction,
            "direction_ko": direction_ko,
            "cardinal_direction": cardinal_direction,
            "cardinal_direction_ko": {"N": "북쪽", "E": "동쪽", "S": "남쪽", "W": "서쪽"}[
                cardinal_direction
            ],
        }

    @classmethod
    def filter_by_direction(
        cls,
        center: dict[str, Any],
        places: list[dict[str, Any]],
        direction: str,
    ) -> list[dict[str, Any]]:
        """Return candidates in a cardinal sector, nearest first."""

        expected = _cardinal_direction(direction)
        matches: list[dict[str, Any]] = []
        for place in places:
            bearing = cls.bearing_to_direction(center, place)
            if bearing["cardinal_direction"] != expected:
                continue
            distance = cls.haversine_distance(center, place)
            matches.append({**place, **bearing, **distance})
        return sorted(matches, key=lambda place: float(place["distance_m"]))

    @staticmethod
    def select_min(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return min(items, key=lambda item: float(_path(item, key)))

    @staticmethod
    def select_max(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return max(items, key=lambda item: float(_path(item, key)))

    @staticmethod
    def sort_by(
        items: list[dict[str, Any]], key: str, descending: bool = False
    ) -> list[dict[str, Any]]:
        return sorted(items, key=lambda item: float(_path(item, key)), reverse=descending)

    @staticmethod
    def compare_routes(routes: list[dict[str, Any]], metric: str = "distance_m") -> dict[str, Any]:
        if metric not in {"distance_m", "duration_s"}:
            raise ValueError("route metric must be distance_m or duration_s")
        best_index = min(range(len(routes)), key=lambda index: float(routes[index][metric]))
        return {"best_index": best_index, "metric": metric, "route": routes[best_index]}

    @staticmethod
    def sum_route_metrics(routes: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "distance_m": sum(int(route["distance_m"]) for route in routes),
            "duration_s": sum(int(route["duration_s"]) for route in routes),
        }


def _path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _normalize_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Accept common planner aliases while keeping one canonical operator implementation."""

    args = dict(arguments)
    if name in {"haversine_distance", "bearing_to_direction"}:
        if "place_a" in args and "place_b" in args:
            return {"place_a": args["place_a"], "place_b": args["place_b"]}
        prefixes = (
            ("lat1", "lon1", "lat2", "lon2"),
            ("lat1", "lng1", "lat2", "lng2"),
            ("start_lat", "start_lng", "end_lat", "end_lng"),
            ("start_lat", "start_lon", "end_lat", "end_lon"),
        )
        for lat1, lon1, lat2, lon2 in prefixes:
            if all(key in args for key in (lat1, lon1, lat2, lon2)):
                return {
                    "place_a": {
                        "latitude": args[lat1],
                        "longitude": args[lon1],
                    },
                    "place_b": {
                        "latitude": args[lat2],
                        "longitude": args[lon2],
                    },
                }
        raise ValueError(f"{name} requires place_a/place_b or two coordinate pairs")

    if name in {"select_min", "select_max"}:
        if "items" in args:
            return {"items": args["items"], "key": args.get("key", "value")}
        source = next(
            (args[key] for key in ("values", "inputs", "list", "candidates") if key in args),
            None,
        )
        if isinstance(source, dict):
            items = [
                {"candidate": candidate, "value": value} for candidate, value in source.items()
            ]
        elif isinstance(source, list):
            items = [
                item if isinstance(item, dict) else {"index": index, "value": item}
                for index, item in enumerate(source)
            ]
        else:
            raise ValueError(f"{name} requires items, values, inputs, list, or candidates")
        key = args.get("key") or _comparison_value_path(items)
        return {"items": items, "key": key}

    if name == "sum_route_metrics":
        routes = next(
            (args[key] for key in ("routes", "inputs", "legs") if key in args),
            None,
        )
        if not isinstance(routes, list):
            raise ValueError("sum_route_metrics requires routes, inputs, or legs")
        if routes and all(isinstance(route, dict) for route in routes):
            return {"routes": routes}
        metric = str(args.get("metric", "distance_m"))
        if metric not in {"distance_m", "duration_s"}:
            raise ValueError("sum metric must be distance_m or duration_s")
        return {
            "routes": [
                {
                    "distance_m": value if metric == "distance_m" else 0,
                    "duration_s": value if metric == "duration_s" else 0,
                }
                for value in routes
            ]
        }

    return args


def _cardinal_direction(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    aliases = {
        "n": "N",
        "north": "N",
        "북": "N",
        "북쪽": "N",
        "e": "E",
        "east": "E",
        "동": "E",
        "동쪽": "E",
        "s": "S",
        "south": "S",
        "남": "S",
        "남쪽": "S",
        "w": "W",
        "west": "W",
        "서": "W",
        "서쪽": "W",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("direction must be north/east/south/west (북쪽/동쪽/남쪽/서쪽)") from exc


def _comparison_value_path(items: list[dict[str, Any]]) -> str:
    values = [item.get("value") for item in items]
    if values and all(isinstance(value, dict) and "distance_m" in value for value in values):
        return "value.distance_m"
    if values and all(isinstance(value, dict) and "duration_s" in value for value in values):
        return "value.duration_s"
    return "value"
