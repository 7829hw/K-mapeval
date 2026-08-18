from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from itertools import permutations
from typing import Any
from zoneinfo import ZoneInfo


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. The one place this repo computes it."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(math.sqrt(hav))


class SpatialOperatorRegistry:
    """Deterministic operators; these never spend Kakao API calls."""

    names = (
        "identity_measure",
        "haversine_distance",
        "pairwise_distances",
        "pairwise_extremes",
        "bearing_to_direction",
        "filter_by_direction",
        "nearest",
        "within_radius",
        "select_min",
        "select_max",
        "sort_by",
        "compare_routes",
        "filter_routes",
        "extract_distance",
        "extract_duration",
        "filter_places",
        "steps_analysis",
        "sum_route_metrics",
        "aggregate_route_groups",
        "merge_places",
        "match_options",
        "match_distance_options",
        "match_type_options",
        "events_from_objects",
        "filter_events",
        "build_route_network",
        "calculate_proportion",
        "open_at_time",
        "timezone",
        "timezone_convert",
        "calculate_start_time",
        "tsp_tw",
    )

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        method = getattr(self, name, None)
        if name not in self.names or method is None:
            raise ValueError(f"Unknown spatial operator: {name}")
        return method(**_normalize_arguments(name, arguments))

    @staticmethod
    def identity_measure(value: Any) -> Any:
        return value

    @staticmethod
    def haversine_distance(place_a: Any, place_b: Any) -> dict[str, float]:
        place_a, place_b = _as_place(place_a, "place_a"), _as_place(place_b, "place_b")
        distance_m = haversine_meters(
            float(place_a["latitude"]),
            float(place_a["longitude"]),
            float(place_b["latitude"]),
            float(place_b["longitude"]),
        )
        return {"distance_m": distance_m, "distance_km": distance_m / 1000}

    @staticmethod
    def bearing_to_direction(place_a: Any, place_b: Any) -> dict[str, Any]:
        place_a, place_b = _as_place(place_a, "place_a"), _as_place(place_b, "place_b")
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
    def pairwise_distances(cls, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs):
            try:
                distance = cls.haversine_distance(pair.get("place_a"), pair.get("place_b"))
            except ValueError:
                results.append(
                    {
                        "pair_index": index,
                        "label": pair.get("label"),
                        "status": "error",
                        "error": "PlaceNotFoundError: unresolved distance endpoint",
                    }
                )
                continue
            results.append({"pair_index": index, "label": pair.get("label"), **distance})
        return results

    @classmethod
    def pairwise_extremes(cls, locations: list[Any]) -> dict[str, Any]:
        locations = [place for _, place in _as_place_list(locations)]
        if len(locations) < 2:
            raise ValueError("pairwise_extremes requires at least two locations")
        pairs = [
            {
                "indexes": [left, right],
                "locations": [locations[left], locations[right]],
                **cls.haversine_distance(locations[left], locations[right]),
            }
            for left in range(len(locations))
            for right in range(left + 1, len(locations))
        ]
        farthest = max(pairs, key=lambda item: float(item["distance_m"]))
        return {"farthest_pair": farthest, "pairs": pairs}

    @classmethod
    def filter_by_direction(
        cls,
        center: Any,
        places: Any,
        direction: str,
    ) -> list[dict[str, Any]]:
        """Return candidates in a cardinal sector, nearest first."""

        expected = _cardinal_direction(direction)
        center = _as_place(center, "center")
        matches: list[dict[str, Any]] = []
        for candidate_index, place in _as_place_list(places):
            bearing = cls.bearing_to_direction(center, place)
            if bearing["cardinal_direction"] != expected:
                continue
            distance = cls.haversine_distance(center, place)
            matches.append({"candidate_index": candidate_index, **place, **bearing, **distance})
        return sorted(matches, key=lambda place: float(place["distance_m"]))

    @classmethod
    def nearest(
        cls,
        anchor: Any,
        candidates: Any,
        metric: str = "haversine",
        routes: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        anchor = _as_place(anchor, "anchor")
        resolved = _as_place_list(candidates)
        if metric == "haversine":
            ranked = [
                {
                    "candidate_index": index,
                    **candidate,
                    **cls.haversine_distance(anchor, candidate),
                }
                for index, candidate in resolved
            ]
            ranked.sort(key=lambda candidate: float(candidate["distance_m"]))
        elif metric == "travel_time":
            route_values = routes.get("routes") if isinstance(routes, dict) else routes
            if not isinstance(route_values, list):
                raise ValueError("travel_time nearest requires aligned routes")
            ranked = []
            for index, candidate in resolved:
                route = next(
                    (
                        item
                        for position, item in enumerate(route_values)
                        if isinstance(item, dict)
                        and item.get("status", "ok") == "ok"
                        and item.get("pair_index", position) == index
                    ),
                    None,
                )
                if route is not None:
                    ranked.append(
                        {
                            "candidate_index": index,
                            **candidate,
                            "duration_s": route["duration_s"],
                            "distance_m": route.get("distance_m"),
                            "route": route,
                        }
                    )
            ranked.sort(key=lambda candidate: float(candidate["duration_s"]))
        else:
            raise ValueError("nearest metric must be haversine or travel_time")
        return {"nearest": ranked[0] if ranked else None, "ranked": ranked}

    @classmethod
    def within_radius(
        cls,
        center: Any,
        candidates: Any,
        radius_m: float,
    ) -> list[dict[str, Any]]:
        center = _as_place(center, "center")
        matches = [
            {"candidate_index": index, **candidate, **cls.haversine_distance(center, candidate)}
            for index, candidate in _as_place_list(candidates)
        ]
        return sorted(
            (candidate for candidate in matches if candidate["distance_m"] <= float(radius_m)),
            key=lambda candidate: float(candidate["distance_m"]),
        )

    @staticmethod
    def select_min(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        comparable = [item for item in items if _has_path(item, key)]
        if not comparable:
            raise ValueError(f"No item contains comparable key: {key}")
        return min(comparable, key=lambda item: float(_path(item, key)))

    @staticmethod
    def select_max(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        comparable = [item for item in items if _has_path(item, key)]
        if not comparable:
            raise ValueError(f"No item contains comparable key: {key}")
        return max(comparable, key=lambda item: float(_path(item, key)))

    @staticmethod
    def sort_by(
        items: list[dict[str, Any]], key: str, descending: bool = False
    ) -> list[dict[str, Any]]:
        return sorted(
            (item for item in items if _has_path(item, key)),
            key=lambda item: float(_path(item, key)),
            reverse=descending,
        )

    @staticmethod
    def compare_routes(routes: list[dict[str, Any]], metric: str = "distance_m") -> dict[str, Any]:
        if metric not in {"distance_m", "duration_s"}:
            raise ValueError("route metric must be distance_m or duration_s")
        best_index = min(range(len(routes)), key=lambda index: float(routes[index][metric]))
        return {"best_index": best_index, "metric": metric, "route": routes[best_index]}

    @staticmethod
    def filter_routes(
        routes: list[dict[str, Any]], keyword: str, include: bool = True
    ) -> dict[str, Any]:
        needle = keyword.casefold()
        matched_indexes: list[int] = []
        matched_routes: list[dict[str, Any]] = []
        for index, route in enumerate(routes):
            instructions = " ".join(
                f"{step.get('instruction', '')} {step.get('road_name', '')}"
                for step in route.get("steps", [])
                if isinstance(step, dict)
            ).casefold()
            contains = needle in instructions
            if contains is include:
                matched_indexes.append(index)
                matched_routes.append(route)
        return {"route_indexes": matched_indexes, "routes": matched_routes}

    @staticmethod
    def extract_distance(route: dict[str, Any]) -> dict[str, float]:
        return {"distance_m": float(route["distance_m"])}

    @staticmethod
    def extract_duration(route: dict[str, Any]) -> dict[str, float]:
        return {"duration_s": float(route["duration_s"])}

    @staticmethod
    def filter_places(
        places: list[dict[str, Any]],
        min_rating: float | None = None,
        price_levels: list[str] | None = None,
        required_types: list[str] | None = None,
        open_now: bool | None = None,
    ) -> list[dict[str, Any]]:
        price_set = {value.casefold() for value in (price_levels or [])}
        type_terms = [value.casefold() for value in (required_types or [])]
        selected: list[dict[str, Any]] = []
        for place in places:
            if min_rating is not None and (
                place.get("rating") is None or float(place["rating"]) < min_rating
            ):
                continue
            if price_set and str(place.get("price_level") or "").casefold() not in price_set:
                continue
            haystack = f"{place.get('category', '')} {place.get('name', '')}".casefold()
            if type_terms and not all(term in haystack for term in type_terms):
                continue
            if open_now is not None and bool(place.get("is_open")) is not open_now:
                continue
            selected.append(place)
        return selected

    @staticmethod
    def steps_analysis(route: dict[str, Any], landmark: str | None = None) -> dict[str, Any]:
        steps = [step for step in route.get("steps", []) if isinstance(step, dict)]
        left = [step for step in steps if "좌회전" in str(step.get("instruction", ""))]
        right = [step for step in steps if "우회전" in str(step.get("instruction", ""))]
        roundabouts = [
            step
            for step in steps
            if re.search(r"회전교차로|로터리|roundabout", str(step.get("instruction", "")), re.I)
        ]
        after_landmark = None
        if landmark:
            for index, step in enumerate(steps[:-1]):
                text = f"{step.get('instruction', '')} {step.get('road_name', '')}"
                if landmark.casefold() in text.casefold():
                    after_landmark = steps[index + 1]
                    break
        return {
            "step_count": len(steps),
            "left_turn_count": len(left),
            "right_turn_count": len(right),
            "roundabout_exit_count": len(roundabouts),
            "instruction_after_landmark": after_landmark,
        }

    @staticmethod
    def sum_route_metrics(routes: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "distance_m": sum(int(route["distance_m"]) for route in routes),
            "duration_s": sum(int(route["duration_s"]) for route in routes),
        }

    @staticmethod
    def aggregate_route_groups(
        routes: list[dict[str, Any]], groups: list[list[int]]
    ) -> dict[str, Any]:
        totals: list[dict[str, Any]] = []
        for option_index, indexes in enumerate(groups):
            selected = [routes[index] for index in indexes]
            errors = [route.get("error") for route in selected if route.get("status") == "error"]
            totals.append(
                {
                    "option_index": option_index,
                    "distance_m": sum(int(route.get("distance_m", 0)) for route in selected),
                    "duration_s": sum(int(route.get("duration_s", 0)) for route in selected),
                    "complete": not errors,
                    "errors": errors,
                }
            )
        complete = [total for total in totals if total["complete"]]
        best_distance = (
            min(complete, key=lambda total: total["distance_m"])["option_index"]
            if complete
            else None
        )
        best_duration = (
            min(complete, key=lambda total: total["duration_s"])["option_index"]
            if complete
            else None
        )
        return {
            "option_totals": totals,
            "best_distance_option": best_distance,
            "best_duration_option": best_duration,
        }

    @staticmethod
    def merge_places(items: list[Any]) -> list[dict[str, Any]]:
        """Merge retrieval branches without losing their normalized place identity."""

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for branch in items:
            values = branch if isinstance(branch, list) else [branch]
            for value in values:
                if not isinstance(value, dict):
                    continue
                key = str(value.get("place_id") or (_name_key(str(value.get("name", "")))))
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(value)
        return merged

    @classmethod
    def match_options(
        cls,
        options: list[str],
        places: Any,
        anchor: Any = None,
        mode: str = "nearest",
        minimum_similarity: float = 0.68,
    ) -> dict[str, Any]:
        """Ground answer options against retrieved POIs using name and spatial evidence."""

        options = [str(option) for option in options]
        anchor_place = _as_place(anchor, "anchor", required=False) if anchor else None
        ranked: list[dict[str, Any]] = []
        for original_rank, place in _as_place_list(places, keep_unresolved=True):
            item = {**place, "retrieval_rank": original_rank}
            if anchor_place is not None:
                try:
                    item.update(cls.haversine_distance(anchor_place, place))
                except ValueError:
                    pass
            ranked.append(item)
        if anchor_place is not None:
            ranked.sort(key=lambda item: float(item.get("distance_m", float("inf"))))
        for rank, item in enumerate(ranked):
            item["rank"] = rank

        if mode == "radius_set":
            return _match_option_sets(options, ranked, minimum_similarity)

        option_matches: list[dict[str, Any]] = []
        for option_index, option in enumerate(options):
            scored = [
                (_name_similarity(option, str(place.get("name", ""))), place)
                for place in ranked
            ]
            similarity, matched = max(scored, key=lambda entry: entry[0], default=(0.0, None))
            option_matches.append(
                {
                    "option_index": option_index,
                    "option": option,
                    "similarity": similarity,
                    "matched": matched if similarity >= minimum_similarity else None,
                    "rank": (
                        matched.get("rank")
                        if matched and similarity >= minimum_similarity
                        else None
                    ),
                    "distance_m": (
                        matched.get("distance_m")
                        if matched and similarity >= minimum_similarity
                        else None
                    ),
                }
            )
        supported = [match for match in option_matches if match["matched"] is not None]
        supported.sort(
            key=lambda match: (
                int(match["rank"]),
                -float(match["similarity"]),
                int(match["option_index"]),
            )
        )
        best = supported[0] if supported else None
        return {
            "mode": mode,
            "retrieved_places": ranked,
            "option_matches": option_matches,
            "best_option": best["option_index"] if best else None,
            "confidence": _match_confidence(best, supported[1] if len(supported) > 1 else None),
        }

    @staticmethod
    def match_distance_options(
        distance: Any,
        options: list[Any],
    ) -> dict[str, Any]:
        distance_m = _distance_value(distance)
        comparisons: list[dict[str, Any]] = []
        for option_index, raw_option in enumerate(options):
            option = str(raw_option)
            match = re.search(r"([\d,.]+)\s*(km|m)?", option, re.IGNORECASE)
            if not match:
                continue
            value = float(match.group(1).replace(",", ""))
            option_m = value * 1000 if (match.group(2) or "m").lower() == "km" else value
            comparisons.append(
                {
                    "option_index": option_index,
                    "option": option,
                    "value_m": option_m,
                    "absolute_error_m": abs(option_m - distance_m),
                }
            )
        comparisons.sort(key=lambda item: (item["absolute_error_m"], item["option_index"]))
        best = comparisons[0] if comparisons else None
        if best is None:
            return {
                "computed_distance_m": distance_m,
                "comparisons": comparisons,
                "best_option": None,
                "error_ratio": None,
                "fits": False,
                "confidence": 0.0,
            }
        # The nearest option is always *some* option, even when the measured distance is kilometres
        # away from every candidate — which means the places were resolved wrong, not that the
        # answer is the least-bad number. Say so instead of reporting a confident match.
        spread = max(item["value_m"] for item in comparisons) or 1.0
        error_ratio = best["absolute_error_m"] / spread
        fits = error_ratio <= 0.25
        return {
            "computed_distance_m": distance_m,
            "comparisons": comparisons,
            "best_option": best["option_index"],
            "error_ratio": round(error_ratio, 4),
            "fits": fits,
            "confidence": (
                1.0
                if best["absolute_error_m"] <= 1
                else 0.9
                if fits
                else max(0.1, 1.0 - error_ratio)
            ),
        }

    @staticmethod
    def match_type_options(place: Any, options: list[Any]) -> dict[str, Any]:
        unwrapped = _unwrap_place(place)
        place = unwrapped if isinstance(unwrapped, dict) else {}
        options = [str(option) for option in options]
        category = str(place.get("category") or "")
        name = str(place.get("name") or "")
        scored = [
            {
                "option_index": index,
                "option": option,
                "similarity": max(
                    _name_similarity(option, category),
                    _name_similarity(option, name),
                    1.0 if _name_key(option) in _name_key(category) else 0.0,
                ),
            }
            for index, option in enumerate(options)
        ]
        scored.sort(key=lambda item: (-float(item["similarity"]), int(item["option_index"])))
        best = scored[0] if scored and scored[0]["similarity"] >= 0.68 else None
        return {
            "place": place,
            "category": category,
            "option_matches": scored,
            "best_option": best["option_index"] if best else None,
            "confidence": best["similarity"] if best else 0.0,
        }

    @staticmethod
    def events_from_objects(
        objects: list[dict[str, Any]],
        event_type: str = "observation",
        timestamp_field: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "event_id": f"event_{index}",
                "event_type": event_type,
                "object": item,
                "timestamp": item.get(timestamp_field) if timestamp_field else None,
            }
            for index, item in enumerate(objects)
            if isinstance(item, dict)
        ]

    @staticmethod
    def filter_events(
        events: list[dict[str, Any]], field: str, operator: str, value: Any
    ) -> list[dict[str, Any]]:
        comparators = {
            "eq": lambda left: left == value,
            "ne": lambda left: left != value,
            "gt": lambda left: left > value,
            "gte": lambda left: left >= value,
            "lt": lambda left: left < value,
            "lte": lambda left: left <= value,
            "contains": lambda left: value in left,
        }
        comparator = comparators.get(operator.lower())
        if comparator is None:
            raise ValueError(f"Unsupported event filter operator: {operator}")
        return [
            event
            for event in events
            if _has_path(event, field) and comparator(_path(event, field))
        ]

    @staticmethod
    def build_route_network(
        nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    @staticmethod
    def calculate_proportion(numerator: Any, denominator: Any) -> dict[str, float]:
        numerator_value = float(len(numerator) if isinstance(numerator, list) else numerator)
        denominator_value = float(
            len(denominator) if isinstance(denominator, list) else denominator
        )
        if denominator_value == 0:
            raise ValueError("Proportion denominator must not be zero")
        proportion = numerator_value / denominator_value
        return {"proportion": proportion, "percentage": proportion * 100}

    @staticmethod
    def open_at_time(
        schedule: dict[str, Any], local_time: str, timezone: str
    ) -> dict[str, Any]:
        if "opening_hours" in schedule:
            schedule = schedule.get("opening_hours") or {}
        moment = _parse_datetime(local_time, timezone)
        weekday = moment.strftime("%A").lower()
        previous = moment - timedelta(days=1)
        interval = schedule.get(weekday) or schedule.get(str(moment.weekday()))
        previous_interval = schedule.get(previous.strftime("%A").lower()) or schedule.get(
            str(previous.weekday())
        )
        intervals = interval if isinstance(interval, list) else [interval] if interval else []
        previous_intervals = (
            previous_interval
            if isinstance(previous_interval, list)
            else [previous_interval]
            if previous_interval
            else []
        )
        current = moment.strftime("%H:%M")
        is_open = any(_time_in_period(current, item, carryover=False) for item in intervals)
        is_open = is_open or any(
            _time_in_period(current, item, carryover=True) for item in previous_intervals
        )
        return {
            "local_time": moment.isoformat(),
            "is_open": is_open,
            "interval": interval,
            "previous_interval": previous_interval,
        }

    @staticmethod
    def timezone(latitude: float, longitude: float, timestamp: int | None = None) -> dict[str, Any]:
        lat, lon = float(latitude), float(longitude)
        if not (32 <= lat <= 39.5 and 123 <= lon <= 133):
            raise ValueError("Offline timezone lookup only covers the Korean benchmark extent")
        zone = ZoneInfo("Asia/Seoul")
        moment = (
            datetime.fromtimestamp(timestamp, zone)
            if timestamp is not None
            else datetime.now(zone)
        )
        return {
            "timezone_id": "Asia/Seoul",
            "timezone_name": moment.tzname(),
            "utc_offset_s": int((moment.utcoffset() or timedelta()).total_seconds()),
        }

    @staticmethod
    def timezone_convert(
        local_time: str, from_timezone: str, to_timezone: str
    ) -> dict[str, str]:
        source = _parse_datetime(local_time, from_timezone)
        target = source.astimezone(ZoneInfo(to_timezone))
        return {
            "source_time": source.isoformat(),
            "converted_time": target.isoformat(),
            "timezone": to_timezone,
        }

    @staticmethod
    def calculate_start_time(
        arrival_time: str, duration_s: float, timezone: str
    ) -> dict[str, Any]:
        arrival = _parse_datetime(arrival_time, timezone)
        start = arrival - timedelta(seconds=float(duration_s))
        return {
            "arrival_time": arrival.isoformat(),
            "duration_s": float(duration_s),
            "start_time": start.isoformat(),
            "timezone": timezone,
        }

    @staticmethod
    def tsp_tw(
        nodes: list[dict[str, Any]],
        distance_matrix: list[list[float]] | dict[str, Any],
        time_windows: list[list[float]] | None = None,
        service_times: list[float] | None = None,
        start_index: int = 0,
        time_budget: float | None = None,
    ) -> dict[str, Any]:
        matrix = (
            distance_matrix.get("matrix")
            if isinstance(distance_matrix, dict)
            else distance_matrix
        )
        if not isinstance(matrix, list) or len(matrix) != len(nodes):
            raise ValueError("tsp_tw distance_matrix must be square and match nodes")
        if len(nodes) > 9:
            raise ValueError("Deterministic tsp_tw supports at most 9 nodes")
        visit_indexes = [index for index in range(len(nodes)) if index != start_index]
        best: dict[str, Any] | None = None
        for order in permutations(visit_indexes):
            route = (start_index, *order)
            elapsed = 0.0
            feasible = True
            for position, node_index in enumerate(route):
                if position:
                    elapsed += float(matrix[route[position - 1]][node_index])
                if time_windows:
                    earliest, latest = map(float, time_windows[node_index])
                    elapsed = max(elapsed, earliest)
                    if elapsed > latest:
                        feasible = False
                        break
                elapsed += float((service_times or [0.0] * len(nodes))[node_index])
                if time_budget is not None and elapsed > float(time_budget):
                    feasible = False
                    break
            if feasible and (best is None or elapsed < best["total_cost"]):
                best = {"order": list(route), "total_cost": elapsed, "feasible": True}
        if best is not None:
            return {**best, "fallback_used": False}
        order = [start_index]
        remaining = set(visit_indexes)
        elapsed = float((service_times or [0.0] * len(nodes))[start_index])
        while remaining:
            current = order[-1]
            candidates = sorted(remaining, key=lambda index: float(matrix[current][index]))
            accepted = None
            for candidate in candidates:
                arrival = elapsed + float(matrix[current][candidate])
                if time_windows:
                    earliest, latest = map(float, time_windows[candidate])
                    arrival = max(arrival, earliest)
                    if arrival > latest:
                        continue
                finish = arrival + float((service_times or [0.0] * len(nodes))[candidate])
                if time_budget is None or finish <= float(time_budget):
                    accepted, elapsed = candidate, finish
                    break
            if accepted is None:
                break
            order.append(accepted)
            remaining.remove(accepted)
        return {
            "order": order,
            "total_cost": elapsed,
            "feasible": not remaining,
            "fallback_used": True,
            "unvisited": sorted(remaining),
        }


def _path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _time_in_period(current: str, period: Any, *, carryover: bool) -> bool:
    if not isinstance(period, dict):
        return False
    opening, closing = str(period.get("open", "00:00")), str(period.get("close", "00:00"))
    if opening == closing:
        return True
    if opening < closing:
        return False if carryover else opening <= current < closing
    return current < closing if carryover else current >= opening


def _has_path(value: dict[str, Any], path: str) -> bool:
    try:
        _path(value, path)
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return True


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


_PLACE_WRAPPER_KEYS = ("place", "location", "nearest", "matched")
_LATITUDE_KEYS = ("latitude", "lat", "y")
_LONGITUDE_KEYS = ("longitude", "lng", "lon", "x")


def _coordinate(value: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, str):
            try:
                return float(candidate)
            except ValueError:
                continue
    return None


def _unwrap_place(value: Any) -> Any:
    """Follow the wrapper shapes operators emit down to the place record inside."""

    current = value
    for _ in range(4):
        if isinstance(current, list):
            current = current[0] if len(current) == 1 else None
        if not isinstance(current, dict):
            return current
        if _coordinate(current, _LATITUDE_KEYS) is not None:
            return current
        nested = next(
            (
                current[key]
                for key in _PLACE_WRAPPER_KEYS
                if isinstance(current.get(key), (dict, list))
            ),
            None,
        )
        if nested is None:
            return current
        current = nested
    return current


def _as_place(value: Any, argument: str, *, required: bool = True) -> dict[str, Any] | None:
    """Normalize a planner-supplied place reference into a coordinate-bearing record.

    Planners routinely hand an operator the object that *contains* a place -- a
    ``batch_geocode`` entry, a ``nearest`` result, a single-branch list -- rather than the
    place itself. Upstream Spatial-Agent resolves those references leniently inside the
    executor, so do the same here instead of failing the whole operator on a shape mismatch.
    """

    place = _unwrap_place(value)
    if isinstance(place, dict):
        latitude = _coordinate(place, _LATITUDE_KEYS)
        longitude = _coordinate(place, _LONGITUDE_KEYS)
        if latitude is not None and longitude is not None:
            return {**place, "latitude": latitude, "longitude": longitude}
    if not required:
        return None
    raise ValueError(f"PlaceNotFoundError: {argument} has no resolved coordinates")


def _as_place_list(
    value: Any, *, keep_unresolved: bool = False
) -> list[tuple[int, dict[str, Any]]]:
    """Normalize a candidate collection, keeping each candidate's original index."""

    values = value if isinstance(value, list) else [value]
    resolved: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(values):
        place = _as_place(item, f"candidate {index}", required=False)
        if place is not None:
            resolved.append((index, place))
            continue
        unwrapped = _unwrap_place(item)
        if keep_unresolved and isinstance(unwrapped, dict):
            resolved.append((index, unwrapped))
    return resolved


def _distance_value(value: Any) -> float:
    """Read a distance in meters from a number or any measured-distance record."""

    if isinstance(value, dict):
        for key in ("distance_m", "distance", "value", "amount", "meters"):
            if key in value:
                return _distance_value(value[key])
        for key in ("distance_km", "km"):
            if key in value:
                return float(value[key]) * 1000
        raise ValueError("match_distance_options requires a measured distance")
    return float(value)


def _name_key(value: str) -> str:
    value = value.casefold().replace("(주)", "")
    replacements = {
        "dunkin donuts": "dunkindonuts",
        "paris baguette": "parisbaguette",
        "home plus express": "homeplusexpress",
    }
    value = replacements.get(" ".join(value.split()), value)
    return "".join(character for character in value if character.isalnum())


def _parse_datetime(value: str, timezone: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    zone = ZoneInfo(timezone)
    return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)


def _name_similarity(left: str, right: str) -> float:
    left_key, right_key = _name_key(left), _name_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    shorter, longer = sorted((left_key, right_key), key=len)
    containment = len(shorter) / len(longer) if shorter in longer else 0.0
    sequence = SequenceMatcher(None, left_key, right_key).ratio()
    if left_key[0] != right_key[0]:
        # Shared generic suffixes (for example, 아트센터) must not make unrelated
        # proper names look like a reliable historical-name match.
        sequence = min(sequence, 0.64)
    return max(sequence, min(0.98, 0.78 + 0.2 * containment) if containment else 0.0)


def _match_option_sets(
    options: list[str], places: list[dict[str, Any]], minimum_similarity: float
) -> dict[str, Any]:
    members = list(
        dict.fromkeys(
            member.strip() for option in options for member in option.split("|") if member.strip()
        )
    )
    member_matches: dict[str, dict[str, Any] | None] = {}
    for member in members:
        scored = [
            (_name_similarity(member, str(place.get("name", ""))), place) for place in places
        ]
        similarity, place = max(scored, key=lambda entry: entry[0], default=(0.0, None))
        member_matches[member] = (
            {"place": place, "similarity": similarity}
            if place is not None and similarity >= minimum_similarity
            else None
        )
    present = {member for member, match in member_matches.items() if match is not None}
    option_scores: list[dict[str, Any]] = []
    for option_index, option in enumerate(options):
        option_members = {member.strip() for member in option.split("|") if member.strip()}
        intersection = len(option_members & present)
        precision = intersection / len(option_members) if option_members else 0.0
        recall = intersection / len(present) if present else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        option_scores.append(
            {
                "option_index": option_index,
                "option": option,
                "members": sorted(option_members),
                "matched_members": sorted(option_members & present),
                "exact_set_match": bool(present) and option_members == present,
                "f1": f1,
            }
        )
    option_scores.sort(
        key=lambda item: (
            -int(item["exact_set_match"]),
            -float(item["f1"]),
            item["option_index"],
        )
    )
    best = option_scores[0] if option_scores and present else None
    return {
        "mode": "radius_set",
        "retrieved_places": places,
        "present_option_members": sorted(present),
        "member_matches": member_matches,
        "option_matches": option_scores,
        "best_option": best["option_index"] if best else None,
        "confidence": 1.0 if best and best["exact_set_match"] else 0.75 if best else 0.0,
    }


def _match_confidence(best: dict[str, Any] | None, second: dict[str, Any] | None) -> float:
    if not best:
        return 0.0
    similarity = float(best["similarity"])
    if second and int(second["rank"]) == int(best["rank"]):
        margin = similarity - float(second["similarity"])
        return 0.9 if margin >= 0.08 else 0.7
    return 0.95 if similarity >= 0.9 else 0.8


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
