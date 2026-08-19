from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from itertools import permutations
from typing import Any
from zoneinfo import ZoneInfo

COORDINATE_LITERAL = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def parse_coordinate_literal(value: str) -> tuple[float, float] | None:
    """A "latitude,longitude" string used where a place is expected.

    An agent that already holds a POI's coordinates asks for what is near *them*, not near a
    place named "37.5771,126.9694". Sending that through the keyword search raises
    PlaceNotFoundError, and a ReAct run then spends its remaining steps re-searching a name that
    was never a name. Coordinates are evidence the agent already has, so resolve them directly.
    Every provider owes the agent this, which is why it lives here and not in one of them.
    """

    match = COORDINATE_LITERAL.fullmatch(value.strip())
    if not match:
        return None
    latitude, longitude = float(match.group(1)), float(match.group(2))
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return latitude, longitude


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
        for candidate_index, place in _excluding_self(center, _as_place_list(places)):
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
        required_type: str | None = None,
    ) -> dict[str, Any]:
        anchor = _as_place(anchor, "anchor")
        resolved = _excluding_self(anchor, _as_place_list(candidates))
        if required_type:
            # The kind asked for decides the answer, and a ranking that ignores it returns the
            # closest place of the wrong kind — which is exactly what a mixed candidate list
            # offers. Kept as a filter, not a requirement: a candidate set that matches nothing
            # is a vocabulary gap in the category strings, not evidence that nothing qualifies.
            terms = [term.casefold() for term in category_terms(required_type)]
            typed = [
                (index, candidate)
                for index, candidate in resolved
                if any(
                    term
                    in f"{candidate.get('category', '')} {candidate.get('name', '')}".casefold()
                    for term in terms
                )
            ]
            if typed:
                resolved = typed
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
                # A metric is the planner's choice of measure, and it has no authority over what
                # evidence exists: asking for travel time without fetching any routes is a plan
                # that forgot a step, not a claim that the candidates are unreachable. Failing
                # here threw away an anchor and a candidate set that were both already resolved,
                # and the generation stage answered from prose — picking the nearest place of any
                # kind, which is exactly the decoy these questions plant. Rank on the geometry
                # that is actually in hand and say so, rather than report a travel time nobody
                # computed.
                metric = "haversine"
                ranked = [
                    {
                        "candidate_index": index,
                        **candidate,
                        **cls.haversine_distance(anchor, candidate),
                    }
                    for index, candidate in resolved
                ]
                ranked.sort(key=lambda candidate: float(candidate["distance_m"]))
                return {
                    "nearest": ranked[0] if ranked else None,
                    "ranked": ranked,
                    "metric_used": "haversine",
                    "metric_requested": "travel_time",
                }
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
        return {"nearest": ranked[0] if ranked else None, "ranked": ranked, "metric_used": metric}

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
        # A requested kind arrives as a Korean noun, as a Kakao category code, or as the words
        # Kakao files the type under; `category_terms` speaks all three. Several types are
        # alternatives — a place of any one of them qualifies — where the old `all` demanded a
        # category path containing every one at once, which no place has.
        type_terms = [
            [term.casefold() for term in category_terms(str(value))]
            for value in (required_types or [])
        ]
        attribute_matches: list[dict[str, Any]] = []
        for _, place in _as_place_list(places, keep_unresolved=True):
            if min_rating is not None and (
                place.get("rating") is None or float(place["rating"]) < min_rating
            ):
                continue
            if price_set and str(place.get("price_level") or "").casefold() not in price_set:
                continue
            if open_now is not None and bool(place.get("is_open")) is not open_now:
                continue
            attribute_matches.append(place)
        if not type_terms:
            return attribute_matches
        selected = [
            place
            for place in attribute_matches
            if any(
                any(term in _category_haystack(place) for term in terms) for terms in type_terms
            )
        ]
        # The kind filter is a preference, exactly as it is in `nearest`: a category vocabulary
        # that does not cover this type is a gap in the lexicon, not evidence that none of these
        # places qualifies. Emptying the list here is worse than not filtering, because the
        # ranking downstream then has nothing to rank and the answer gets guessed.
        return selected or attribute_matches

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
        landmark_index = None
        if landmark:
            for index, step in enumerate(steps):
                text = f"{step.get('instruction', '')} {step.get('road_name', '')}"
                if landmark.casefold() in text.casefold():
                    landmark_index = index
                    after_landmark = steps[index + 1] if index + 1 < len(steps) else None
                    break
        result = {
            "step_count": len(steps),
            "left_turn_count": len(left),
            "right_turn_count": len(right),
            "roundabout_exit_count": len(roundabouts),
            "instruction_after_landmark": after_landmark,
            "landmark_index": landmark_index,
        }
        if landmark_index is None:
            return result

        # A drive is often asked about in halves — "how many left turns *before* I reach X". With
        # totals alone the only available number is the whole route's, which reads as a confident
        # over-count rather than as a missing capability.
        def _count(window: list[dict[str, Any]], pattern: str) -> int:
            return sum(1 for step in window if re.search(pattern, str(step.get("instruction", ""))))

        before, after = steps[:landmark_index], steps[landmark_index + 1 :]
        for label, window in (("before", before), ("after", after)):
            result[f"step_count_{label}_landmark"] = len(window)
            result[f"left_turn_count_{label}_landmark"] = _count(window, "좌회전")
            result[f"right_turn_count_{label}_landmark"] = _count(window, "우회전")
            result[f"roundabout_exit_count_{label}_landmark"] = _count(
                window, r"회전교차로|로터리|roundabout"
            )
        return result

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

        assignments = _assign_unique_matches(options, ranked, minimum_similarity)
        option_matches: list[dict[str, Any]] = []
        for option_index, option in enumerate(options):
            assigned = assignments.get(option_index)
            matched = assigned[0] if assigned else None
            option_matches.append(
                {
                    "option_index": option_index,
                    "option": option,
                    "similarity": (
                        assigned[1]
                        if assigned
                        else max(
                            (
                                _name_similarity(option, str(place.get("name", "")))
                                for place in ranked
                            ),
                            default=0.0,
                        )
                    ),
                    "matched": matched,
                    "rank": matched.get("rank") if matched else None,
                    "distance_m": matched.get("distance_m") if matched else None,
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
        # Relative to the measurement itself, not to the largest option: dividing by the option
        # scale called a 188 m miss on a ~770 m question a fit.
        scale = max(distance_m, best["value_m"], 1.0)
        error_ratio = best["absolute_error_m"] / scale
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
        arrival_time: str,
        duration_s: float,
        timezone: str,
        stay_durations_s: list[float] | None = None,
    ) -> dict[str, Any]:
        """Latest departure that still meets an arrival, counting stops made on the way.

        `duration_s` is travel; anything spent *at* a stop delays departure just as much. A plan
        that summed the legs and forgot the errands answered a whole visit late.
        """

        arrival = _parse_datetime(arrival_time, timezone)
        stay_seconds = sum(float(value) for value in (stay_durations_s or []))
        total = float(duration_s) + stay_seconds
        start = arrival - timedelta(seconds=total)
        return {
            "arrival_time": arrival.isoformat(),
            "duration_s": float(duration_s),
            "stay_duration_s": stay_seconds,
            "total_duration_s": total,
            "start_time": start.isoformat(),
            "derived_clock": "start_time",
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
        end_index: int | None = None,
    ) -> dict[str, Any]:
        matrix = _matrix_argument(distance_matrix, len(nodes))
        if matrix is None:
            raise ValueError("tsp_tw distance_matrix must be square and match nodes")
        if len(nodes) > 9:
            raise ValueError("Deterministic tsp_tw supports at most 9 nodes")
        if end_index is not None and not 0 <= int(end_index) < len(nodes):
            raise ValueError("tsp_tw end_index must name one of the nodes")
        if end_index is not None and int(end_index) == start_index:
            raise ValueError("tsp_tw end_index must differ from start_index")
        # A tour that must end somewhere is not free to end anywhere. "I have an appointment at X
        # at 7pm, with errands on the way" fixes the last stop and leaves only the errands to
        # order; without saying so, the search finds a cheaper route that ends at an errand and
        # answers a departure time for a trip that never reaches the appointment.
        fixed_end = None if end_index is None else int(end_index)
        visit_indexes = [
            index for index in range(len(nodes)) if index not in (start_index, fixed_end)
        ]
        best: dict[str, Any] | None = None
        for order in permutations(visit_indexes):
            route = (start_index, *order) if fixed_end is None else (start_index, *order, fixed_end)
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
                stays = service_times or [0.0] * len(nodes)
                service = sum(float(stays[index]) for index in route)
                best = {
                    "order": list(route),
                    "total_cost": elapsed,
                    # `total_cost` is the whole tour, stays included. Reporting the halves as well
                    # is what stops a planner adding the stays a second time on the way into
                    # `calculate_start_time` — a whole visit, wider than the gap between options.
                    "travel_cost": elapsed - service,
                    "service_cost": service,
                    "feasible": True,
                }
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
        if fixed_end is not None:
            elapsed += float(matrix[order[-1]][fixed_end])
            elapsed += float((service_times or [0.0] * len(nodes))[fixed_end])
            order.append(fixed_end)
        return {
            "order": order,
            "total_cost": elapsed,
            "feasible": not remaining,
            "fallback_used": True,
            "unvisited": sorted(remaining),
        }


# What a kind of place is called in a question is not what Kakao calls it in a category path:
# a subway station is filed under 지하철,전철, a 대형마트 under 슈퍼마켓 > 대형슈퍼. Matching the
# question's noun alone silently emptied the filter for those, which then fell back to the whole
# unfiltered list — the same wrong answer as having no constraint. Terms observed in Kakao's own
# category strings, not invented.
CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "지하철역": ("지하철", "전철"),
    "역": ("지하철", "전철"),
    # Never a bare "마트": Kakao files 이마트24 as 가정,생활 > 편의점 > 이마트24, so the loose
    # term let a convenience-store brand answer a 대형마트 question. Only the words the
    # taxonomy itself uses at the type level.
    "대형마트": ("대형마트", "대형슈퍼", "슈퍼마켓"),
    "마트": ("대형마트", "대형슈퍼", "슈퍼마켓"),
    "은행": ("은행", "금융"),
    "편의점": ("편의점",),
    "약국": ("약국",),
    "주유소": ("주유소", "충전소"),
    "카페": ("카페",),
    "병원": ("병원", "의원", "의료"),
    "주차장": ("주차장", "교통시설"),
    "음식점": ("음식점", "한식", "분식"),
    "학교": ("학교",),
    "숙박시설": ("숙박",),
    "관광명소": ("관광", "명소"),
    "문화시설": ("문화시설", "영화", "공연"),
}


# Kakao's own category codes, in the vocabulary the questions and the aliases speak. A planner
# copies the code out of the prompt's list and writes it where a kind of place belongs, so the
# filter has to read it as that kind rather than look for the letters "CS2" in a category path.
CATEGORY_CODE_NOUNS: dict[str, str] = {
    "MT1": "대형마트",
    "CS2": "편의점",
    "PS3": "어린이집",
    "SC4": "학교",
    "AC5": "학원",
    "PK6": "주차장",
    "OL7": "주유소",
    "SW8": "지하철역",
    "BK9": "은행",
    "CT1": "문화시설",
    "AG2": "부동산",
    "PO3": "공공기관",
    "AT4": "관광명소",
    "AD5": "숙박시설",
    "FD6": "음식점",
    "CE7": "카페",
    "HP8": "병원",
    "PM9": "약국",
}


def category_terms(required_type: str) -> tuple[str, ...]:
    """The strings that identify a requested kind of place inside a Kakao category path."""

    key = "".join(required_type.split()).casefold()
    noun = CATEGORY_CODE_NOUNS.get(key.upper())
    if noun is not None:
        key = noun.casefold()
    for alias, terms in CATEGORY_ALIASES.items():
        if key == alias.casefold():
            return terms
    return (noun or required_type,)


def _category_haystack(place: dict[str, Any]) -> str:
    return f"{place.get('category', '')} {place.get('name', '')}".casefold()


def build_duration_matrix(routes: Any) -> dict[str, Any]:
    """Turn a `distance_matrix` route list into the square matrix `tsp_tw` consumes.

    Without this the paper's flagship trip path is unreachable: `distance_matrix` returns
    `{"routes": [...]}` and `tsp_tw` reads `distance_matrix["matrix"]`, so the only matrix a
    planner could supply was one it invented. Legs are keyed by the endpoint labels the routes
    carry, and a matrix missing any off-diagonal leg is reported as incomplete rather than
    silently filled — an absent leg is missing evidence, not a zero-cost hop.
    """

    if isinstance(routes, dict):
        routes = routes.get("routes", routes)
    entries = [entry for entry in (routes or []) if isinstance(entry, dict)]
    labels: list[str] = []
    for entry in entries:
        for key in ("origin", "destination"):
            label = entry.get(key)
            if isinstance(label, str) and label and label not in labels:
                labels.append(label)
    size = len(labels)
    index_of = {label: index for index, label in enumerate(labels)}
    matrix: list[list[float | None]] = [
        [0.0 if row == column else None for column in range(size)] for row in range(size)
    ]
    for entry in entries:
        if entry.get("status") not in (None, "ok"):
            continue
        row = index_of.get(str(entry.get("origin")))
        column = index_of.get(str(entry.get("destination")))
        if row is None or column is None or row == column:
            continue
        duration = entry.get("duration_s")
        if duration is None:
            continue
        matrix[row][column] = float(duration)
    missing = [
        [labels[row], labels[column]]
        for row in range(size)
        for column in range(size)
        if row != column and matrix[row][column] is None
    ]
    return {"nodes": labels, "matrix": matrix, "missing_legs": missing, "complete": not missing}


def _matrix_argument(value: Any, node_count: int) -> list[list[float]] | None:
    """Accept the shapes a planner can actually produce for `tsp_tw.distance_matrix`."""

    candidate: Any = value
    if isinstance(value, dict):
        candidate = value.get("matrix")
        if candidate is None and "routes" in value:
            built = build_duration_matrix(value)
            candidate = built["matrix"] if built["complete"] else None
    if isinstance(value, list) and value and isinstance(value[0], dict):
        built = build_duration_matrix(value)
        candidate = built["matrix"] if built["complete"] else None
    if not isinstance(candidate, list) or len(candidate) != node_count:
        return None
    if any(not isinstance(row, list) or len(row) != node_count for row in candidate):
        return None
    if any(cell is None for row in candidate for cell in row):
        return None
    return [[float(cell) for cell in row] for row in candidate]


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


def _excluding_self(
    anchor: dict[str, Any], resolved: list[tuple[int, dict[str, Any]]]
) -> list[tuple[int, dict[str, Any]]]:
    """Drop the anchor from its own neighbour ranking — a place is not near itself.

    "가장 가까운 X" never means X, but the anchor is a place of the type being asked about often
    enough to sit among the candidates: the options of a nearest-convenience-store question include
    the convenience store the question starts from, and a stored retrieval heads its own block at
    zero metres. Ranked by distance it wins with 0.0 m every time, and the generation stage then
    reports that faithfully — GS25 화곡초교점 was answered as its own nearest neighbour. A place is
    the anchor when it carries the anchor's id or stands on the same spot. Kept when it is the only
    candidate there is, because an empty ranking answers nothing at all.
    """

    anchor_id = anchor.get("place_id")
    kept = [
        (index, place)
        for index, place in resolved
        if not (
            (anchor_id is not None and place.get("place_id") == anchor_id)
            or _coordinate(place, ("latitude", "lat", "y")) is not None
            and haversine_meters(
                float(anchor["latitude"]),
                float(anchor["longitude"]),
                float(_coordinate(place, ("latitude", "lat", "y")) or 0.0),
                float(_coordinate(place, ("longitude", "lng", "lon", "x")) or 0.0),
            )
            < 1.0
        )
    ]
    return kept or resolved


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
    if values and not resolved:
        # Every candidate was unusable -- bare names the operators cannot look up, or the error
        # markers of a step that failed. Returning the empty list reads downstream as "no
        # candidate qualifies", which is a fabricated negative: a direction filter answered
        # "nothing lies north" and the generation stage then guessed from coordinates it read
        # off the trace. Missing evidence has to fail as missing evidence.
        raise ValueError(
            "PlaceNotFoundError: no candidate carries coordinates "
            f"({len(values)} unresolved)"
        )
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


LOCATION_QUALIFIER = re.compile(
    r"\s+[-–]\s+(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|충북|충남|전북|전라"
    r"|전남|경북|경상|경남|제주)\S*.*$"
)


def strip_location_qualifier(value: str) -> str:
    """Drop the address a dataset appends to tell two same-named options apart.

    Option texts arrive as "버거킹 - 서울특별시 용산구 한강로2가 한강대로 92" when the source
    generator had to separate namesakes. Kakao stores the address in its own field, never in the
    name, so the appended tail is pure noise to every name comparison and to the keyword search:
    it drags similarity far below the matching floor and makes the option unresolvable.
    """

    return LOCATION_QUALIFIER.sub("", value).strip() or value.strip()


def _name_key(value: str) -> str:
    value = strip_location_qualifier(value).casefold().replace("(주)", "")
    replacements = {
        "dunkin donuts": "dunkindonuts",
        "paris baguette": "parisbaguette",
        "home plus express": "homeplusexpress",
    }
    value = replacements.get(" ".join(value.split()), value)
    return "".join(character for character in value if character.isalnum())


# A Korean question states a time in Korean, so a planner copies "오전 10시 00분" straight out of
# it. Accepting only ISO 8601 meant the temporal operators could never be driven from the very
# question they were meant to answer — the tool raised and the agent fell back to guessing.
_KOREAN_CLOCK = re.compile(r"(오전|오후|아침|저녁|밤)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?")
# A planner normalizes the question's "오전 10시 00분" into the machine form before writing it
# into an argument, and `datetime.fromisoformat` takes neither `10:00` nor `17:00:00` as a
# datetime. Rejecting the wall clock a planner actually emits failed the clock step outright,
# and the generation stage then answered a time-window question from prose arithmetic.
_NUMERIC_CLOCK = re.compile(
    r"^\s*(오전|오후|아침|저녁|밤)?\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$"
)


def parse_clock_text(
    value: str, timezone: str, *, reference: datetime | None = None
) -> datetime | None:
    """Read a Korean wall-clock expression, anchored to a reference date.

    Only the clock matters to these questions — the options are clock times — so an expression
    with no date is placed on the reference day rather than rejected.
    """

    match = _KOREAN_CLOCK.search(value) or _NUMERIC_CLOCK.search(value)
    if not match:
        return None
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    if not (0 <= hour <= 24 and 0 <= minute < 60):
        return None
    period = match.group(1)
    if period in ("오후", "저녁", "밤") and hour < 12:
        hour += 12
    elif period in ("오전", "아침") and hour == 12:
        hour = 0
    if hour == 24:
        hour = 0
    zone = ZoneInfo(timezone)
    day = (reference or datetime.now(zone)).astimezone(zone)
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _parse_datetime(value: str, timezone: str) -> datetime:
    zone = ZoneInfo(timezone)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        korean = parse_clock_text(value, timezone)
        if korean is None:
            raise
        return korean
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
    if not containment:
        sequence = min(sequence, distinguishing_similarity(left_key, right_key))
    return max(sequence, min(0.98, 0.78 + 0.2 * containment) if containment else 0.0)


def distinguishing_similarity(left_key: str, right_key: str) -> float:
    """How alike two names are once the part they share with their whole type is removed.

    Korean POI names of the same kind share long generic affixes: 서울오륜초등학교 and
    서울공릉초등학교 agree on six of eight characters and score 0.75, well above the matching
    floor, while naming different schools. What identifies the place is the residue — 오륜 versus
    공릉 — so cap the score by how well *that* matches. Short residues (CU 가락센트럴점 against
    Kakao's CU 가락센타점) are spelling variants of one name, not two names, and are left alone.
    """

    prefix = 0
    while prefix < min(len(left_key), len(right_key)) and left_key[prefix] == right_key[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(left_key), len(right_key)) - prefix
        and left_key[len(left_key) - 1 - suffix] == right_key[len(right_key) - 1 - suffix]
    ):
        suffix += 1
    left_core = left_key[prefix : len(left_key) - suffix]
    right_core = right_key[prefix : len(right_key) - suffix]
    if len(left_core) < 2 or len(right_core) < 2:
        return 1.0
    return SequenceMatcher(None, left_core, right_core).ratio()


def _match_option_sets(
    options: list[str], places: list[dict[str, Any]], minimum_similarity: float
) -> dict[str, Any]:
    members = list(
        dict.fromkeys(
            member.strip() for option in options for member in option.split("|") if member.strip()
        )
    )
    assignments = _assign_unique_matches(members, places, minimum_similarity)
    member_matches: dict[str, dict[str, Any] | None] = {
        member: (
            {"place": assignments[index][0], "similarity": assignments[index][1]}
            if index in assignments
            else None
        )
        for index, member in enumerate(members)
    }
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



def _assign_unique_matches(
    labels: list[str], places: list[dict[str, Any]], minimum_similarity: float
) -> dict[int, tuple[dict[str, Any], float]]:
    """Pair labels with places so that neither side is used twice.

    A retrieved POI is one physical place, so it can support one option at most. Scoring every
    option independently against the whole result set let a single hit answer several options at
    once — 서울오륜초등학교 and 서울평화초등학교 both cleared the floor against the same
    서울공릉초등학교, and the tie-break then handed the answer to whichever came first in the
    option list. Assign the strongest pairings first, nearest place first when scores tie.
    """

    scored = sorted(
        (
            (-_name_similarity(label, str(place.get("name", ""))), place_index, label_index)
            for label_index, label in enumerate(labels)
            for place_index, place in enumerate(places)
        )
    )
    assignments: dict[int, tuple[dict[str, Any], float]] = {}
    claimed: set[int] = set()
    for negative_similarity, place_index, label_index in scored:
        similarity = -negative_similarity
        if similarity < minimum_similarity:
            break
        if label_index in assignments or place_index in claimed:
            continue
        assignments[label_index] = (places[place_index], similarity)
        claimed.add(place_index)
    return assignments


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
