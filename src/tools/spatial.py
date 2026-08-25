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


# A distance written in an option: a number that starts with a digit, and the unit that follows
# it. `_BARE_NUMBER` is the fallback for an option that states a plain metre count.
_MEASURED_DISTANCE = r"(\d[\d,]*(?:\.\d+)?)\s*(?P<unit>km|m)\b"
_BARE_NUMBER = r"(\d[\d,]*(?:\.\d+)?)"


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
        "select_by_index",
        "compare_routes",
        "filter_routes",
        "extract_distance",
        "extract_duration",
        "filter_places",
        "steps_analysis",
        "sum_route_metrics",
        "sum_amounts",
        "difference",
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
            typed = [
                (index, candidate)
                for index, candidate in resolved
                if matches_required_type(candidate, required_type)
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
        # A place is not near itself, here as in `nearest` and `filter_by_direction`. The anchor
        # is a place of the kind being asked about often enough to end up among the options, and
        # at 0 m it satisfies every radius — in one question it was the *only* place inside 600 m,
        # which reported the anchor as the answer to "which mart is within 600 m of it".
        matches = [
            {"candidate_index": index, **candidate, **cls.haversine_distance(center, candidate)}
            for index, candidate in _excluding_self(center, _as_place_list(candidates))
        ]
        return sorted(
            (candidate for candidate in matches if candidate["distance_m"] <= float(radius_m)),
            key=lambda candidate: float(candidate["distance_m"]),
        )

    @staticmethod
    def select_min(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        comparable = [item for item in items if _has_comparable(item, key)]
        if not comparable:
            raise ValueError(f"No item contains comparable key: {key}")
        return min(comparable, key=lambda item: float(_path(item, key)))

    @staticmethod
    def select_max(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        comparable = [item for item in items if _has_comparable(item, key)]
        if not comparable:
            raise ValueError(f"No item contains comparable key: {key}")
        return max(comparable, key=lambda item: float(_path(item, key)))

    @staticmethod
    def sort_by(
        items: list[dict[str, Any]], key: str, descending: bool = False
    ) -> list[dict[str, Any]]:
        return sorted(
            (item for item in items if _has_comparable(item, key)),
            key=lambda item: float(_path(item, key)),
            reverse=descending,
        )

    @classmethod
    def select_by_index(
        cls,
        items: Any,
        index: int,
        key: str | None = None,
        descending: bool = False,
    ) -> dict[str, Any]:
        """The k-th item of a ranked collection, which nothing else here could reach.

        `sort_by` orders a list and `select_min`/`select_max` take an end off it, so an ordinal
        question -- "the second closest", "the third furthest" -- had no operator to finish on.
        Planners wrote one anyway: across the runs in `logs/` they invented `select_by_index` six
        times and `select_second_closest`, `select_second_nearest`, `select_second_min` and
        `select_subset` once each, which is eleven questions lost to a missing operator rather
        than to reasoning.

        `index` is 0-based, like every other index in this project, so the second item is index 1
        -- and that is what the planners themselves wrote when they invented the name. An index
        outside the collection fails rather than clamping to an end: the nearest item is not the
        second nearest, and answering as though it were is a fabricated measurement.
        """

        collection = _amount_collection(items)
        if key is not None:
            collection = cls.sort_by(collection, key, descending)
        if not collection:
            raise ValueError("select_by_index received an empty collection")
        position = int(index)
        if not -len(collection) <= position < len(collection):
            raise ValueError(
                f"select_by_index({position}) is outside a collection of {len(collection)}; "
                "the index is 0-based, so the second item is index 1"
            )
        chosen = collection[position]
        if isinstance(chosen, dict):
            return {"selected_index": position, **chosen}
        return {"selected_index": position, "value": chosen}

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
    def extract_distance(route: Any) -> dict[str, float] | list[dict[str, float]]:
        """One route's distance, or one per route when handed the list.

        `extract_distance(routes="$segments.routes")` after a three-leg `distance_matrix` is what
        a planner writes when it wants the legs measured before adding them, and it means one
        unambiguous thing. `_normalize_arguments` maps the plural spelling onto this slot; a list
        arriving here is measured element-wise rather than raising on `list["distance_m"]`.
        """

        return _extract_metric(route, "distance_m")

    @staticmethod
    def extract_duration(route: Any) -> dict[str, float] | list[dict[str, float]]:
        return _extract_metric(route, "duration_s")

    @staticmethod
    def filter_places(
        places: list[dict[str, Any]],
        min_rating: float | None = None,
        price_levels: list[str] | None = None,
        required_types: list[str] | None = None,
        open_now: bool | None = None,
    ) -> list[dict[str, Any]]:
        price_set = {value.casefold() for value in (price_levels or [])}
        candidates = [place for _, place in _as_place_list(places, keep_unresolved=True)]
        # A filter over a field none of these places carries is not a filter, it is a way to
        # return nothing: Kakao Local publishes no rating, no price level and no opening hours, so
        # `min_rating=4.0` deleted every candidate and the ranking after it answered from an empty
        # list. Where some place does carry the field, an empty result is real evidence and the
        # filter stands. Same rule as the kind filter below, one attribute earlier.
        rated = evidence_carries(candidates, "rating")
        priced = evidence_carries(candidates, "price_level")
        timed = evidence_carries(candidates, "is_open")
        attribute_matches: list[dict[str, Any]] = []
        for place in candidates:
            if min_rating is not None and rated and float(place.get("rating") or 0) < min_rating:
                continue
            if price_set and priced and (
                str(place.get("price_level") or "").casefold() not in price_set
            ):
                continue
            if open_now is not None and timed and bool(place.get("is_open")) is not open_now:
                continue
            attribute_matches.append(place)
        if not required_types:
            return attribute_matches
        selected = [
            place
            for place in attribute_matches
            if any(matches_required_type(place, required) for required in required_types or [])
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
    def sum_amounts(amounts: Any, key: str | None = None) -> dict[str, Any]:
        """Add measurements that separate nodes produced.

        `sum_route_metrics` totals a route list and `aggregate_route_groups` totals route indexes
        per option, but a graph that measured two legs with `extract_distance` and wanted their
        total had nothing to add them with. Planners invented `sum_amounts` in four questions and
        `calculate_total_distance`/`calculate_path_distance` in three more.

        A sum of seconds is not a sum of metres, so the result carries `duration_s` without
        `value`: `match_distance_options` reads `value` as metres, and a plan that pipes a
        duration into it has to fail where it stands rather than answer in the wrong unit.
        """

        values = _amount_collection(amounts)
        if not values:
            raise ValueError("sum_amounts received nothing to add")
        records = [item for item in values if isinstance(item, dict)]
        result: dict[str, Any] = {"count": len(values)}
        if key is None and len(records) == len(values):
            # Route-shaped records carry both metrics, and a trip total wants each of them.
            for metric in ("distance_m", "duration_s"):
                if all(metric in item for item in records):
                    result[metric] = float(
                        sum(_amount_number(item[metric], where="sum_amounts") for item in records)
                    )
        addends = [_amount_number(item, key, where="sum_amounts") for item in values]
        total = float(sum(addends))
        result["addends"] = addends
        result["total"] = total
        kind = _amount_kind(key, values)
        if kind == "distance" and "distance_m" not in result:
            result["distance_m"] = total
        if "distance_m" in result:
            result["distance_km"] = result["distance_m"] / 1000
        if kind != "duration":
            result["value"] = total
        return result

    @staticmethod
    def difference(minuend: Any, subtrahend: Any, key: str | None = None) -> dict[str, Any]:
        """One measurement less another, which the operator set had no way to express.

        A detour cost is a subtraction and so is "how much farther is A than B"; both families
        exist in these benchmarks and both had to be composed out of operators that only ever
        added. Planners wrote `subtraction` and `calculate_difference` instead.

        `difference` keeps the sign so a plan can tell which way round it subtracted, while
        `value` carries the magnitude: an option states the ordering in words -- "약 3.2km 더
        멀다" -- and leaves the number positive, so a graph that happened to subtract the other
        way would otherwise match no option at all.
        """

        first = _amount_number(minuend, key, where="difference")
        second = _amount_number(subtrahend, key, where="difference")
        signed = first - second
        magnitude = abs(signed)
        result: dict[str, Any] = {
            "minuend": first,
            "subtrahend": second,
            "difference": signed,
            "absolute_difference": magnitude,
            "value": magnitude,
        }
        if _amount_kind(key, [minuend, subtrahend]) == "distance":
            result["distance_m"] = magnitude
            result["distance_km"] = magnitude / 1000
        return result

    @staticmethod
    def aggregate_route_groups(
        routes: list[dict[str, Any]], groups: list[list[int]]
    ) -> dict[str, Any]:
        # `$matrix` is the whole `distance_matrix` node; indexing a dict by 0 raised KeyError and
        # lost every option total the question compares.
        routes = _route_list(routes)
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
            # The number has to start with a digit, and the measurement with a unit wins. The old
            # `[\d,.]+` matched the comma in "남쪽, 약 6.6km" and then called float("") — every
            # option of a direction-and-distance question failed on its own separator.
            match = re.search(_MEASURED_DISTANCE, option, re.IGNORECASE) or re.search(
                _BARE_NUMBER, option
            )
            if not match:
                continue
            value = float(match.group(1).replace(",", ""))
            unit = (match.groupdict().get("unit") or "m").lower()
            option_m = value * 1000 if unit == "km" else value
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
        stay_seconds = sum(_duration_value(value) for value in (stay_durations_s or []))
        travel_seconds = _duration_value(duration_s)
        total = travel_seconds + stay_seconds
        start = arrival - timedelta(seconds=total)
        return {
            "arrival_time": arrival.isoformat(),
            "duration_s": travel_seconds,
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
        fixed_order: bool = False,
        metric: str = "duration",
        return_to_start: bool = False,
    ) -> dict[str, Any]:
        """Order an itinerary, or walk one the question already ordered.

        `fixed_order` is not a tuning knob, it is the other half of the problem. "몇 곳을 방문할
        수 있나요" over stops listed 적힌 순서대로 states the sequence and asks how much of it
        fits; permuting it answers a question nobody asked. Left to reorder, the greedy fallback
        below visited stops nearest-first and reported one more than the stated order can reach,
        which was 15 of 26 misses in that family.
        """

        if metric not in MATRIX_METRICS:
            raise ValueError(f"tsp_tw metric must be one of {sorted(MATRIX_METRICS)}")
        clock = (service_times, time_windows, time_budget)
        if metric != "duration" and any(value is not None for value in clock):
            # Seconds have no meaning in a matrix of metres. A stay added to a distance, or a
            # budget compared against one, is an invented measurement, and the tour it picks is
            # arithmetic nobody can read. Refuse rather than let the units cancel out silently.
            raise ValueError(
                "tsp_tw metric='distance' measures metres, so service_times, time_windows and "
                "time_budget do not apply; drop them or ask for metric='duration'"
            )
        matrix = _matrix_argument(distance_matrix, len(nodes), metric)
        if matrix is None:
            raise ValueError("tsp_tw distance_matrix must be square and match nodes")
        if len(nodes) > 9:
            raise ValueError("Deterministic tsp_tw supports at most 9 nodes")
        if end_index is not None and not 0 <= int(end_index) < len(nodes):
            raise ValueError("tsp_tw end_index must name one of the nodes")
        if end_index is not None and int(end_index) == start_index:
            raise ValueError("tsp_tw end_index must differ from start_index")
        if fixed_order:
            # The sequence is the question's, so `end_index` has nothing left to fix: whatever the
            # order ends on is where the trip ends.
            return _walk_stated_order(
                nodes, matrix, time_windows, service_times, start_index, time_budget, metric
            )
        # A tour that must end somewhere is not free to end anywhere. "I have an appointment at X
        # at 7pm, with errands on the way" fixes the last stop and leaves only the errands to
        # order; without saying so, the search finds a cheaper route that ends at an errand and
        # answers a departure time for a trip that never reaches the appointment.
        # "…를 둘러본 뒤 다시 제일모텔로 돌아옵니다" is a closed tour, and the cheapest open path
        # is not the cheapest loop: the order that ends furthest from the start looks best right
        # up until the drive home is counted. `end_index` cannot express it — it refuses to name
        # the start — so the closing leg is its own flag.
        if return_to_start and end_index is not None and int(end_index) != start_index:
            raise ValueError("tsp_tw cannot both return to the start and end somewhere else")
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
            if feasible and return_to_start:
                elapsed += float(matrix[route[-1]][start_index])
                if time_budget is not None and elapsed > float(time_budget):
                    feasible = False
            if feasible and (best is None or elapsed < best["total_cost"]):
                stays = service_times or [0.0] * len(nodes)
                service = sum(float(stays[index]) for index in route)
                best = {
                    "order": [*route, start_index] if return_to_start else list(route),
                    # How many of the requested stops the trip actually reaches, start excluded.
                    # "몇 곳을 방문할 수 있나요" is answered by this number, and leaving it to be
                    # counted off `order` in prose is what made the count a guess. The drive home
                    # is not a visit, so a closed tour counts the same stops an open one does.
                    "visited_count": len(route) - 1,
                    "unvisited": [],
                    "total_cost": elapsed,
                    # `total_cost` is the whole tour, stays included. Reporting the halves as well
                    # is what stops a planner adding the stays a second time on the way into
                    # `calculate_start_time` — a whole visit, wider than the gap between options.
                    "travel_cost": elapsed - service,
                    "service_cost": service,
                    "feasible": True,
                    "objective": "optimal_order",
                    "metric": MATRIX_METRICS[metric],
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
            finish = (
                elapsed
                + float(matrix[order[-1]][fixed_end])
                + float((service_times or [0.0] * len(nodes))[fixed_end])
            )
            # The end was appended unconditionally, so a tour that had already spent its budget
            # came back naming a stop it cannot reach and a `total_cost` above the `time_budget`
            # it was handed. A stop that does not fit is unvisited, like any other.
            if time_budget is None or finish <= float(time_budget):
                elapsed = finish
                order.append(fixed_end)
            else:
                remaining.add(fixed_end)
        stays = service_times or [0.0] * len(nodes)
        service = sum(float(stays[index]) for index in order)
        return {
            "order": order,
            "visited_count": len(order) - 1,
            "total_cost": elapsed,
            "travel_cost": elapsed - service,
            "service_cost": service,
            "feasible": not remaining,
            "fallback_used": True,
            "unvisited": sorted(remaining),
            # The whole tour did not fit, so this answers a *different* question than the one that
            # failed: how many stops a nearest-first walk reaches. Say so, rather than let a
            # reader take `order` for the itinerary they asked about.
            "objective": "greedy_partial",
            "metric": MATRIX_METRICS[metric],
        }


def _walk_stated_order(
    nodes: list[dict[str, Any]],
    matrix: list[list[float]],
    time_windows: list[list[float]] | None,
    service_times: list[float] | None,
    start_index: int,
    time_budget: float | None,
    metric: str = "duration",
) -> dict[str, Any]:
    """Follow the itinerary as listed and report how much of it fits.

    The stops arrive in the order the question wrote them, so the only question left is where the
    budget runs out. Travel counts: a walk that adds only the stays reaches exactly one stop too
    many, which is the shape of nearly every miss this family recorded.
    """

    stays = [float(value) for value in (service_times or [0.0] * len(nodes))]
    sequence = [start_index, *(index for index in range(len(nodes)) if index != start_index)]
    order = [start_index]
    elapsed = 0.0
    unvisited: list[int] = []
    for previous, current in zip(sequence, sequence[1:], strict=False):
        if unvisited:
            # Once one stop is out of reach the rest of a stated order is too: they sit behind it.
            unvisited.append(current)
            continue
        arrival = elapsed + float(matrix[previous][current])
        if time_windows:
            earliest, latest = map(float, time_windows[current])
            arrival = max(arrival, earliest)
            if arrival > latest:
                unvisited.append(current)
                continue
        finish = arrival + stays[current]
        if time_budget is not None and finish > float(time_budget):
            unvisited.append(current)
            continue
        elapsed = finish
        order.append(current)
    service = sum(stays[index] for index in order)
    return {
        "order": order,
        "visited_count": len(order) - 1,
        "total_cost": elapsed,
        "travel_cost": elapsed - service,
        "service_cost": service,
        "feasible": not unvisited,
        "fallback_used": False,
        "unvisited": unvisited,
        "objective": "stated_order",
        "metric": MATRIX_METRICS[metric],
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


def evidence_carries(places: list[dict[str, Any]], field: str) -> bool:
    """Whether any candidate actually carries the field a filter would test.

    An attribute filter says which of these places qualifies. Over a field the evidence source
    never populates it says something else — that none of them do — and a source that publishes
    no ratings is not a source in which every place is unrated. So the filter is dropped when the
    field is absent everywhere, and applied normally the moment one place carries it.
    """

    return any(place.get(field) is not None for place in places)


def _category_haystack(place: dict[str, Any]) -> str:
    return f"{place.get('category', '')} {place.get('name', '')}".casefold()


# The vocabulary of kinds this lexicon can name, in the terms each one wears inside a Kakao
# category path. Both tables speak about types, so both are vocabulary: the aliases are the words
# a question uses, the code nouns are the words a planner copies out of the operator prompt.
TYPE_VOCABULARY: dict[str, tuple[str, ...]] = {
    noun: category_terms(noun)
    for noun in sorted({*CATEGORY_ALIASES, *CATEGORY_CODE_NOUNS.values()})
}


def _finer_type_overrides(category: Any, required_terms: tuple[str, ...]) -> bool:
    """Whether the path names a *more specific* kind than the one asked for, below the match.

    Kakao files a category as a path from coarse to fine — `음식점 > 카페 > 커피전문점`,
    `의료,건강 > 약국` — so the word that names a kind also names the parent of its neighbours,
    and a question about a meal was answered by a cafe 220 m nearer than the restaurant it meant.
    The taxonomy already says which of the two the place is: whichever kind this lexicon can name
    sits *deepest*. So the rule is structural rather than a list of pairs to keep in step with the
    benchmark — every kind the vocabulary knows excludes every coarser kind above it, including
    pairs nobody has hit yet.
    """

    levels = [level.strip() for level in str(category or "").split(">")]
    matched = -1
    for index, level in enumerate(levels):
        if any(term in level for term in required_terms):
            matched = index
    if matched < 0:
        return False
    own = set(required_terms)
    for level in levels[matched + 1 :]:
        for terms in TYPE_VOCABULARY.values():
            if own.intersection(terms):
                continue  # the same kind under one of its own other names
            if any(term in level for term in terms):
                return True
    return False


def matches_required_type(place: dict[str, Any], required_type: str) -> bool:
    """Whether a place is of the kind asked for, in Kakao's own category vocabulary."""

    terms = category_terms(str(required_type))
    if not any(term.casefold() in _category_haystack(place) for term in terms):
        return False
    return not _finer_type_overrides(place.get("category"), terms)


MATRIX_METRICS: dict[str, str] = {"duration": "duration_s", "distance": "distance_m"}


def build_duration_matrix(routes: Any, metric: str = "duration") -> dict[str, Any]:
    """Turn a `distance_matrix` route list into the square matrix `tsp_tw` consumes.

    Without this the paper's flagship trip path is unreachable: `distance_matrix` returns
    `{"routes": [...]}` and `tsp_tw` reads `distance_matrix["matrix"]`, so the only matrix a
    planner could supply was one it invented. Legs are keyed by the endpoint labels the routes
    carry, and a matrix missing any off-diagonal leg is reported as incomplete rather than
    silently filled — an absent leg is missing evidence, not a zero-cost hop.

    `metric` decides which of the two numbers every leg carries fills it. "총 주행거리가 가장 짧은
    방문 순서" asks for metres and the tours it chooses between are separated by about 2% of their
    length, so ranking them by seconds is not an approximation of ranking them by metres: over the
    `trip_optimal_order` rows here, replayed on real cached legs, the distance-optimal order is the
    gold answer 93 times in 114 and the duration-optimal order 42 times.
    """

    field = MATRIX_METRICS.get(metric)
    if field is None:
        raise ValueError(f"Unknown matrix metric {metric!r}: use one of {sorted(MATRIX_METRICS)}")

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
        measure = entry.get(field)
        if measure is None:
            continue
        matrix[row][column] = float(measure)
    missing = [
        [labels[row], labels[column]]
        for row in range(size)
        for column in range(size)
        if row != column and matrix[row][column] is None
    ]
    return {
        "nodes": labels,
        "matrix": matrix,
        "missing_legs": missing,
        "complete": not missing,
        "metric": field,
    }


def _matrix_argument(
    value: Any, node_count: int, metric: str = "duration"
) -> list[list[float]] | None:
    """Accept the shapes a planner can actually produce for `tsp_tw.distance_matrix`."""

    candidate: Any = value
    if isinstance(value, dict):
        # The routes are the source of truth and every leg carries both numbers, so read them
        # again in the metric that was asked for. `distance_matrix` returns a pre-built `matrix`
        # *beside* its routes, and that one is always durations — preferring it is how asking for
        # metres silently got seconds, which made this whole argument a no-op on the one input
        # shape the graphs actually use.
        built = build_duration_matrix(value, metric) if "routes" in value else None
        if built is not None and built["complete"]:
            candidate = built["matrix"]
        else:
            candidate = value.get("matrix")
            built_metric = value.get("metric")
            if candidate is not None and MATRIX_METRICS[metric] != (built_metric or "duration_s"):
                # No routes left to re-read, and the matrix in hand is in the other unit. An
                # unlabelled one is a duration matrix: that is what every producer here emits.
                raise ValueError(
                    f"tsp_tw was asked for {metric} but the matrix it was given holds "
                    f"{built_metric or 'duration_s'} and carries no routes to re-read"
                )
    if isinstance(value, list) and value and isinstance(value[0], dict):
        built = build_duration_matrix(value, metric)
        candidate = built["matrix"] if built["complete"] else None
    if not isinstance(candidate, list) or len(candidate) != node_count:
        return None
    if any(not isinstance(row, list) or len(row) != node_count for row in candidate):
        return None
    if any(cell is None for row in candidate for cell in row):
        return None
    return [[float(cell) for cell in row] for row in candidate]


# What a planner calls a metric, against what the operators store it as. A unit conversion is
# never an alias -- `distance_km` is not `distance_m` -- and `amount` names no metric at all, so
# it resolves only when the item carries exactly one of them and there is nothing to guess.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "distance": ("distance_m",),
    "duration": ("duration_s",),
    "travel_time": ("duration_s",),
    "cost": ("total_cost",),
}
_AMBIGUOUS_METRICS = ("distance_m", "duration_s")
# Wrappers a planner points at instead of the record inside, the same shapes `_as_place` unwraps.
_PATH_WRAPPERS = ("place", "location", "route", "nearest", "value", "result", "farthest_pair")


def _path(value: dict[str, Any], path: str) -> Any:
    try:
        return _exact_path(value, path)
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    head, _, remainder = path.partition(".")
    for alias in _METRIC_ALIASES.get(head, ()):
        try:
            return _exact_path(value, f"{alias}.{remainder}" if remainder else alias)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    if head == "amount" and isinstance(value, dict):
        present = [metric for metric in _AMBIGUOUS_METRICS if metric in value]
        if len(present) == 1:
            return value[present[0]]
    if isinstance(value, dict):
        for wrapper in _PATH_WRAPPERS:
            inner = value.get(wrapper)
            if isinstance(inner, dict | list):
                try:
                    return _path(inner, path)
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
    raise KeyError(path)


def _exact_path(value: dict[str, Any], path: str) -> Any:
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


def _has_comparable(value: dict[str, Any], path: str) -> bool:
    """Whether the item carries that key *as a number*, which is what ranking it needs.

    A key that is present and null is not a comparable value: `select_min` raised
    `TypeError: float() argument must be ... not 'NoneType'` and took the whole plan's answer with
    it, where skipping the item leaves the comparison to the items that do carry a measurement.
    """

    if not _has_path(value, path):
        return False
    found = _path(value, path)
    if isinstance(found, bool) or found is None:
        return False
    try:
        float(found)
    except (TypeError, ValueError):
        return False
    return True


def _as_flag(value: Any) -> bool:
    """A boolean the planner may have written as a word."""

    if isinstance(value, str):
        return value.strip().casefold() not in {"", "false", "no", "0", "none"}
    return bool(value)


def _normalize_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Accept common planner aliases while keeping one canonical operator implementation."""

    args = dict(arguments)
    if name == "tsp_tw":
        # A planner writing "the order is given" reaches for whichever of these words it thinks in.
        for alias in ("preserve_order", "keep_order", "in_order", "sequential", "ordered"):
            if alias in args:
                args.setdefault("fixed_order", args.pop(alias))
        if "fixed_order" in args:
            args["fixed_order"] = _as_flag(args["fixed_order"])

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

    if name == "select_by_index":
        items = next(
            (
                args[key]
                for key in ("items", "list", "candidates", "values", "places", "routes")
                if key in args
            ),
            None,
        )
        if items is None:
            raise ValueError("select_by_index requires items")
        # Only aliases that mean the same thing. `rank`, `k` and `position` read as 1-based to
        # about as many writers as read them 0-based, and an ordinal question answered one place
        # off is indistinguishable from one answered wrongly.
        index = next((args[key] for key in ("index", "i") if key in args), None)
        if index is None:
            raise ValueError(
                "select_by_index requires a 0-based index; the second item is index 1"
            )
        normalized: dict[str, Any] = {"items": items, "index": index}
        for optional in ("key", "descending"):
            if optional in args:
                normalized[optional] = args[optional]
        return normalized

    if name == "sum_amounts":
        amounts = next(
            (
                args[key]
                for key in (
                    "amounts",
                    "items",
                    "values",
                    "inputs",
                    "distances",
                    "legs",
                    "routes",
                    "numbers",
                )
                if key in args
            ),
            None,
        )
        if amounts is None:
            raise ValueError("sum_amounts requires a list of amounts")
        return {"amounts": amounts, **({"key": args["key"]} if "key" in args else {})}

    if name == "difference":
        optional_key = {"key": args["key"]} if "key" in args else {}
        for first, second in (
            ("minuend", "subtrahend"),
            ("a", "b"),
            ("first", "second"),
            ("left", "right"),
            ("value_a", "value_b"),
            ("amount1", "amount2"),
            ("x", "y"),
        ):
            if first in args and second in args:
                return {
                    "minuend": args[first],
                    "subtrahend": args[second],
                    **optional_key,
                }
        pair = next(
            (args[key] for key in ("values", "amounts", "items", "inputs") if key in args),
            None,
        )
        if isinstance(pair, list) and len(pair) == 2:
            return {"minuend": pair[0], "subtrahend": pair[1], **optional_key}
        raise ValueError(
            "difference requires two measurements, as minuend/subtrahend or a two-element list"
        )

    if name in {"select_min", "select_max"}:
        if "items" in args:
            return {
                "items": args["items"],
                "key": args.get("key") or _ranking_key(args["items"]),
            }
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

    if name in {"extract_distance", "extract_duration"}:
        # The plural is the same slot: a plan that measures every leg of a trip writes `routes`
        # about as often as `route`, and `$segments.routes` is a list either way. Refusing the
        # spelling cost a `trip_total_distance` question its answer on the v7a draw.
        if "route" not in args:
            for alias in ("routes", "legs", "route_list"):
                if alias in args:
                    args["route"] = args.pop(alias)
                    break
        return args

    if name == "nearest":
        # `center` is what `nearby_places` calls the point it measures from, so a planner that
        # retrieves with `nearby_places(center=...)` and ranks with `nearest(center=...)` in the
        # next node has written one vocabulary, not two. The implementation calls it `anchor` and
        # nothing else, so the plan died on a spelling: two `nearby_subtype_kth` questions lost to
        # "missing arguments: anchor" across the two 300-row draws. Only spellings that mean the
        # same point -- no ordinal, no candidate list.
        if "anchor" not in args:
            for alias in ("center", "origin", "from_place", "reference"):
                if alias in args:
                    args["anchor"] = args.pop(alias)
                    break
        return args

    if name == "sum_route_metrics":
        routes = next(
            (args[key] for key in ("routes", "inputs", "legs") if key in args),
            None,
        )
        # `$legs` is the whole `distance_matrix` node, which carries its routes under "routes"
        # alongside the matrix it also emits. Referencing the node rather than the list is the
        # same shape leniency the place arguments get, and refusing it here lost the leg sum and
        # every clock built on it.
        routes = _route_list(routes)
        if not isinstance(routes, list):
            raise ValueError("sum_route_metrics requires routes, inputs, or legs")
        if routes and all(isinstance(route, dict) for route in routes):
            return {"routes": [route for route in routes if route.get("status") != "error"]}
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


# Where an operator's output keeps its places. `match_options` reports under `retrieved_places`,
# `nearest` under `ranked`; a planner passes the node, not the field, and the whole record then
# read as one unresolvable candidate.
_PLACE_COLLECTION_KEYS = (
    "ranked",
    "retrieved_places",
    "places",
    "matches",
    "candidates",
    "results",
)


def _place_collection(value: Any) -> list[Any]:
    """The candidates in hand, whether a list was passed or the node that carries one."""

    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "latitude" not in value:
        for key in _PLACE_COLLECTION_KEYS:
            found = value.get(key)
            if isinstance(found, list) and found:
                return found
    return [value]


def _as_place_list(
    value: Any, *, keep_unresolved: bool = False
) -> list[tuple[int, dict[str, Any]]]:
    """Normalize a candidate collection, keeping each candidate's original index."""

    values = _place_collection(value)
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


def _extract_metric(route: Any, metric: str) -> dict[str, float] | list[dict[str, float]]:
    """Read one metric off a route, off the node that carries routes, or off each of a list."""

    route = _route_list(route)
    if isinstance(route, list):
        return [_single_metric(item, metric) for item in route]
    return _single_metric(route, metric)


def _single_metric(route: Any, metric: str) -> dict[str, float]:
    if not isinstance(route, dict) or metric not in route:
        raise ValueError(
            f"extract expected a route carrying {metric}, got {type(route).__name__}. "
            "A `$node` reference that never resolved measures nothing, so this fails "
            "instead of reporting zero."
        )
    return {metric: float(route[metric])}


def _route_list(value: Any) -> Any:
    """The routes a planner meant, whether it referenced them or the node that carries them."""

    if isinstance(value, dict):
        for key in ("routes", "legs", "inputs"):
            found = value.get(key)
            if isinstance(found, list):
                return found
    return value


def _duration_value(value: Any) -> float:
    """Read a duration in seconds from a number or from whatever carries one.

    A planner hands `calculate_start_time` the route it just measured rather than the route's
    duration, and `float({...})` raised a TypeError that took the whole clock with it. The keys
    are the ones the operators themselves emit — a tour's `total_cost`, a route's `duration_s` —
    so this reads a measurement the run already made, never one it did not.
    """

    if isinstance(value, dict):
        for key in ("duration_s", "total_duration_s", "total_cost", "travel_duration_s",
                    "duration", "seconds", "value", "amount"):
            if key in value:
                return _duration_value(value[key])
        raise ValueError("PlaceNotFoundError: no duration in the value supplied")
    if isinstance(value, list) and len(value) == 1:
        return _duration_value(value[0])
    return float(value)


_DISTANCE_KEYS = ("distance_m", "distance", "meters", "distance_km", "km")
_DURATION_KEYS = ("duration_s", "duration", "seconds")
_GENERIC_AMOUNT_KEYS = ("value", "amount", "total")


def _amount_collection(value: Any) -> list[Any]:
    """The measurements a planner meant, however it wrapped them."""

    if isinstance(value, dict):
        for key in (
            "amounts",
            "items",
            "values",
            "routes",
            "legs",
            "inputs",
            "distances",
            "numbers",
            "list",
        ):
            found = value.get(key)
            if isinstance(found, list):
                return found
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def _amount_number(value: Any, key: str | None = None, *, where: str) -> float:
    """One measurement, read off a number or off whatever record carries it."""

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"{where} received {value!r}, which is text rather than a measurement. A `$node` "
                "reference that never resolved adds up to nothing, so this fails instead of "
                "counting it as zero."
            ) from None
    if isinstance(value, dict):
        if key is not None:
            if key not in value:
                raise ValueError(f"{where}: no {key!r} among {sorted(value)[:8]}")
            return _amount_number(value[key], where=where)
        for candidate in ("distance_m", "duration_s", *_GENERIC_AMOUNT_KEYS):
            if candidate in value:
                return _amount_number(value[candidate], where=where)
        for candidate in ("distance_km", "km"):
            if candidate in value:
                return _amount_number(value[candidate], where=where) * 1000
        raise ValueError(f"{where}: no measurement among {sorted(value)[:8]}")
    if value is None or isinstance(value, bool):
        raise ValueError(f"{where} received {value!r}, which is not a measurement")
    return float(value)


def _amount_kind(key: str | None, values: list[Any]) -> str:
    """Whether a set of amounts is metres, seconds, or plain numbers.

    A route record carries both, and distance wins there because `_amount_number` reads it first.
    """

    if key is not None:
        if key in _DISTANCE_KEYS:
            return "distance"
        if key in _DURATION_KEYS:
            return "duration"
        return "plain"
    for item in values:
        if isinstance(item, dict):
            if any(name in item for name in _DISTANCE_KEYS):
                return "distance"
            if any(name in item for name in _DURATION_KEYS):
                return "duration"
    return "plain"


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


def _ranking_key(items: Any) -> str:
    """The key a ranking meant when the planner spelled none.

    `select_min`/`select_max` fall back to `"value"`, and a list of `haversine_distance` records
    carries no `value` -- so a graph that ranked three measured distances without naming the key
    raised "No item contains comparable key: value" and lost the question. Forty-five calls in
    `logs/` are written that way. The measurement the records actually carry is the one they
    meant; there is nothing else in them to rank by, and refusing the plan instead only moved the
    same loss earlier.
    """

    if not isinstance(items, list):
        return "value"
    records = [item for item in items if isinstance(item, dict)]
    if not records:
        return "value"
    for candidate in ("distance_m", "duration_s", "value", "amount", "total", "rating"):
        if all(_has_comparable(item, candidate) for item in records):
            return candidate
    return "value"


def _comparison_value_path(items: list[dict[str, Any]]) -> str:
    values = [item.get("value") for item in items]
    if values and all(isinstance(value, dict) and "distance_m" in value for value in values):
        return "value.distance_m"
    if values and all(isinstance(value, dict) and "duration_s" in value for value in values):
        return "value.duration_s"
    return "value"
