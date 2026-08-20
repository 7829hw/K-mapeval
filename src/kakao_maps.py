"""Kakao Local + Kakao Mobility behind Google Maps' client surface.

This module is the whole API swap. It replaces `Spatial-Agent/src/tools/google_maps.py`
(`GoogleMapsClient`) and the Google-backed HTTP calls in `MapEval-API/FormattedTools.py`
with Kakao, and it is deliberately the *only* hand-written file in this port: every method
below keeps the name, the argument names, the argument types and the return shape of its
Google counterpart, so the ~9,900 vendored lines above it call it unchanged.

Both architectures read this one client, so an accuracy difference between them cannot come
from having been shown different evidence.

Where Kakao cannot answer what Google answered, the method says so rather than inventing a
value. Those cases are enumerated in `docs/UPSTREAM_MAPPING.md`; the short list is that
Kakao Local publishes no rating, price level or opening hours, and Kakao Mobility routes
cars only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LOCAL_BASE_URL = "https://dapi.kakao.com/v2/local"
MOBILITY_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
MOBILITY_WAYPOINT_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/waypoints/directions"

# Kakao Local's own category group codes. A retrieval that can name one of these is served by
# the category endpoint, which is what Google's `type=` parameter selects.
KAKAO_CATEGORY_CODES = frozenset(
    {
        "MT1", "CS2", "PS3", "SC4", "AC5", "PK6", "OL7", "SW8", "BK9",
        "CT1", "AG2", "PO3", "AT4", "AD5", "FD6", "CE7", "HP8", "PM9",
    }
)

# Google place types (what a vendored planner emits, since the upstream prompts were written
# against Google) and the Korean nouns a Korean question asks by, onto Kakao's vocabulary.
# `code` selects Kakao's category endpoint; `keyword` is the query used when no code covers
# the type, or as the narrowing term inside a coarse code.
TYPE_VOCABULARY: dict[str, dict[str, str | None]] = {
    "restaurant": {"code": "FD6", "keyword": None},
    "food": {"code": "FD6", "keyword": None},
    "meal_takeaway": {"code": "FD6", "keyword": None},
    "meal_delivery": {"code": "FD6", "keyword": None},
    "cafe": {"code": "CE7", "keyword": None},
    "bakery": {"code": "CE7", "keyword": "베이커리"},
    "convenience_store": {"code": "CS2", "keyword": None},
    "supermarket": {"code": "MT1", "keyword": None},
    "grocery_or_supermarket": {"code": "MT1", "keyword": None},
    "shopping_mall": {"code": "MT1", "keyword": "쇼핑몰"},
    "department_store": {"code": "MT1", "keyword": "백화점"},
    "store": {"code": None, "keyword": "상점"},
    "hospital": {"code": "HP8", "keyword": None},
    "doctor": {"code": "HP8", "keyword": None},
    "dentist": {"code": "HP8", "keyword": "치과"},
    "pharmacy": {"code": "PM9", "keyword": None},
    "drugstore": {"code": "PM9", "keyword": None},
    "bank": {"code": "BK9", "keyword": None},
    "atm": {"code": "BK9", "keyword": "ATM"},
    "school": {"code": "SC4", "keyword": None},
    "primary_school": {"code": "SC4", "keyword": "초등학교"},
    "secondary_school": {"code": "SC4", "keyword": "중학교"},
    "university": {"code": "SC4", "keyword": "대학교"},
    "parking": {"code": "PK6", "keyword": None},
    "gas_station": {"code": "OL7", "keyword": None},
    "subway_station": {"code": "SW8", "keyword": None},
    "transit_station": {"code": "SW8", "keyword": None},
    "train_station": {"code": None, "keyword": "기차역"},
    "bus_station": {"code": None, "keyword": "버스정류장"},
    "lodging": {"code": "AD5", "keyword": None},
    "hotel": {"code": "AD5", "keyword": "호텔"},
    "tourist_attraction": {"code": "AT4", "keyword": None},
    "museum": {"code": "CT1", "keyword": "박물관"},
    "art_gallery": {"code": "CT1", "keyword": "미술관"},
    "movie_theater": {"code": "CT1", "keyword": "영화관"},
    "library": {"code": "CT1", "keyword": "도서관"},
    "park": {"code": "AT4", "keyword": "공원"},
    "police": {"code": "PO3", "keyword": None},
    "post_office": {"code": "PO3", "keyword": "우체국"},
    "gym": {"code": None, "keyword": "헬스장"},
    "real_estate_agency": {"code": "AG2", "keyword": None},
    "car_repair": {"code": None, "keyword": "자동차정비"},
    "beauty_salon": {"code": None, "keyword": "미용실"},
    "laundry": {"code": None, "keyword": "세탁소"},
    # The Korean nouns a Korean question asks by, so a planner that copies the question's own
    # word reaches the same retrieval a Google type would have.
    "음식점": {"code": "FD6", "keyword": None},
    "식당": {"code": "FD6", "keyword": None},
    "카페": {"code": "CE7", "keyword": None},
    "편의점": {"code": "CS2", "keyword": None},
    "마트": {"code": "MT1", "keyword": None},
    "대형마트": {"code": "MT1", "keyword": None},
    "슈퍼마켓": {"code": "MT1", "keyword": None},
    "병원": {"code": "HP8", "keyword": None},
    "약국": {"code": "PM9", "keyword": None},
    "은행": {"code": "BK9", "keyword": None},
    "학교": {"code": "SC4", "keyword": None},
    "초등학교": {"code": "SC4", "keyword": "초등학교"},
    "주차장": {"code": "PK6", "keyword": None},
    "주유소": {"code": "OL7", "keyword": None},
    "지하철역": {"code": "SW8", "keyword": None},
    "숙박": {"code": "AD5", "keyword": None},
    "관광명소": {"code": "AT4", "keyword": None},
    "공원": {"code": "AT4", "keyword": "공원"},
    "문화시설": {"code": "CT1", "keyword": None},
    "공공기관": {"code": "PO3", "keyword": None},
    "우체국": {"code": "PO3", "keyword": "우체국"},
    "치안센터": {"code": "PO3", "keyword": "치안센터"},
    "파출소": {"code": "PO3", "keyword": "파출소"},
}

# Kakao Local caps a page at 15 results and a radius at 20 km; Kakao Mobility caps waypoints
# at 30. Asking for more is an error from Kakao, not a larger answer.
MAX_PAGE_SIZE = 15
MAX_RADIUS_M = 20_000
MAX_WAYPOINTS = 30

# Korea observes KST year-round and has had no DST since 1988, so the timezone Google served
# from an API is a constant here. See `get_timezone`.
KST_OFFSET_SECONDS = 9 * 3600


class KakaoMapsClient:
    """Kakao Map API, presented with Google Maps' client surface.

    Every public method mirrors `GoogleMapsClient` in
    `ecerybao/Spatial-Agent@6876bba:src/tools/google_maps.py`: same name, same arguments,
    same return shape, same "return None / [] and log" failure convention. The vendored
    agents call it without knowing which provider is underneath.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
        cache_path: str | None = None,
        cache_ttl_seconds: int | None = None,
        search_center: tuple[float, float] | None = None,
        search_radius_m: int | None = None,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key or os.getenv("KAKAO_REST_API_KEY")
        if not self.api_key:
            raise ValueError("Kakao Map API key is required")

        if timeout is None:
            timeout = float(os.getenv("KAKAO_TIMEOUT_SECONDS", "30"))
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

        # The region prior. Korean POI names repeat across cities, so a nationwide keyword
        # search resolves a bare brand name to whichever branch has the shortest name,
        # anywhere in the country. This biases the *first* query only; a name with no match
        # in the region still falls back to the nationwide search below, so the prior can
        # never hide a place. It is deployment configuration applied to whichever agent
        # queries, never gold metadata.
        self._search_center = search_center if search_center is not None else _env_center()
        radius = search_radius_m if search_radius_m is not None else int(
            os.getenv("KAKAO_SEARCH_RADIUS_M", "20000") or 0
        )
        self._search_radius_m = max(0, min(radius, MAX_RADIUS_M))
        if self._search_radius_m == 0:
            self._search_center = None

        ttl = cache_ttl_seconds if cache_ttl_seconds is not None else int(
            os.getenv("KAKAO_CACHE_TTL_SECONDS", "86400") or 0
        )
        # None means "take the deployment default"; "" means "no cache at all", which is
        # what the tests pass so an assertion about API calls is about API calls.
        if cache_path is None:
            cache_path = os.getenv("KAKAO_CACHE_DB_PATH", "data/kakao_cache.db")
        self._cache = _ResponseCache(cache_path, ttl_seconds=ttl)

        # Kakao Local has no place-details endpoint, so details are served from what the
        # searches already returned. `_places` is this client's memory of every place it has
        # handed out, and the cache persists it across runs.
        self._places: dict[str, dict[str, Any]] = {}

        self.api_call_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> KakaoMapsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._cache.close()
        if self._owns_client:
            self._client.close()

    def reset_counters(self) -> None:
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    # ------------------------------------------------------------------ geocoding

    def geocode(
        self,
        address: str,
        location_bias: tuple[float, float] | None = None,
        region: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a name or an address to coordinates.

        Returns `{lat, lng, formatted_address, place_id}` or None, exactly as the Google
        client did, including its `location_bias` selection and its distance thresholds.

        `region` was Google's ccTLD country bias. Kakao Local serves Korea only, so there is
        no country left to choose between: the argument is accepted and ignored.
        """

        if not address or not str(address).strip():
            return None
        address = str(address).strip()
        if region:
            logger.debug("Ignoring region bias %r: Kakao Local covers Korea only", region)

        try:
            candidates = self._geocode_candidates(address, location_bias)
            if not candidates:
                return None

            if not location_bias:
                return _as_geocode_result(candidates[0])

            anchor_lat, anchor_lng = location_bias
            logger.debug(
                "Using location bias: (%s, %s), found %d candidates",
                anchor_lat,
                anchor_lng,
                len(candidates),
            )

            min_distance = float("inf")
            closest = None
            for candidate in candidates:
                distance = (
                    haversine(anchor_lat, anchor_lng, candidate["lat"], candidate["lng"]) / 1000
                )
                if distance < min_distance:
                    min_distance = distance
                    closest = candidate

            if closest is None:
                return None

            logger.debug("Selected closest result: %.1fkm from anchor", min_distance)

            # Upstream's three thresholds, unchanged. Case 1 was Google's Plus Code, which
            # Kakao does not index; the branch is kept so the ported logic still reads
            # against its original, and a Plus Code simply never reaches it.
            if re.match(r"^[A-Z0-9]{4,}\+[A-Z0-9]{2,}", address):
                max_threshold = 200
            elif len(str(closest["formatted_address"]).split(",")) <= 2:
                max_threshold = 200
            else:
                max_threshold = 100

            if min_distance > max_threshold:
                logger.warning(
                    "Geocoding result too far from anchor: %.1fkm > %dkm. Rejecting: %s",
                    min_distance,
                    max_threshold,
                    closest["formatted_address"],
                )
                return None

            return _as_geocode_result(closest)
        except KakaoError:
            raise
        except Exception as e:  # noqa: BLE001 - upstream's convention: log and return None
            logger.error(f"Geocoding error for {address}: {e}")
            return None

    def reverse_geocode(self, lat: float, lng: float) -> str | None:
        """Coordinates to a formatted address, or None."""

        try:
            data = self._get_local(
                "/geo/coord2address.json",
                {"x": lng, "y": lat, "input_coord": "WGS84"},
            )
            for document in data.get("documents") or []:
                road = document.get("road_address") or {}
                jibun = document.get("address") or {}
                formatted = road.get("address_name") or jibun.get("address_name")
                if formatted:
                    return str(formatted)
            return None
        except KakaoError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Reverse geocoding error for {lat}, {lng}: {e}")
            return None

    # ------------------------------------------------------------------ retrieval

    def nearby_search(
        self,
        location: tuple[float, float],
        radius: int = 5000,
        place_type: str = None,
        keyword: str = None,
        min_rating: float = None,
        open_now: bool = False,
    ) -> list[dict[str, Any]]:
        """Places around a coordinate, as a list of Google-shaped place dicts.

        `min_rating` and `open_now` are accepted and ignored: Kakao Local publishes neither
        a rating nor opening hours, and a filter over a field the evidence does not carry is
        not a filter but a delete — applying `min_rating` against an all-zero `rating` would
        empty every candidate list. A source that publishes no ratings is not a source in
        which every place is unrated.
        """

        try:
            lat, lng = float(location[0]), float(location[1])
        except (TypeError, ValueError, IndexError):
            logger.error(f"Nearby search error: invalid location {location!r}")
            return []

        if min_rating:
            logger.debug("Ignoring min_rating=%s: Kakao Local publishes no ratings", min_rating)
        if open_now:
            logger.debug("Ignoring open_now: Kakao Local publishes no opening hours")

        try:
            documents = self._nearby_documents(lat, lng, radius, place_type, keyword)
            places = [_as_nearby_place(document) for document in documents]

            # Upstream ranked by (rating, user_ratings_total). Kakao carries neither, so the
            # key is constant and Python's stable sort preserves Kakao's own distance order —
            # which is the ranking a nearby search should return anyway. The line is kept so
            # the ported code still reads against its original.
            places.sort(key=lambda x: (x["rating"], x["user_ratings_total"]), reverse=True)
            return places
        except KakaoError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Nearby search error: {e}")
            return []

    def text_search(
        self,
        query: str,
        location: tuple[float, float] = None,
        radius: int = 50000,
    ) -> list[dict[str, Any]]:
        """Keyword search, optionally centered on a coordinate."""

        try:
            params: dict[str, Any] = {"query": query, "size": MAX_PAGE_SIZE}
            if location:
                params.update(
                    {
                        "x": float(location[1]),
                        "y": float(location[0]),
                        "radius": max(1, min(int(radius), MAX_RADIUS_M)),
                        "sort": "distance",
                    }
                )
            documents = self._keyword_documents(params)
            return [_as_text_search_place(document) for document in documents]
        except KakaoError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Text search error for {query}: {e}")
            return []

    def get_place_details(self, place_id: str) -> dict[str, Any] | None:
        """Details for a place id this client has handed out.

        Kakao Local has no counterpart to Google's Place Details endpoint: a search response
        already carries everything Kakao knows about a place. So details are served from the
        place table every search writes to, persisted across runs by the cache. An id this
        client never issued is unknown, which is the same answer Google gave for a malformed
        one.

        `rating`, `user_ratings_total`, `price_level` and `opening_hours` are None rather
        than absent: the keys stay so the vendored formatters keep working, and None says
        Kakao does not publish the field instead of asserting a value.
        """

        try:
            place = self._lookup_place(str(place_id))
            if place is None:
                return None
            return {
                "name": place["name"],
                "place_id": place["place_id"],
                "lat": place["lat"],
                "lng": place["lng"],
                "formatted_address": place.get("formatted_address", ""),
                "rating": None,
                "user_ratings_total": None,
                "price_level": None,
                "opening_hours": None,
                "formatted_phone_number": place.get("phone") or None,
                "website": place.get("place_url") or None,
                "types": place.get("types", []),
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Place details error for {place_id}: {e}")
            return None

    # ------------------------------------------------------------------ routing

    def get_directions(
        self,
        origin: str,
        destination: str,
        mode: str = "driving",
        waypoints: list[str] = None,
        optimize_waypoints: bool = False,
        alternatives: bool = False,
    ) -> dict[str, Any] | None:
        """Driving routes between two places, as `{routes: [...], primary: {...}}`.

        `origin` / `destination` / `waypoints` accept what the vendored callers pass: a
        "lat,lng" literal, a place id this client issued, or a name to look up.

        Kakao Mobility routes cars only — there is no public walking, cycling or transit
        directions API — so any other `mode` returns None rather than answering a driving
        question in its place. `optimize_waypoints` is accepted and ignored: Kakao visits
        waypoints in the order given.
        """

        try:
            logger.info(
                "[directions] Kakao API: %s -> %s (mode=%s, waypoints=%s)",
                origin,
                destination,
                mode,
                waypoints,
            )

            if str(mode).lower() not in {"driving", "car"}:
                logger.warning(
                    "Kakao Mobility supports driving only; refusing to answer a %s route with one",
                    mode,
                )
                return None

            origin_place = self._dereference(origin)
            destination_place = self._dereference(destination)
            if origin_place is None or destination_place is None:
                logger.warning(f"Directions could not resolve {origin!r} -> {destination!r}")
                return None

            waypoint_places = []
            for value in waypoints or []:
                place = self._dereference(value)
                if place is None:
                    logger.warning(f"Directions could not resolve waypoint {value!r}")
                    return None
                waypoint_places.append(place)
            if len(waypoint_places) > MAX_WAYPOINTS:
                logger.warning("Kakao Mobility supports at most %d waypoints", MAX_WAYPOINTS)
                return None
            if optimize_waypoints and waypoint_places:
                logger.debug("Ignoring optimize_waypoints: Kakao visits waypoints in order")

            data = self._route_payload(
                origin_place, destination_place, waypoint_places, alternatives=alternatives
            )
            if data is None:
                return None

            routes: list[dict[str, Any]] = []
            for route in data.get("routes") or []:
                if route.get("result_code") != 0:
                    logger.warning(
                        "Kakao route failed: %s", route.get("result_msg", "unknown reason")
                    )
                    continue
                summary = route.get("summary") or {}
                distance = int(summary.get("distance") or 0)
                duration = int(summary.get("duration") or 0)
                routes.append(
                    {
                        "distance": distance,
                        "duration": duration,
                        "distance_text": format_distance(distance),
                        "duration_text": format_duration(duration),
                        "start_address": origin_place.get("formatted_address")
                        or origin_place["name"],
                        "end_address": destination_place.get("formatted_address")
                        or destination_place["name"],
                        "steps": _route_steps(route),
                        # Kakao returns route vertices, not an encoded polyline.
                        "overview_polyline": None,
                        "summary": _route_summary_name(route, destination_place),
                    }
                )

            if not routes:
                return None

            return {"routes": routes, "primary": routes[0]}
        except KakaoError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Directions error from {origin} to {destination}: {e}")
            return None

    def get_distance_matrix(
        self,
        origins: list[str],
        destinations: list[str],
        mode: str = "driving",
        departure_time: str | None = None,
        format_output: bool = False,
    ) -> Any | None:
        """A driving duration/distance matrix, in Google's Distance Matrix shape.

        Kakao Mobility has no single call that answers a matrix, so this is one route
        request per off-diagonal pair, cached like every other call. The diagonal is filled
        with zero *locally*: Kakao refuses a leg whose endpoints are within 5 m of each other
        ("출발지와 도착지가 5 m 이내로 설정된 경우 경로를 탐색할 수 없음"), and a trip matrix asks
        for its own diagonal on every run. It is the only leg that may be filled — a missing
        off-diagonal leg is reported as a failed element, never as a free hop.

        `departure_time` is accepted and ignored: Kakao's route already reflects current
        traffic and takes no departure timestamp.
        """

        try:
            if str(mode).lower() not in {"driving", "car"}:
                logger.warning("Kakao Mobility supports driving only; no %s matrix", mode)
                return None
            if departure_time:
                logger.debug(
                    "Ignoring departure_time=%r: Kakao routes against current traffic",
                    departure_time,
                )

            origin_places = [self._dereference(value) for value in origins]
            destination_places = [self._dereference(value) for value in destinations]

            rows: list[dict[str, Any]] = []
            for origin_place in origin_places:
                elements: list[dict[str, Any]] = []
                for destination_place in destination_places:
                    elements.append(self._matrix_element(origin_place, destination_place))
                rows.append({"elements": elements})

            result = {"status": "OK", "rows": rows}

            if not format_output:
                return result

            matrix = []
            for row in result["rows"]:
                matrix_row = []
                for element in row["elements"]:
                    if element["status"] == "OK":
                        matrix_row.append(
                            {
                                "distance": element["distance"]["value"],
                                "duration": element["duration"]["value"],
                                "distance_text": element["distance"]["text"],
                                "duration_text": element["duration"]["text"],
                            }
                        )
                    else:
                        matrix_row.append(None)
                matrix.append(matrix_row)
            return matrix
        except KakaoError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Distance matrix error: {e}")
            return None

    def get_timezone(self, lat: float, lng: float, timestamp: int) -> dict[str, Any] | None:
        """The timezone at a coordinate, in Google's Time Zone API shape.

        Kakao publishes no timezone endpoint, and it does not need to: its coverage is Korea,
        which is Asia/Seoul at UTC+9 and has observed no daylight saving since 1988. So this
        is answered locally at zero API cost, which is a fact about the coverage area rather
        than a value invented to fill a gap.
        """

        del timestamp  # KST has no DST, so the answer does not depend on the instant
        if not _within_korea(lat, lng):
            logger.warning("Timezone requested outside Kakao's coverage: (%s, %s)", lat, lng)
        return {
            "timeZoneId": "Asia/Seoul",
            "timeZoneName": "Korean Standard Time",
            "rawOffset": KST_OFFSET_SECONDS,
            "dstOffset": 0,
        }

    # ------------------------------------------------------------------ internals

    def _geocode_candidates(
        self, address: str, location_bias: tuple[float, float] | None
    ) -> list[dict[str, Any]]:
        """Candidates for a geocode, address index first and then the keyword index.

        Kakao splits what Google's Geocoding API answered in one place: `/search/address.json`
        indexes addresses and `/search/keyword.json` indexes POI names. A Korean question
        names a POI far more often than an address, so a name that the address index does not
        carry has to reach the keyword index or it cannot resolve at all.
        """

        documents = self._local_documents(
            "/search/address.json", {"query": address, "size": MAX_PAGE_SIZE}
        )
        candidates = [_as_geocode_candidate(document) for document in documents]
        if candidates:
            return candidates

        params: dict[str, Any] = {"query": address, "size": MAX_PAGE_SIZE}
        if location_bias:
            # A bias the caller gave outranks the deployment-wide region prior: it is a
            # statement about this lookup, and the prior is a statement about the run.
            params.update(
                {
                    "x": float(location_bias[1]),
                    "y": float(location_bias[0]),
                    "radius": MAX_RADIUS_M,
                }
            )
        documents = self._keyword_documents(params)
        return [_as_geocode_candidate(document) for document in documents]

    def _keyword_documents(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Keyword search, region prior first and the nationwide search as the fallback."""

        biased: list[dict[str, Any]] = []
        if self._search_center is not None and "x" not in params:
            lat, lng = self._search_center
            biased = self._local_documents(
                "/search/keyword.json",
                {**params, "x": lng, "y": lat, "radius": self._search_radius_m},
            )
        if biased:
            return biased
        return self._local_documents("/search/keyword.json", params)

    def _nearby_documents(
        self,
        lat: float,
        lng: float,
        radius: int,
        place_type: str | None,
        keyword: str | None,
    ) -> list[dict[str, Any]]:
        """One neighbourhood retrieval, through whichever Kakao index names the request.

        Google took `type` and `keyword` side by side. Kakao takes a category group code on
        one endpoint and a query on another, so the requested type is translated into
        whichever of the two can express it, and a type Kakao has no code for is asked for by
        name.
        """

        normalized_radius = max(1, min(int(radius or MAX_RADIUS_M), MAX_RADIUS_M))
        base = {
            "x": lng,
            "y": lat,
            "radius": normalized_radius,
            "sort": "distance",
            "size": MAX_PAGE_SIZE,
        }

        code, type_keyword = _kakao_type(place_type)
        query = keyword or type_keyword or (place_type if code is None else None)

        if code and not query:
            return self._local_documents(
                "/search/category.json", {**base, "category_group_code": code}
            )
        if code and query:
            documents = self._local_documents(
                "/search/keyword.json", {**base, "query": query, "category_group_code": code}
            )
            if documents:
                return documents
            # The narrowing term found nothing inside the code. An empty result from a
            # vocabulary gap is not evidence that the neighbourhood holds no such place, so
            # the coarser retrieval answers instead.
            return self._local_documents(
                "/search/category.json", {**base, "category_group_code": code}
            )
        if query:
            return self._local_documents(
                "/search/keyword.json", {**base, "query": query}
            )
        return []

    def _local_documents(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        data = self._get_local(path, params)
        documents = [d for d in (data.get("documents") or []) if isinstance(d, dict)]
        self._remember(documents)
        return documents

    def _remember(self, documents: list[dict[str, Any]]) -> None:
        """Record every place handed out, so `get_place_details` can answer for it later."""

        for document in documents:
            place = _as_place_record(document)
            if place is None:
                continue
            self._places[place["place_id"]] = place
            self._cache.set_place(place)

    def _lookup_place(self, place_id: str) -> dict[str, Any] | None:
        place = self._places.get(place_id)
        if place is not None:
            return place
        place = self._cache.get_place(place_id)
        if place is not None:
            self._places[place_id] = place
        return place

    def _dereference(self, value: Any) -> dict[str, Any] | None:
        """Turn what a caller is holding into a place record.

        A "lat,lng" literal, a place id this client issued, a place dict a previous call
        returned, or — as Google's own directions endpoint accepted — a name to look up.
        """

        if value is None:
            return None
        if isinstance(value, dict):
            if value.get("lat") is not None and value.get("lng") is not None:
                return {
                    "place_id": str(
                        value.get("place_id") or f"{value['lat']},{value['lng']}"
                    ),
                    "name": str(value.get("name") or value.get("formatted_address") or ""),
                    "formatted_address": str(value.get("formatted_address") or ""),
                    "lat": float(value["lat"]),
                    "lng": float(value["lng"]),
                    "phone": str(value.get("phone") or ""),
                    "place_url": str(value.get("place_url") or ""),
                    "types": list(value.get("types") or []),
                }
            return None

        text = str(value).strip()
        if not text:
            return None

        coordinates = parse_coordinate_literal(text)
        if coordinates is not None:
            lat, lng = coordinates
            return {
                "place_id": f"{lat},{lng}",
                "name": text,
                "formatted_address": "",
                "lat": lat,
                "lng": lng,
                "phone": "",
                "place_url": "",
                "types": [],
            }

        place = self._lookup_place(text)
        if place is not None:
            return place

        resolved = self.geocode(text)
        if resolved is None:
            return None
        return self._lookup_place(str(resolved["place_id"])) or {
            "place_id": str(resolved["place_id"]),
            "name": text,
            "formatted_address": str(resolved.get("formatted_address") or ""),
            "lat": float(resolved["lat"]),
            "lng": float(resolved["lng"]),
            "phone": "",
            "place_url": "",
            "types": [],
        }

    def _route_payload(
        self,
        origin: dict[str, Any],
        destination: dict[str, Any],
        waypoints: list[dict[str, Any]],
        *,
        alternatives: bool,
    ) -> dict[str, Any] | None:
        priority = os.getenv("KAKAO_ROUTE_PRIORITY", "RECOMMEND").upper()
        if priority not in {"RECOMMEND", "TIME", "DISTANCE"}:
            priority = "RECOMMEND"

        if waypoints:
            return self._request(
                MOBILITY_WAYPOINT_DIRECTIONS_URL,
                method="POST",
                json_body={
                    "origin": _mobility_point(origin),
                    "destination": _mobility_point(destination),
                    "waypoints": [_mobility_point(place) for place in waypoints],
                    "priority": priority,
                    "summary": False,
                },
            )
        return self._request(
            MOBILITY_DIRECTIONS_URL,
            params={
                "origin": f"{origin['lng']},{origin['lat']}",
                "destination": f"{destination['lng']},{destination['lat']}",
                "priority": priority,
                "alternatives": str(bool(alternatives)).lower(),
                "summary": "false",
            },
        )

    def _matrix_element(
        self, origin: dict[str, Any] | None, destination: dict[str, Any] | None
    ) -> dict[str, Any]:
        if origin is None or destination is None:
            return {"status": "NOT_FOUND"}

        if _same_place(origin, destination):
            # Kakao refuses a leg shorter than 5 m, and a matrix asks for its own diagonal.
            return {
                "status": "OK",
                "distance": {"value": 0, "text": format_distance(0)},
                "duration": {"value": 0, "text": format_duration(0)},
            }

        data = self._route_payload(origin, destination, [], alternatives=False)
        route = None
        for candidate in (data or {}).get("routes") or []:
            if candidate.get("result_code") == 0:
                route = candidate
                break
        if route is None:
            return {"status": "ZERO_RESULTS"}

        summary = route.get("summary") or {}
        distance = int(summary.get("distance") or 0)
        duration = int(summary.get("duration") or 0)
        return {
            "status": "OK",
            "distance": {"value": distance, "text": format_distance(distance)},
            "duration": {"value": duration, "text": format_duration(duration)},
        }

    # ------------------------------------------------------------------ transport

    def _get_local(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._request(f"{LOCAL_BASE_URL}{path}", params=params) or {}

    def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        method: str = "GET",
    ) -> dict[str, Any] | None:
        cache_key = _cache_key(method, url, params, json_body)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hit_count += 1
            return cached
        self.cache_miss_count += 1

        self.api_call_count += 1
        headers = {
            "Authorization": f"KakaoAK {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        try:
            if method == "POST":
                response = self._client.post(url, headers=headers, params=params, json=json_body)
            else:
                response = self._client.get(url, headers=headers, params=params)
        except httpx.TimeoutException as exc:
            raise KakaoTimeoutError("Kakao API request timed out") from exc
        except httpx.HTTPError as exc:
            raise KakaoError(f"Kakao API request failed: {exc}") from exc

        if response.status_code in {401, 403}:
            raise KakaoAuthError("Kakao API rejected the configured REST API key")
        if response.status_code == 429:
            raise KakaoRateLimitError("Kakao API rate limit exceeded")
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise KakaoError("Kakao API returned a non-object response")
        self._cache.set(cache_key, payload)
        return payload


# ---------------------------------------------------------------------- errors


class KakaoError(RuntimeError):
    """A Kakao request could not be served."""


class KakaoAuthError(KakaoError):
    """The configured REST API key was rejected."""


class KakaoRateLimitError(KakaoError):
    """Kakao is rate limiting this key."""


class KakaoTimeoutError(KakaoError):
    """Kakao did not answer in time.

    Its own class because the harness retries a question whose provider could not answer
    *right now*, and never one whose provider answered that the place does not exist.
    """


# ---------------------------------------------------------------------- shaping


def _as_place_record(document: dict[str, Any]) -> dict[str, Any] | None:
    """One Kakao Local document as this client's internal place record."""

    road = document.get("road_address") or {}
    place_id = str(document.get("id") or document.get("address_name") or "").strip()
    name = str(
        document.get("place_name")
        or road.get("building_name")
        or document.get("address_name")
        or ""
    ).strip()
    if not place_id or not name:
        return None
    try:
        lat = float(document["y"])
        lng = float(document["x"])
    except (KeyError, TypeError, ValueError):
        return None
    category = str(document.get("category_name") or document.get("address_type") or "")
    return {
        "place_id": place_id,
        "name": name,
        "formatted_address": str(
            document.get("road_address_name") or document.get("address_name") or ""
        ),
        "lat": lat,
        "lng": lng,
        "phone": str(document.get("phone") or ""),
        "place_url": str(document.get("place_url") or ""),
        # Kakao files a category coarse-to-fine ("음식점 > 카페 > 커피전문점"), which is the same
        # information Google's `types` list carried, in path form.
        "types": [part.strip() for part in category.split(">") if part.strip()],
    }


def _as_geocode_candidate(document: dict[str, Any]) -> dict[str, Any]:
    place = _as_place_record(document)
    if place is None:
        return {"lat": 0.0, "lng": 0.0, "formatted_address": "", "place_id": ""}
    return {
        "lat": place["lat"],
        "lng": place["lng"],
        "formatted_address": place["formatted_address"] or place["name"],
        "place_id": place["place_id"],
    }


def _as_geocode_result(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "lat": candidate["lat"],
        "lng": candidate["lng"],
        "formatted_address": candidate["formatted_address"],
        "place_id": candidate["place_id"],
    }


def _as_nearby_place(document: dict[str, Any]) -> dict[str, Any]:
    place = _as_place_record(document) or {}
    return {
        "name": place.get("name", ""),
        "place_id": place.get("place_id", ""),
        "lat": place.get("lat", 0.0),
        "lng": place.get("lng", 0.0),
        # Kakao Local publishes no rating, no review count and no price level. Zero is what
        # upstream's own `place.get('rating', 0)` produced for a place without one, so the
        # sort below behaves identically; None would break it.
        "rating": 0,
        "user_ratings_total": 0,
        "price_level": None,
        "types": place.get("types", []),
        "vicinity": place.get("formatted_address", ""),
        "open_now": False,
        # Kakao's own distance from the search centre, in metres, when it served one.
        "distance": _int_or_none(document.get("distance")),
    }


def _as_text_search_place(document: dict[str, Any]) -> dict[str, Any]:
    place = _as_place_record(document) or {}
    return {
        "name": place.get("name", ""),
        "place_id": place.get("place_id", ""),
        "lat": place.get("lat", 0.0),
        "lng": place.get("lng", 0.0),
        "formatted_address": place.get("formatted_address", ""),
        "rating": None,
        "user_ratings_total": None,
        "types": place.get("types", []),
    }


def _route_steps(route: dict[str, Any]) -> list[dict[str, Any]]:
    """Kakao's turn-by-turn guides as Google-shaped steps.

    Upstream's `DirectionsTool` prints `html_instructions` for every step, so Kakao's
    `guidance` (already a full Korean instruction, e.g. "왕십리로 방면으로 좌회전") goes there
    verbatim. The road each guide belongs to is looked up through `road_index`, because a
    turn-count question is asked about a named road and the guidance alone does not always
    name it.
    """

    steps: list[dict[str, Any]] = []
    for section in route.get("sections") or []:
        roads = section.get("roads") or []
        for guide in section.get("guides") or []:
            road_index = guide.get("road_index", -1)
            road_name = ""
            if isinstance(road_index, int) and 0 <= road_index < len(roads):
                road_name = str(roads[road_index].get("name") or "")
            instruction = str(guide.get("guidance") or guide.get("name") or "")
            distance = int(guide.get("distance") or 0)
            duration = int(guide.get("duration") or 0)
            steps.append(
                {
                    "html_instructions": instruction,
                    "distance": format_distance(distance),
                    "duration": format_duration(duration),
                    "maneuver": _maneuver(guide),
                    "road_name": road_name,
                }
            )
    return steps


# Kakao's guide `type` codes for the turns a route-step question counts. Codes outside this
# map are left as None, which is what Google did for a step with no maneuver.
KAKAO_MANEUVERS = {
    1: "straight",
    2: "turn-left",
    3: "turn-right",
    4: "turn-slight-left",
    5: "turn-slight-right",
    6: "turn-sharp-left",
    7: "turn-sharp-right",
    8: "uturn",
    11: "turn-left",
    12: "turn-right",
    14: "uturn",
}


def _maneuver(guide: dict[str, Any]) -> str | None:
    code = guide.get("type")
    if isinstance(code, int):
        return KAKAO_MANEUVERS.get(code)
    return None


def _route_summary_name(route: dict[str, Any], destination: dict[str, Any]) -> str:
    """Google's `summary` was the route's main road. Kakao names roads per section."""

    for section in route.get("sections") or []:
        for road in section.get("roads") or []:
            name = str(road.get("name") or "").strip()
            if name:
                return name
    return str(destination.get("name") or "")


def _mobility_point(place: dict[str, Any]) -> dict[str, Any]:
    return {"name": place.get("name") or "", "x": place["lng"], "y": place["lat"]}


def _same_place(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("place_id") and a.get("place_id") == b.get("place_id"):
        return True
    return haversine(a["lat"], a["lng"], b["lat"], b["lng"]) < 5.0


# ---------------------------------------------------------------------- formatting


def format_distance(meters: int) -> str:
    """Google's distance phrasing, which the vendored formatters print verbatim."""

    meters = int(meters)
    if meters < 1000:
        return f"{meters} m"
    return f"{meters / 1000:.1f} km"


def format_duration(seconds: int) -> str:
    """Google's duration phrasing ("1 min", "14 mins", "1 hour 5 mins")."""

    seconds = int(seconds)
    minutes = max(0, round(seconds / 60))
    if minutes < 60:
        return "1 min" if minutes <= 1 else f"{minutes} mins"
    hours, remainder = divmod(minutes, 60)
    hour_text = "1 hour" if hours == 1 else f"{hours} hours"
    if remainder == 0:
        return hour_text
    return f"{hour_text} {remainder} mins"


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres, matching the vendored `utils.optimization.haversine`."""

    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


COORDINATE_LITERAL = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_coordinate_literal(text: str) -> tuple[float, float] | None:
    """"lat,lng" as a coordinate pair, or None when the text is a name.

    The vendored callers write a place they have already resolved as "lat,lng" (see
    `nodes.get_directions`), so this is how a resolved place travels between them.
    """

    match = COORDINATE_LITERAL.match(str(text))
    if not match:
        return None
    lat, lng = float(match.group(1)), float(match.group(2))
    if -90 <= lat <= 90 and -180 <= lng <= 180:
        return lat, lng
    return None


def _kakao_type(place_type: str | None) -> tuple[str | None, str | None]:
    """A requested place type as (Kakao category group code, narrowing keyword)."""

    if not place_type:
        return None, None
    text = str(place_type).strip()
    if text.upper() in KAKAO_CATEGORY_CODES:
        return text.upper(), None
    entry = TYPE_VOCABULARY.get(text) or TYPE_VOCABULARY.get(text.lower())
    if entry is None:
        return None, None
    return entry["code"], entry["keyword"]


def _within_korea(lat: float, lng: float) -> bool:
    return 33.0 <= lat <= 39.0 and 124.0 <= lng <= 132.0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_center() -> tuple[float, float] | None:
    raw = (os.getenv("KAKAO_SEARCH_CENTER") or "").strip()
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError('KAKAO_SEARCH_CENTER must be "latitude,longitude"')
    lat, lng = (float(part) for part in parts)
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("KAKAO_SEARCH_CENTER is outside valid coordinate ranges")
    return lat, lng


def _cache_key(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
) -> str:
    return json.dumps(
        {"method": method, "url": url, "params": params or {}, "body": json_body or {}},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


# ---------------------------------------------------------------------- cache


class _ResponseCache:
    """A SQLite cache of Kakao responses, plus the place table details are served from.

    It exists for the same reason upstream's `data/context_cache.db` does: a benchmark run
    re-asks the same lookups, and Kakao quota is finite. `ttl_seconds=0` never expires.
    Setting `KAKAO_CACHE_DB_PATH=` (blank) disables it entirely, which is what the tests do.
    """

    def __init__(self, path: str | None, *, ttl_seconds: int = 86_400):
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        if not path:
            return
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS kakao_responses ("
            "key TEXT PRIMARY KEY, payload TEXT NOT NULL, stored_at REAL NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS kakao_places ("
            "place_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        if self._connection is None:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, stored_at FROM kakao_responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload, stored_at = row
        if self.ttl_seconds and time.time() - stored_at > self.ttl_seconds:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, payload: dict[str, Any]) -> None:
        if self._connection is None:
            return
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO kakao_responses (key, payload, stored_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            self._connection.commit()

    def get_place(self, place_id: str) -> dict[str, Any] | None:
        if self._connection is None:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM kakao_places WHERE place_id = ?", (place_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def set_place(self, place: dict[str, Any]) -> None:
        if self._connection is None:
            return
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO kakao_places (place_id, payload) VALUES (?, ?)",
                (place["place_id"], json.dumps(place, ensure_ascii=False)),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
