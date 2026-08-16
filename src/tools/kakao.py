from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from src.models import Place, Route, RouteStep
from src.tools.cache import SQLiteMapCache
from src.tools.map import (
    MapProvider,
    PlaceNotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RouteNotFoundError,
    UnsupportedTravelModeError,
)

LOCAL_BASE_URL = "https://dapi.kakao.com/v2/local"
MOBILITY_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
KAKAO_CATEGORY_CODES = frozenset(
    {
        "MT1",
        "CS2",
        "PS3",
        "SC4",
        "AC5",
        "PK6",
        "OL7",
        "SW8",
        "BK9",
        "CT1",
        "AG2",
        "PO3",
        "AT4",
        "AD5",
        "FD6",
        "CE7",
        "HP8",
        "PM9",
    }
)


class KakaoMapProvider(MapProvider):
    """Kakao Local + Kakao Mobility implementation of the common map interface.

    Every normalized retrieval is checked in SQLite before Kakao is called. Kakao Local
    has no standalone place-details endpoint, so details come from the persistent place table.
    """

    def __init__(
        self,
        rest_api_key: str,
        *,
        timeout: float = 15.0,
        cache_path: str = "data/kakao_cache.db",
        cache_ttl_seconds: int = 86_400,
        client: httpx.Client | None = None,
    ) -> None:
        if not rest_api_key:
            raise ValueError("Kakao REST API key must not be empty")
        self._rest_api_key = rest_api_key
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._cache = SQLiteMapCache(cache_path, ttl_seconds=cache_ttl_seconds)
        self._api_call_count = 0
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._places: dict[str, Place] = {}

    def __enter__(self) -> KakaoMapProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._cache.close()
        if self._owns_client:
            self._client.close()

    @property
    def api_call_count(self) -> int:
        return self._api_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count

    @property
    def cache_miss_count(self) -> int:
        return self._cache_miss_count

    @staticmethod
    def normalize_place(document: Mapping[str, Any]) -> Place:
        place_id = str(document.get("id") or document.get("address_name") or "").strip()
        road_address = document.get("road_address") or {}
        name = str(
            document.get("place_name")
            or road_address.get("building_name")
            or document.get("address_name")
            or ""
        ).strip()
        if not place_id or not name:
            raise ProviderError("Kakao place response is missing an id or name")
        try:
            latitude = float(document["y"])
            longitude = float(document["x"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("Kakao place response has invalid coordinates") from exc
        address = str(document.get("road_address_name") or document.get("address_name") or "")
        category = str(document.get("category_name") or document.get("address_type") or "address")
        return Place(
            place_id=place_id,
            name=name,
            address=address,
            latitude=latitude,
            longitude=longitude,
            category=category,
        )

    @staticmethod
    def normalize_route(data: Mapping[str, Any], origin: Place, destination: Place) -> Route:
        routes = data.get("routes") or []
        successful = next((route for route in routes if route.get("result_code") == 0), None)
        if successful is None:
            message = routes[0].get("result_msg", "No Kakao route found") if routes else "No route"
            raise RouteNotFoundError(str(message))
        summary = successful.get("summary") or {}
        steps: list[RouteStep] = []
        for section in successful.get("sections") or []:
            roads = section.get("roads") or []
            for guide in section.get("guides") or []:
                road_index = guide.get("road_index", -1)
                road_name = ""
                if isinstance(road_index, int) and 0 <= road_index < len(roads):
                    road_name = str(roads[road_index].get("name") or "")
                steps.append(
                    RouteStep(
                        instruction=str(guide.get("guidance") or guide.get("name") or ""),
                        road_name=road_name,
                        distance_m=max(0, int(guide.get("distance") or 0)),
                        duration_s=max(0, int(guide.get("duration") or 0)),
                    )
                )
        try:
            distance_m = int(summary["distance"])
            duration_s = int(summary["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("Kakao route response has an invalid summary") from exc
        return Route(
            origin=origin.name,
            destination=destination.name,
            distance_m=distance_m,
            duration_s=duration_s,
            steps=tuple(steps),
        )

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        if not query.strip():
            raise ValueError("query must not be empty")
        size = _size(limit)
        cache_args = {"query": _normalized_text(query), "limit": size}
        cached = self._cache.get_places("search_place", cache_args)
        if cached is not None:
            self._record_cache_hit(cached)
            return cached
        self._cache_miss_count += 1
        data = self._get_local("/search/keyword.json", {"query": query, "size": size})
        places = self._normalize_places(data, size)
        self._cache.set_places("search_place", cache_args, places)
        return places

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        if not address.strip():
            raise ValueError("address must not be empty")
        size = _size(limit)
        cache_args = {"address": _normalized_text(address), "limit": size}
        cached = self._cache.get_places("geocode", cache_args)
        if cached is not None:
            self._record_cache_hit(cached)
            return cached
        self._cache_miss_count += 1
        data = self._get_local("/search/address.json", {"query": address, "size": size})
        places = self._normalize_places(data, size)
        self._cache.set_places("geocode", cache_args, places)
        return places

    def nearby_search(
        self,
        center: str | Place,
        *,
        query: str | None = None,
        category_code: str | None = None,
        radius_m: int = 2000,
        limit: int = 15,
    ) -> list[Place]:
        normalized_query = query.strip() if query and query.strip() else None
        normalized_category = category_code.strip().upper() if category_code else None
        if not normalized_query and not normalized_category:
            raise ValueError("nearby_search requires query or category_code")
        if normalized_category and normalized_category not in KAKAO_CATEGORY_CODES:
            raise ValueError(
                f"Unknown Kakao category code: {normalized_category}. "
                "Use an official category group code or provide query instead."
            )
        center_place = self._resolve_place(center)
        size = _size(limit)
        normalized_radius = max(1, min(radius_m, 20000))
        params: dict[str, Any] = {
            "x": center_place.longitude,
            "y": center_place.latitude,
            "radius": normalized_radius,
            "sort": "distance",
            "size": size,
        }
        if normalized_query:
            path = "/search/keyword.json"
            params["query"] = normalized_query
            if normalized_category:
                params["category_group_code"] = normalized_category
        else:
            path = "/search/category.json"
            params["category_group_code"] = normalized_category
        cache_args = {
            "center_place_id": center_place.place_id,
            "center_latitude": center_place.latitude,
            "center_longitude": center_place.longitude,
            "query": _normalized_text(normalized_query) if normalized_query else None,
            "category_code": normalized_category,
            "radius_m": normalized_radius,
            "limit": size,
        }
        if normalized_query and normalized_category:
            # Invalidates entries created by the old implementation, which ignored query
            # whenever category_code was also present.
            cache_args["category_filter_mode"] = "keyword"
        cached = self._cache.get_places("nearby_search", cache_args)
        if cached is not None:
            self._record_cache_hit(cached)
            return cached
        self._cache_miss_count += 1
        data = self._get_local(path, params)
        places = self._normalize_places(data, size)
        self._cache.set_places("nearby_search", cache_args, places)
        return places

    def place_details(self, place_id: str) -> Place:
        place = self._cache.get_place(str(place_id))
        if place is not None:
            self._cache_hit_count += 1
            self._places[place.place_id] = place
            return place
        self._cache_miss_count += 1
        raise PlaceNotFoundError(
            "Unknown place_id. Call place_search, geocode, or nearby_search first."
        )

    def directions(
        self,
        origin: str | Place,
        destination: str | Place,
        *,
        mode: str = "driving",
        priority: str = "RECOMMEND",
    ) -> Route:
        if mode.lower() not in {"driving", "car"}:
            raise UnsupportedTravelModeError(
                "K-MapEval MVP uses Kakao Mobility and currently supports driving only"
            )
        origin_place = self._resolve_place(origin)
        destination_place = self._resolve_place(destination)
        normalized_priority = priority.upper()
        if normalized_priority not in {"RECOMMEND", "TIME", "DISTANCE"}:
            raise ValueError("priority must be RECOMMEND, TIME, or DISTANCE")
        cache_args = {
            "origin_place_id": origin_place.place_id,
            "origin_latitude": origin_place.latitude,
            "origin_longitude": origin_place.longitude,
            "destination_place_id": destination_place.place_id,
            "destination_latitude": destination_place.latitude,
            "destination_longitude": destination_place.longitude,
            "mode": "driving",
            "priority": normalized_priority,
        }
        cached = self._cache.get_route("directions", cache_args)
        if cached is not None:
            self._cache_hit_count += 1
            return cached
        self._cache_miss_count += 1
        response = self._request(
            MOBILITY_DIRECTIONS_URL,
            api_key=self._rest_api_key,
            params={
                "origin": f"{origin_place.longitude},{origin_place.latitude}",
                "destination": f"{destination_place.longitude},{destination_place.latitude}",
                "priority": normalized_priority,
                # Benchmarks use route distance/duration, so avoid downloading roads,
                # vertices, bounds, and turn-by-turn guides.
                "summary": "true",
            },
        )
        route = self.normalize_route(response, origin_place, destination_place)
        self._cache.set_route("directions", cache_args, route)
        return route

    def _resolve_place(self, value: str | Place) -> Place:
        if isinstance(value, Place):
            self._places[value.place_id] = value
            return value
        if value in self._places:
            return self._places[value]
        cached_place = self._cache.get_place(value)
        if cached_place is not None:
            self._cache_hit_count += 1
            self._places[cached_place.place_id] = cached_place
            return cached_place
        matches = self.search_place(value, limit=1)
        if not matches:
            raise PlaceNotFoundError(f"Place not found: {value}")
        return matches[0]

    def _get_local(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request(f"{LOCAL_BASE_URL}{path}", api_key=self._rest_api_key, params=params)

    def _request(self, url: str, *, api_key: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self._api_call_count += 1
        try:
            response = self._client.get(
                url,
                headers={
                    "Authorization": f"KakaoAK {api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                params=params,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Kakao API request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Kakao API request failed: {exc}") from exc
        if response.status_code in {401, 403}:
            raise ProviderAuthError("Kakao API rejected the configured REST API key")
        if response.status_code == 429:
            raise ProviderRateLimitError("Kakao API rate limit exceeded")
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise ProviderError(f"Kakao API returned HTTP {response.status_code}") from exc
        if not isinstance(payload, Mapping):
            raise ProviderError("Kakao API returned a non-object response")
        return payload

    def _normalize_places(self, data: Mapping[str, Any], limit: int) -> list[Place]:
        places = [self.normalize_place(document) for document in (data.get("documents") or [])]
        places = places[:limit]
        self._places.update({place.place_id: place for place in places})
        return places

    def _record_cache_hit(self, places: list[Place]) -> None:
        self._cache_hit_count += 1
        self._places.update({place.place_id: place for place in places})


def _size(limit: int) -> int:
    return max(1, min(limit, 15))


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()
