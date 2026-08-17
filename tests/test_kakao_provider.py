from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest

from src.tools.kakao import KakaoMapProvider
from src.tools.map import PlaceNotFoundError, UnsupportedTravelModeError


def _document(place_id: str, name: str, x: float, y: float) -> dict:
    return {
        "id": place_id,
        "place_name": name,
        "address_name": f"서울 {name}",
        "road_address_name": f"서울로 {name}",
        "category_name": "관광명소",
        "x": str(x),
        "y": str(y),
    }


def test_search_nearby_details_and_route_are_normalized() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "dapi.kakao.com":
            query = request.url.params.get("query")
            if query == "경복궁":
                docs = [_document("1", "경복궁", 126.977, 37.5796)]
            elif query == "서울역":
                docs = [_document("2", "서울역", 126.9707, 37.5547)]
            else:
                docs = [_document("3", "경복궁역", 126.9735, 37.5758)]
            return httpx.Response(200, json={"documents": docs})
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "result_code": 0,
                        "summary": {"distance": 4100, "duration": 900},
                        "sections": [
                            {
                                "roads": [{"name": "세종대로"}],
                                "guides": [
                                    {
                                        "guidance": "직진",
                                        "road_index": 0,
                                        "distance": 300,
                                        "duration": 60,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = KakaoMapProvider("test-key", cache_path=":memory:", client=client)

    palace = provider.search_place("경복궁", limit=1)[0]
    assert palace.model_dump() == {
        "place_id": "1",
        "name": "경복궁",
        "address": "서울로 경복궁",
        "latitude": 37.5796,
        "longitude": 126.977,
        "category": "관광명소",
        "phone": "",
        "place_url": "",
        "rating": None,
        "price_level": None,
        "opening_hours": None,
        "timezone": None,
        "is_open": None,
    }
    assert provider.place_details("1") == palace
    nearby = provider.nearby_search(palace, query="지하철역", limit=1)
    assert nearby[0].name == "경복궁역"
    route = provider.directions("서울역", palace)
    assert route.origin == "서울역"
    assert route.destination == "경복궁"
    assert route.distance_m == 4100
    assert route.duration_s == 900
    assert route.steps[0].road_name == "세종대로"
    assert provider.api_call_count == 4
    nearby_request = next(
        request for request in requests if request.url.params.get("sort") == "distance"
    )
    assert nearby_request.url.path == "/v2/local/search/keyword.json"
    assert nearby_request.url.params["radius"] == "2000"
    assert nearby_request.url.params["x"] == str(palace.longitude)
    assert nearby_request.url.params["y"] == str(palace.latitude)
    directions_request = next(
        request for request in requests if request.url.host == "apis-navi.kakaomobility.com"
    )
    assert directions_request.url.path == "/v1/directions"
    assert directions_request.url.params["summary"] == "true"
    assert directions_request.headers["Authorization"] == "KakaoAK test-key"
    assert directions_request.headers["Content-Type"] == "application/json"


def test_nearby_query_with_category_uses_keyword_filter() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"documents": []})

    center = KakaoMapProvider.normalize_place(_document("1", "서울역", 126.9707, 37.5547))
    provider = KakaoMapProvider(
        "test-key",
        cache_path=":memory:",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.nearby_search(center, query="카페", category_code="ce7") == []
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v2/local/search/keyword.json"
    assert request.url.params["query"] == "카페"
    assert request.url.params["category_group_code"] == "CE7"
    assert request.url.params["sort"] == "distance"


def test_reverse_geocode_is_normalized_and_cached() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "road_address": {
                            "address_name": "서울 종로구 사직로 161",
                            "building_name": "경복궁",
                        },
                        "address": {"address_name": "서울 종로구 세종로 1-1"},
                    }
                ]
            },
        )

    provider = KakaoMapProvider(
        "test-key",
        cache_path=":memory:",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.reverse_geocode(37.5796, 126.977, limit=1)[0].name == "경복궁"
    assert provider.reverse_geocode(37.5796, 126.977, limit=1)[0].name == "경복궁"
    assert len(requests) == 1
    assert requests[0].url.path == "/v2/local/geo/coord2address.json"


def test_waypoint_route_uses_post_and_verifies_summary() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "result_code": 0,
                        "summary": {
                            "distance": 100,
                            "duration": 20,
                            "waypoints": [{"name": "경유", "x": 127.1, "y": 37.1}],
                        },
                        "sections": [
                            {
                                "roads": [{"name": "도로"}],
                                "guides": [{"guidance": "우회전", "road_index": 0}],
                            }
                        ],
                    }
                ]
            },
        )

    provider = KakaoMapProvider(
        "test-key",
        cache_path=":memory:",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    origin = KakaoMapProvider.normalize_place(_document("o", "출발", 127.0, 37.0))
    waypoint = KakaoMapProvider.normalize_place(_document("w", "경유", 127.1, 37.1))
    destination = KakaoMapProvider.normalize_place(_document("d", "도착", 127.2, 37.2))
    route = provider.directions(
        origin, destination, waypoints=[waypoint], include_steps=True
    )
    assert route.waypoints == ("경유",)
    assert route.steps[0].instruction == "우회전"
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v1/waypoints/directions"
    assert json.loads(requests[0].content)["summary"] is False


def test_nearby_retains_ranked_results_across_kakao_pages() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = int(request.url.params["page"])
        start = (page - 1) * 15
        documents = [
            _document(str(index), f"장소-{index}", 126.97, 37.55)
            for index in range(start, min(start + 15, 32))
        ]
        return httpx.Response(
            200,
            json={"documents": documents, "meta": {"is_end": page == 3}},
        )

    center = KakaoMapProvider.normalize_place(_document("center", "서울역", 126.97, 37.55))
    provider = KakaoMapProvider(
        "test-key",
        cache_path=":memory:",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    places = provider.nearby_search(center, query="서점", limit=45)

    assert len(places) == 32
    assert [request.url.params["page"] for request in requests] == ["1", "2", "3"]
    assert provider.api_call_count == 3


def test_nearby_rejects_non_kakao_category_without_calling_api() -> None:
    def fail_if_called(_: httpx.Request) -> httpx.Response:
        raise AssertionError("An invalid category code must not reach Kakao")

    center = KakaoMapProvider.normalize_place(_document("1", "서울역", 126.9707, 37.5547))
    provider = KakaoMapProvider(
        "test-key",
        cache_path=":memory:",
        client=httpx.Client(transport=httpx.MockTransport(fail_if_called)),
    )
    with pytest.raises(ValueError, match="official category group code"):
        provider.nearby_search(center, category_code="LIBRARY")
    assert provider.api_call_count == 0


def test_place_details_requires_a_prior_retrieval() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    provider = KakaoMapProvider("test-key", cache_path=":memory:", client=client)
    with pytest.raises(PlaceNotFoundError):
        provider.place_details("missing")


def test_non_driving_route_is_explicitly_unsupported() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    provider = KakaoMapProvider("test-key", cache_path=":memory:", client=client)
    with pytest.raises(UnsupportedTravelModeError):
        provider.directions("서울역", "경복궁", mode="walking")
    assert provider.api_call_count == 0


def test_address_response_without_place_id_is_normalized() -> None:
    place = KakaoMapProvider.normalize_place(
        {
            "address_name": "서울 종로구 세종로 1-1",
            "address_type": "REGION_ADDR",
            "x": "126.978",
            "y": "37.5665",
            "road_address": None,
        }
    )
    assert place.place_id == "서울 종로구 세종로 1-1"
    assert place.name == "서울 종로구 세종로 1-1"


def test_sqlite_cache_survives_provider_restart_and_skips_api(tmp_path) -> None:
    cache_path = tmp_path / "kakao.db"
    request_count = 0

    def first_handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={"documents": [_document("1", "경복궁", 126.977, 37.5796)]},
        )

    first_client = httpx.Client(transport=httpx.MockTransport(first_handler))
    first = KakaoMapProvider("test-key", cache_path=str(cache_path), client=first_client)
    assert first.search_place("  경복궁  ", limit=1)[0].name == "경복궁"
    assert first.api_call_count == 1
    assert first.cache_miss_count == 1
    first.close()

    def fail_if_called(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Kakao API must not be called for a persistent cache hit")

    second_client = httpx.Client(transport=httpx.MockTransport(fail_if_called))
    second = KakaoMapProvider("test-key", cache_path=str(cache_path), client=second_client)
    assert second.search_place("경복궁", limit=1)[0].name == "경복궁"
    assert second.api_call_count == 0
    assert second.cache_hit_count == 1
    assert request_count == 1
    second.close()


def test_directions_uses_cached_places_and_route_after_restart(tmp_path) -> None:
    cache_path = tmp_path / "routes.db"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "dapi.kakao.com":
            query = request.url.params["query"]
            document = (
                _document("a", "서울역", 126.9707, 37.5547)
                if query == "서울역"
                else _document("b", "경복궁", 126.977, 37.5796)
            )
            return httpx.Response(200, json={"documents": [document]})
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "result_code": 0,
                        "summary": {"distance": 4100, "duration": 900},
                        "sections": [],
                    }
                ]
            },
        )

    first_client = httpx.Client(transport=httpx.MockTransport(handler))
    first = KakaoMapProvider("test-key", cache_path=str(cache_path), client=first_client)
    assert first.directions("서울역", "경복궁").distance_m == 4100
    assert first.api_call_count == 3
    first.close()

    def fail_if_called(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Persistent route cache should prevent every Kakao API call")

    second_client = httpx.Client(transport=httpx.MockTransport(fail_if_called))
    second = KakaoMapProvider("test-key", cache_path=str(cache_path), client=second_client)
    assert second.directions("서울역", "경복궁").duration_s == 900
    assert second.api_call_count == 0
    assert second.cache_hit_count == 3
    second.close()


def test_directions_resolves_persisted_place_ids_without_local_api_calls(tmp_path) -> None:
    cache_path = tmp_path / "place-ids.db"

    def local_handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        document = (
            _document("origin-id", "서울역", 126.9707, 37.5547)
            if query == "서울역"
            else _document("destination-id", "경복궁", 126.977, 37.5796)
        )
        return httpx.Response(200, json={"documents": [document]})

    first = KakaoMapProvider(
        "test-key",
        cache_path=str(cache_path),
        client=httpx.Client(transport=httpx.MockTransport(local_handler)),
    )
    first.search_place("서울역", limit=1)
    first.search_place("경복궁", limit=1)
    first.close()

    requests: list[httpx.Request] = []

    def route_only_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "apis-navi.kakaomobility.com"
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "result_code": 0,
                        "summary": {"distance": 4100, "duration": 900},
                        "sections": [],
                    }
                ]
            },
        )

    second = KakaoMapProvider(
        "test-key",
        cache_path=str(cache_path),
        client=httpx.Client(transport=httpx.MockTransport(route_only_handler)),
    )
    route = second.directions("origin-id", "destination-id")
    assert route.distance_m == 4100
    assert len(requests) == 1
    assert second.api_call_count == 1
    assert second.cache_hit_count == 2
    second.close()


def test_four_provider_sessions_share_sqlite_cache_concurrently(tmp_path) -> None:
    cache_path = tmp_path / "concurrent.db"
    request_barrier = Barrier(4)

    def run_session(index: int) -> str:
        def handler(_: httpx.Request) -> httpx.Response:
            request_barrier.wait(timeout=5)
            return httpx.Response(
                200,
                json={
                    "documents": [
                        _document(str(index), f"장소-{index}", 126.97 + index / 1000, 37.55)
                    ]
                },
            )

        provider = KakaoMapProvider(
            "test-key",
            cache_path=str(cache_path),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        try:
            return provider.search_place(f"장소-{index}", limit=1)[0].name
        finally:
            provider.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        names = list(executor.map(run_session, range(4)))
    assert names == [f"장소-{index}" for index in range(4)]

    def fail_if_called(_: httpx.Request) -> httpx.Response:
        raise AssertionError("All four concurrent results should be cached")

    cached = KakaoMapProvider(
        "test-key",
        cache_path=str(cache_path),
        client=httpx.Client(transport=httpx.MockTransport(fail_if_called)),
    )
    try:
        assert [
            cached.search_place(f"장소-{index}", limit=1)[0].name for index in range(4)
        ] == names
        assert cached.api_call_count == 0
        assert cached.cache_hit_count == 4
    finally:
        cached.close()
