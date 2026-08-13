from __future__ import annotations

import httpx
import pytest

from k_mapeval.providers.base import PlaceNotFoundError, UnsupportedTravelModeError
from k_mapeval.providers.kakao import KakaoMapProvider


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
    def handler(request: httpx.Request) -> httpx.Response:
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
