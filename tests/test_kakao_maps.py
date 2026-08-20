"""`KakaoMapsClient` has to keep Google Maps' shape, because ~9,900 vendored lines read it.

Every assertion here is about the *contract*, not about Kakao: what a vendored caller in
`spatial_agent/agent/operators.py` or `mapeval_api/FormattedTools.py` indexes into. If one of
these fails, upstream code that was never touched starts returning None a long way from the
change that broke it.

Kakao is stubbed with `httpx.MockTransport` throughout; no key and no network are needed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.kakao_maps import (
    KakaoAuthError,
    KakaoMapsClient,
    KakaoRateLimitError,
    KakaoTimeoutError,
    format_distance,
    format_duration,
    parse_coordinate_literal,
)

STARBUCKS = {
    "id": "26338954",
    "place_name": "스타벅스 강남점",
    "category_name": "음식점 > 카페 > 커피전문점 > 스타벅스",
    "phone": "02-555-1234",
    "address_name": "서울 강남구 역삼동 825-22",
    "road_address_name": "서울 강남구 강남대로 390",
    "x": "127.02758",
    "y": "37.49794",
    "place_url": "http://place.map.kakao.com/26338954",
    "distance": "120",
}
TWOSOME = {
    "id": "11111111",
    "place_name": "투썸플레이스 역삼점",
    "category_name": "음식점 > 카페 > 커피전문점",
    "address_name": "서울 강남구 역삼동 700",
    "road_address_name": "서울 강남구 테헤란로 152",
    "x": "127.03650",
    "y": "37.50060",
    "distance": "450",
}
BUSAN_STARBUCKS = {
    "id": "99999999",
    "place_name": "스타벅스 해운대점",
    "category_name": "음식점 > 카페 > 커피전문점 > 스타벅스",
    "address_name": "부산 해운대구 우동",
    "road_address_name": "부산 해운대구 해운대해변로 264",
    "x": "129.16000",
    "y": "35.16000",
}

ROUTE = {
    "trans_id": "t",
    "routes": [
        {
            "result_code": 0,
            "result_msg": "길찾기 성공",
            "summary": {"distance": 5300, "duration": 900},
            "sections": [
                {
                    "distance": 5300,
                    "duration": 900,
                    "roads": [{"name": "강남대로"}, {"name": "테헤란로"}],
                    "guides": [
                        {
                            "guidance": "왕십리로 방면으로 좌회전",
                            "type": 2,
                            "distance": 300,
                            "duration": 60,
                            "road_index": 0,
                        },
                        {
                            "guidance": "직진",
                            "type": 1,
                            "distance": 5000,
                            "duration": 840,
                            "road_index": 1,
                        },
                    ],
                }
            ],
        }
    ],
}


def build_client(
    routes: list[tuple[str, dict[str, Any]]], **kwargs: Any
) -> tuple[KakaoMapsClient, list[httpx.Request]]:
    """A client whose Kakao is a scripted transport, and the requests it actually made."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        for fragment, payload in routes:
            if fragment in str(request.url):
                if isinstance(payload, int):
                    return httpx.Response(payload, json={"errorType": "e"})
                return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"documents": [], "meta": {"is_end": True}})

    client = KakaoMapsClient(
        "test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        # The cache is a run-level convenience, not part of the contract under test. Every
        # test here asserts on the requests actually issued, so it is off unless asked for.
        cache_path=kwargs.pop("cache_path", ""),
        search_center=kwargs.pop("search_center", None),
        search_radius_m=kwargs.pop("search_radius_m", 0),
        **kwargs,
    )
    return client, seen


# ------------------------------------------------------------------ geocoding


def test_geocode_returns_the_four_keys_every_vendored_caller_indexes() -> None:
    client, _ = build_client([("search/keyword", {"documents": [STARBUCKS]})])

    result = client.geocode("스타벅스 강남점")

    assert set(result) == {"lat", "lng", "formatted_address", "place_id"}
    assert result["lat"] == pytest.approx(37.49794)
    assert result["lng"] == pytest.approx(127.02758)
    assert result["place_id"] == "26338954"


def test_a_name_the_address_index_does_not_carry_still_reaches_the_keyword_index() -> None:
    """Kakao splits what Google's Geocoding API answered in one place.

    A Korean question names a POI far more often than an address, so a geocode that stopped at
    `/search/address.json` would fail to resolve most of the benchmark.
    """

    client, seen = build_client([("search/keyword", {"documents": [STARBUCKS]})])

    assert client.geocode("스타벅스 강남점") is not None
    assert "search/address" in str(seen[0].url)
    assert "search/keyword" in str(seen[1].url)


def test_geocode_picks_the_candidate_nearest_the_bias() -> None:
    client, _ = build_client([("search/keyword", {"documents": [BUSAN_STARBUCKS, STARBUCKS]})])

    result = client.geocode("스타벅스", location_bias=(37.4979, 127.0276))

    assert result["place_id"] == STARBUCKS["id"]


def test_a_candidate_too_far_from_the_bias_is_refused_rather_than_returned() -> None:
    """Upstream's own threshold: the nearest candidate is always *some* candidate."""

    client, _ = build_client([("search/keyword", {"documents": [BUSAN_STARBUCKS]})])

    assert client.geocode("스타벅스 강남점", location_bias=(37.4979, 127.0276)) is None


def test_the_region_prior_biases_the_first_query_and_never_hides_a_place() -> None:
    """A name with no match in the region still resolves nationwide.

    Korean POI names repeat across cities, so the prior is what stops a bare brand name from
    resolving to another province — but it says where to look first, not where a place may be.
    """

    empty = {"documents": [], "meta": {"is_end": True}}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "search/address" in str(request.url):
            return httpx.Response(200, json=empty)
        # The biased query carries the centre; the nationwide fallback does not.
        if "x=127" in str(request.url):
            return httpx.Response(200, json=empty)
        return httpx.Response(200, json={"documents": [BUSAN_STARBUCKS]})

    client = KakaoMapsClient(
        "test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        cache_path="",
        search_center=(37.4979, 127.0276),
        search_radius_m=20000,
    )

    result = client.geocode("스타벅스 해운대점")

    assert result["place_id"] == BUSAN_STARBUCKS["id"]
    keyword_calls = [call for call in calls if "search/keyword" in call]
    assert len(keyword_calls) == 2
    assert "x=127" in keyword_calls[0] and "x=127" not in keyword_calls[1]


def test_reverse_geocode_prefers_the_road_address() -> None:
    client, _ = build_client(
        [
            (
                "coord2address",
                {
                    "documents": [
                        {
                            "road_address": {"address_name": "서울 강남구 강남대로 390"},
                            "address": {"address_name": "서울 강남구 역삼동 825-22"},
                        }
                    ]
                },
            )
        ]
    )

    assert client.reverse_geocode(37.4979, 127.0276) == "서울 강남구 강남대로 390"


# ------------------------------------------------------------------ retrieval


def test_nearby_search_returns_the_google_place_keys() -> None:
    client, _ = build_client([("search/category", {"documents": [STARBUCKS, TWOSOME]})])

    places = client.nearby_search((37.4979, 127.0276), radius=1000, place_type="cafe")

    assert [place["name"] for place in places] == ["스타벅스 강남점", "투썸플레이스 역삼점"]
    assert set(places[0]) >= {
        "name", "place_id", "lat", "lng", "rating", "user_ratings_total",
        "price_level", "types", "vicinity", "open_now",
    }
    assert places[0]["types"] == ["음식점", "카페", "커피전문점", "스타벅스"]


def test_a_requested_type_reaches_kakaos_own_category_code() -> None:
    client, seen = build_client([("search/category", {"documents": [STARBUCKS]})])

    client.nearby_search((37.4979, 127.0276), place_type="convenience_store")

    assert "category_group_code=CS2" in str(seen[0].url)


def test_a_type_kakao_has_no_code_for_is_asked_for_by_name() -> None:
    client, seen = build_client([("search/keyword", {"documents": [STARBUCKS]})])

    client.nearby_search((37.4979, 127.0276), place_type="gym")

    assert "search/keyword" in str(seen[0].url)
    assert "%ED%97%AC%EC%8A%A4%EC%9E%A5" in str(seen[0].url)  # 헬스장


def test_min_rating_does_not_delete_every_candidate() -> None:
    """A filter over a field the evidence does not carry is not a filter, it is a delete.

    Kakao Local publishes no rating, so upstream's `rating < min_rating` would have emptied
    every candidate list — and an empty list is what the generation stage guesses over.
    """

    client, _ = build_client([("search/category", {"documents": [STARBUCKS, TWOSOME]})])

    places = client.nearby_search((37.4979, 127.0276), place_type="cafe", min_rating=4.5)

    assert len(places) == 2


def test_the_ranking_preserves_kakaos_distance_order() -> None:
    """Upstream sorted by (rating, user_ratings_total). Both are constant here, and Python's
    sort is stable, so the order Kakao returned — nearest first — survives."""

    client, _ = build_client([("search/category", {"documents": [STARBUCKS, TWOSOME]})])

    places = client.nearby_search((37.4979, 127.0276), place_type="cafe")

    assert [place["distance"] for place in places] == [120, 450]


# ------------------------------------------------------------------ details


def test_place_details_answers_for_an_id_the_client_issued() -> None:
    client, _ = build_client([("search/keyword", {"documents": [STARBUCKS]})])
    client.text_search("스타벅스 강남점")

    details = client.get_place_details("26338954")

    assert details["name"] == "스타벅스 강남점"
    assert details["lat"] == pytest.approx(37.49794)
    assert details["formatted_phone_number"] == "02-555-1234"
    # The keys stay so the vendored formatters keep working; None says Kakao does not publish
    # the field, which is not the same as asserting a value for it.
    assert details["rating"] is None
    assert details["opening_hours"] is None


def test_an_id_the_client_never_issued_is_unknown() -> None:
    client, _ = build_client([])

    assert client.get_place_details("does-not-exist") is None


# ------------------------------------------------------------------ routing


def test_get_directions_returns_routes_and_primary() -> None:
    client, _ = build_client([("directions", ROUTE)])

    result = client.get_directions("37.49794,127.02758", "37.50060,127.03650")

    assert set(result) == {"routes", "primary"}
    primary = result["primary"]
    assert primary["distance"] == 5300
    assert primary["duration"] == 900
    assert primary["distance_text"] == "5.3 km"
    assert primary["duration_text"] == "15 mins"
    assert primary["summary"] == "강남대로"


def test_a_route_step_carries_the_instruction_the_formatters_print() -> None:
    client, _ = build_client([("directions", ROUTE)])

    steps = client.get_directions("37.49794,127.02758", "37.50060,127.03650")["primary"]["steps"]

    assert [step["html_instructions"] for step in steps] == ["왕십리로 방면으로 좌회전", "직진"]
    assert steps[0]["maneuver"] == "turn-left"
    # A turn-count question is asked about a named road, and the guidance does not always
    # name it, so the road each guide belongs to is resolved through `road_index`.
    assert steps[0]["road_name"] == "강남대로"
    assert steps[1]["road_name"] == "테헤란로"


def test_a_non_driving_route_is_refused_rather_than_answered_by_car() -> None:
    """Kakao Mobility has no walking, cycling or transit API.

    Answering a walking question with a driving route would answer a different question.
    """

    client, seen = build_client([("directions", ROUTE)])

    assert client.get_directions("37.4,127.0", "37.5,127.1", mode="walking") is None
    assert seen == []


def test_waypoints_go_to_kakaos_waypoint_endpoint_in_the_order_given() -> None:
    client, seen = build_client([("waypoints/directions", ROUTE)])

    client.get_directions(
        "37.49,127.02", "37.51,127.04", waypoints=["37.50,127.03", "37.505,127.035"]
    )

    body = json.loads(seen[0].content)
    assert [point["y"] for point in body["waypoints"]] == [37.50, 37.505]


# ------------------------------------------------------------------ matrix


def test_the_matrix_is_shaped_the_way_the_vendored_tsp_reads_it() -> None:
    """`operators.tsp_tw` walks `rows[i]['elements'][j]['duration']['value']` (line 1955).

    The operator was never touched, so the matrix has to arrive in exactly that shape or the
    whole trip family silently falls back to a haversine estimate.
    """

    client, _ = build_client([("directions", ROUTE)])

    matrix = client.get_distance_matrix(["37.49,127.02"], ["37.51,127.04"])

    element = matrix["rows"][0]["elements"][0]
    assert element["status"] == "OK"
    assert element["duration"]["value"] == 900
    assert element["distance"]["value"] == 5300
    assert element["duration"]["text"] == "15 mins"


def test_the_diagonal_is_answered_locally_because_kakao_refuses_that_leg() -> None:
    """Kakao refuses a leg under 5 m, and a trip matrix asks for its own diagonal every run.

    It is the only leg that may be filled: a missing off-diagonal leg stays a failed element,
    never a free hop.
    """

    client, seen = build_client([("directions", ROUTE)])

    matrix = client.get_distance_matrix(["37.49,127.02"], ["37.49,127.02"])

    assert matrix["rows"][0]["elements"][0] == {
        "status": "OK",
        "distance": {"value": 0, "text": "0 m"},
        "duration": {"value": 0, "text": "1 min"},
    }
    assert seen == []


def test_a_leg_kakao_could_not_route_is_reported_missing_not_zero() -> None:
    refusal = {"routes": [{"result_code": 104, "result_msg": "출발지와 도착지가 5 m 이내"}]}
    client, _ = build_client([("directions", refusal)])

    matrix = client.get_distance_matrix(["37.49,127.02"], ["37.61,127.44"])

    assert matrix["rows"][0]["elements"][0] == {"status": "ZERO_RESULTS"}


def test_format_output_flattens_the_matrix_as_upstream_did() -> None:
    client, _ = build_client([("directions", ROUTE)])

    matrix = client.get_distance_matrix(
        ["37.49,127.02"], ["37.51,127.04"], format_output=True
    )

    assert matrix == [[{
        "distance": 5300,
        "duration": 900,
        "distance_text": "5.3 km",
        "duration_text": "15 mins",
    }]]


# ------------------------------------------------------------------ timezone


def test_the_timezone_is_answered_locally_because_kakaos_coverage_is_one_zone() -> None:
    client, seen = build_client([])

    zone = client.get_timezone(37.4979, 127.0276, 1_700_000_000)

    assert zone == {
        "timeZoneId": "Asia/Seoul",
        "timeZoneName": "Korean Standard Time",
        "rawOffset": 32400,
        "dstOffset": 0,
    }
    assert seen == []


# ------------------------------------------------------------------ failures


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, KakaoAuthError), (403, KakaoAuthError), (429, KakaoRateLimitError)],
)
def test_a_rejected_key_and_a_rate_limit_are_distinct_failures(status, expected) -> None:
    """The Evaluator retries a question the provider could not answer *right now*, and never
    one whose key is wrong. That decision is made on the class name."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"errorType": "e"})

    client = KakaoMapsClient(
        "bad", client=httpx.Client(transport=httpx.MockTransport(handler)), cache_path=""
    )

    with pytest.raises(expected):
        client.text_search("스타벅스")


def test_a_timeout_is_its_own_class_so_the_question_can_be_asked_again() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = KakaoMapsClient(
        "k", client=httpx.Client(transport=httpx.MockTransport(handler)), cache_path=""
    )

    with pytest.raises(KakaoTimeoutError):
        client.text_search("스타벅스")


# ------------------------------------------------------------------ cache


def test_a_repeated_lookup_is_a_cache_hit_not_a_second_api_call(tmp_path) -> None:
    client, seen = build_client(
        [("search/keyword", {"documents": [STARBUCKS]})],
        cache_path=str(tmp_path / "kakao.db"),
    )

    client.text_search("스타벅스 강남점")
    client.text_search("스타벅스 강남점")

    assert len(seen) == 1
    assert client.api_call_count == 1
    assert client.cache_hit_count == 1


def test_details_survive_a_restart_through_the_cache(tmp_path) -> None:
    """Kakao has no details endpoint, so a place id is only answerable from what a search
    already returned. Losing that table between questions would break the ReAct baseline's
    whole PlaceSearch-then-PlaceDetails idiom."""

    path = str(tmp_path / "kakao.db")
    first, _ = build_client([("search/keyword", {"documents": [STARBUCKS]})], cache_path=path)
    first.text_search("스타벅스 강남점")
    first.close()

    second, _ = build_client([], cache_path=path)

    assert second.get_place_details("26338954")["name"] == "스타벅스 강남점"


# ------------------------------------------------------------------ formatting


@pytest.mark.parametrize(
    ("meters", "text"),
    [(0, "0 m"), (450, "450 m"), (999, "999 m"), (1000, "1.0 km"), (5300, "5.3 km")],
)
def test_distance_text_matches_googles_phrasing(meters, text) -> None:
    assert format_distance(meters) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "1 min"),
        (59, "1 min"),
        (120, "2 mins"),
        (900, "15 mins"),
        (3600, "1 hour"),
        (3900, "1 hour 5 mins"),
    ],
)
def test_duration_text_matches_googles_phrasing(seconds, text) -> None:
    assert format_duration(seconds) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("37.49794,127.02758", (37.49794, 127.02758)),
        (" 37.5 , 127.0 ", (37.5, 127.0)),
        ("스타벅스 강남점", None),
        ("서울, 강남구", None),
        ("999,999", None),
    ],
)
def test_a_resolved_place_travels_between_callers_as_a_coordinate_literal(text, expected) -> None:
    """`nodes.get_directions` writes a place it has already resolved as "lat,lng"."""

    assert parse_coordinate_literal(text) == expected
