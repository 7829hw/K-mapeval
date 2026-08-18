from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset import load_dataset
from src.tools import ContextMapProvider, ToolRegistry
from src.tools.context import parse_context
from src.tools.map import PlaceNotFoundError, RouteNotFoundError, UnsupportedTravelModeError

NEARBY_CONTEXT = """Information of <b>평화시장</b>:
- Location: 주소 정보 없음(37.5692700, 127.0059920).
- OSM ID: node/368867533.
- Type: supermarket.

Nearby book_store of 평화시장 are (sorted by distance in ascending order):
1. <b>어라운드북</b>(37.5679133, 127.0070815)
   - Address: 주소 정보 없음.
2. <b>교보문고</b>(37.5685166, 127.0079934)
   - Address: 서울특별시 중구 을지로3가 을지로 108.
3. <b>소요서가</b>(37.5681402, 126.9955093)
   - Address: 주소 정보 없음.
"""

RADIUS_CONTEXT = """Information of <b>미소의집</b>:
- Location: 서울특별시 서초구 반포동 신반포로 27-6(37.5021773, 126.9877564).
- OSM ID: node/12192225021.
- Type: restaurant.

Nearby amenity=bank of 미소의집 are (in requested radius):
1. <b>우리은행</b>(37.5020575, 126.9867916)
   - Address: 주소 정보 없음.
"""

ROUTE_CONTEXT = """Information of <b>회기버거</b>:
- Location: 주소 정보 없음(37.5686414, 126.9650578).
- OSM ID: node/13681878975.

Information of <b>GS25 서대문남가좌점</b>:
- Location: 서울특별시 서대문구 증가로6길 32(37.5778520, 126.9260540).
- OSM ID: node/13911613405.

Route from 회기버거 to GS25 서대문남가좌점 by 자동차:
- Duration: 7분.
- Distance: 4994m.
- Origin snapped node: 8382521110.
- Destination snapped node: 4582139114.
"""


def test_a_context_block_becomes_places_a_retrieval_and_a_route() -> None:
    nearby = parse_context(NEARBY_CONTEXT)
    anchor = nearby.places[0]
    assert (anchor.name, anchor.place_id, anchor.category) == (
        "평화시장",
        "node/368867533",
        "supermarket",
    )
    assert anchor.address == ""  # "주소 정보 없음" is an absent address, not an address
    assert [place.name for place in nearby.nearby[0].places] == [
        "어라운드북",
        "교보문고",
        "소요서가",
    ]
    assert nearby.nearby[0].places[1].address == "서울특별시 중구 을지로3가 을지로 108"
    assert nearby.nearby[0].radius_bounded is False
    assert parse_context(RADIUS_CONTEXT).nearby[0].radius_bounded is True

    route = parse_context(ROUTE_CONTEXT).routes[0]
    assert (route.origin, route.destination) == ("회기버거", "GS25 서대문남가좌점")
    assert (route.distance_m, route.duration_s) == (4994, 7 * 60)


def test_the_context_is_the_cache_so_no_lookup_costs_an_api_call() -> None:
    provider = ContextMapProvider()
    provider.activate_context(NEARBY_CONTEXT)

    assert [place.name for place in provider.search_place("어라운드북")] == ["어라운드북"]
    assert provider.api_call_count == 0
    assert provider.cache_hit_count == 1
    assert provider.cache_miss_count == 0


def test_a_place_the_context_does_not_hold_is_not_found() -> None:
    """A distractor the dataset invented has to fail the way a missing Kakao POI fails."""

    provider = ContextMapProvider()
    provider.activate_context(RADIUS_CONTEXT)

    assert provider.search_place("IBK기업은행") == []
    with pytest.raises(PlaceNotFoundError):
        provider.nearby_search("IBK기업은행")
    assert provider.cache_miss_count == 2


def test_one_questions_context_never_answers_the_next() -> None:
    provider = ContextMapProvider()
    provider.activate_context(NEARBY_CONTEXT)
    assert provider.search_place("어라운드북")

    provider.activate_context(RADIUS_CONTEXT)
    assert provider.search_place("어라운드북") == []
    provider.activate_context(None)
    assert provider.search_place("우리은행") == []


def test_a_k_nearest_block_is_served_whole_and_a_radius_block_is_bounded() -> None:
    """The radius argument narrows a stored retrieval only when the retrieval had one.

    A k-nearest block records no radius, so cutting it down to whatever radius the agent guessed
    would report an absence the context never states; a bounded block already is the answer set.
    """

    provider = ContextMapProvider()
    provider.activate_context(NEARBY_CONTEXT)
    assert [place.name for place in provider.nearby_search("평화시장", radius_m=100)] == [
        "어라운드북",
        "교보문고",
        "소요서가",
    ]

    provider.activate_context(RADIUS_CONTEXT)
    bounded = provider.nearby_search("미소의집", radius_m=500)
    assert [place.name for place in bounded] == ["우리은행"]
    assert provider.nearby_search("미소의집", radius_m=10) == []


def test_a_type_word_does_not_resolve_to_the_place_that_ends_in_it() -> None:
    """Otherwise a place-type question answers itself: the option would find its own anchor."""

    provider = ContextMapProvider()
    provider.activate_context(
        "Information of <b>다모아편의점</b>:\n"
        "- Location: 주소 정보 없음(37.5658790, 127.0036030).\n"
        "- OSM ID: node/368948685.\n"
        "- Type: convenience_store.\n"
    )
    assert provider.search_place("편의점") == []
    assert [place.name for place in provider.search_place("다모아편의점")] == ["다모아편의점"]


def test_a_brand_resolves_to_the_branch_that_extends_it() -> None:
    provider = ContextMapProvider()
    provider.activate_context(
        "Information of <b>CU 삼청점</b>:\n"
        "- Location: 서울특별시 종로구 삼청로 68(37.5828000, 126.9810000).\n"
        "- OSM ID: node/1.\n"
    )
    assert [place.name for place in provider.search_place("CU")] == ["CU 삼청점"]
    # A dataset that had to separate namesakes appends the address to the option text.
    assert [
        place.name for place in provider.search_place("CU 삼청점 - 서울특별시 종로구 삼청로 68,")
    ] == ["CU 삼청점"]


def test_a_branch_name_does_not_resolve_to_the_bare_brand_the_context_stores() -> None:
    """Containment is evidence in one direction only.

    A retrieval that recorded a plain GS25 found whichever GS25 was nearby; it is not the
    GS25 합정프리미엄점 an option from another district names.
    """

    provider = ContextMapProvider()
    provider.activate_context(
        "Information of <b>GS25</b>:\n"
        "- Location: 주소 정보 없음(37.5600000, 126.9300000).\n"
        "- OSM ID: node/2.\n"
    )
    assert provider.search_place("GS25 합정프리미엄점") == []
    assert [place.name for place in provider.search_place("GS25")] == ["GS25"]


def test_an_anchor_of_the_type_asked_about_still_heads_its_own_retrieval() -> None:
    """Served as recorded: the context lists the anchor first, at zero metres, and so do we.

    A convenience store asked for its nearest convenience store is in its own result, exactly as a
    live API would return it. Reading past it is the agent's job, and it is the same job for both
    architectures.
    """

    provider = ContextMapProvider()
    provider.activate_context(
        "Information of <b>GS25 화곡초교점</b>:\n"
        "- Location: 주소 정보 없음(37.5400000, 126.8400000).\n"
        "- OSM ID: node/3.\n"
        "- Type: convenience_store.\n"
        "\n"
        "Nearby convenience_store of GS25 화곡초교점 are (sorted by distance in ascending order):\n"
        "1. <b>GS25 화곡초교점</b>(37.5400000, 126.8400000)\n"
        "   - Address: 주소 정보 없음.\n"
        "2. <b>CU 화곡본동점</b>(37.5410000, 126.8410000)\n"
        "   - Address: 주소 정보 없음.\n"
    )
    found = provider.nearby_search("GS25 화곡초교점", query="편의점")
    assert [place.name for place in found] == ["GS25 화곡초교점", "CU 화곡본동점"]


def test_a_recorded_route_is_returned_and_an_unrecorded_one_fails() -> None:
    provider = ContextMapProvider()
    provider.activate_context(ROUTE_CONTEXT)

    route = provider.directions("회기버거", "GS25 서대문남가좌점")
    assert (route.distance_m, route.duration_s) == (4994, 420)
    with pytest.raises(RouteNotFoundError):
        provider.directions("GS25 서대문남가좌점", "회기버거")
    with pytest.raises(UnsupportedTravelModeError):
        provider.directions("회기버거", "GS25 서대문남가좌점", mode="walking")
    assert provider.api_call_count == 0


def test_the_shared_tool_layer_reads_the_context_like_any_other_provider() -> None:
    """Both architectures reach the context only through the registry, unchanged."""

    provider = ContextMapProvider()
    provider.activate_context(NEARBY_CONTEXT)
    registry = ToolRegistry(provider)

    execution = registry.invoke(
        "batch_geocode", {"place_names": ["평화시장", "어라운드북"], "anchor": "평화시장"}
    )
    assert execution.status == "ok"
    assert [entry["place"]["name"] for entry in execution.output] == ["평화시장", "어라운드북"]
    assert execution.output[0]["place"]["place_id"] == "node/368867533"
    assert execution.output[1]["place"]["category"] == "book_store"

    # The block was retrieved by type, so a Korean keyword none of its names contains still
    # answers with it: the context recorded one retrieval for this anchor and that is the answer.
    nearby = registry.invoke("nearby_places", {"center": "평화시장", "query": "서점"})
    assert nearby.status == "ok"
    assert [place["name"] for place in nearby.output] == ["어라운드북", "교보문고", "소요서가"]
    assert provider.api_call_count == 0


def test_every_benchmark_context_parses_into_evidence() -> None:
    for item in load_dataset(Path("dataset/seoul_mapeval_v1_mcq_100.jsonl")):
        assert item.context is not None
        document = parse_context(item.context)
        assert document.places, item.id
        if item.classification == "routing":
            assert document.routes, item.id
        if item.classification in ("nearby", "radius"):
            assert document.nearby, item.id
