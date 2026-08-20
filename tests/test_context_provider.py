from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset import load_dataset
from src.models import Place, Route
from src.tools import ContextMapProvider, MapProvider, ToolRegistry
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


def _ref(provider, name: str):
    """Resolve a name the way an agent must: search first, then pass what came back.

    A place argument is a reference the provider issued — `mapeval-api/FormattedTools.py` gives
    its baseline no other way — so these tests thread the resolved place rather than the name.
    """

    return provider.search_place(name, limit=1)[0]


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
    provider = ContextMapProvider([NEARBY_CONTEXT])

    assert [place.name for place in provider.search_place("어라운드북")] == ["어라운드북"]
    assert provider.api_call_count == 0
    assert provider.cache_hit_count == 1
    assert provider.cache_miss_count == 0


def test_a_place_the_context_does_not_hold_is_not_found() -> None:
    """A distractor the dataset invented has to fail the way a missing Kakao POI fails."""

    provider = ContextMapProvider([RADIUS_CONTEXT])

    assert provider.search_place("IBK기업은행") == []
    # And a name is not a reference: the retrieval refuses it rather than searching behind the
    # tool call, the same way `KakaoMapProvider` does.
    with pytest.raises(PlaceNotFoundError):
        provider.nearby_search("IBK기업은행")
    assert provider.cache_miss_count == 1


def test_the_corpus_is_shared_by_every_question_the_way_upstreams_cache_is() -> None:
    """One database over the whole benchmark, not a per-question oracle.

    Upstream builds `context_cache.db` from the entire MapEval-Textual corpus and every question
    queries all of it. Scoping it per question would make the mere existence of a name an answer
    signal: a distractor invented for one question's options is absent from that question's own
    context but present in the corpus, exactly as a real map holds places that are not the answer.
    """

    provider = ContextMapProvider([NEARBY_CONTEXT, RADIUS_CONTEXT, ROUTE_CONTEXT])

    assert [place.name for place in provider.search_place("어라운드북")] == ["어라운드북"]
    assert [place.name for place in provider.search_place("우리은행")] == ["우리은행"]
    burger, store = _ref(provider, "회기버거"), _ref(provider, "GS25 서대문남가좌점")
    assert provider.directions(burger, store).distance_m == 4994


def test_a_retrieval_is_computed_over_the_corpus_not_replayed_from_a_block() -> None:
    """The corpus is a place database, not an answer sheet.

    A MapEval context stores the result of the query its question asks — already filtered by type,
    already sorted by distance — so replaying that block hands the agent the answer for one tool
    call and makes the benchmark indistinguishable from MapEval-Textual. The block's places are
    kept; the ranking is computed here, over every place the corpus holds.
    """

    provider = ContextMapProvider([NEARBY_CONTEXT, RADIUS_CONTEXT])

    # The radius applies to every retrieval, because it is applied rather than looked up.
    market = _ref(provider, "평화시장")
    near = provider.nearby_search(market, query="서점", radius_m=190)
    assert [place.name for place in near] == ["어라운드북"]
    assert [place.name for place in provider.nearby_search(market, query="서점")] == [
        "어라운드북",
        "교보문고",
        "소요서가",
    ]

    # A place from another question's context is a neighbour like any other when it is one.
    house = _ref(provider, "미소의집")
    assert [place.name for place in provider.nearby_search(house, query="은행")] == [
        "우리은행"
    ]


def test_a_retrieval_filters_by_type_in_whichever_vocabulary_the_caller_speaks() -> None:
    provider = ContextMapProvider([NEARBY_CONTEXT, RADIUS_CONTEXT])

    market = _ref(provider, "평화시장")
    by_noun = provider.nearby_search(market, query="서점", radius_m=20000)
    by_code = provider.nearby_search(market, category_code="BK9", radius_m=20000)
    by_token = provider.nearby_search(market, query="book_store", radius_m=20000)
    assert [place.name for place in by_noun] == [place.name for place in by_token]
    assert "우리은행" in [place.name for place in by_code]
    assert "어라운드북" not in [place.name for place in by_code]

    # A type this corpus does not record is not evidence of absence, so the neighbourhood answers.
    unknown = provider.nearby_search(_ref(provider, "평화시장"), query="병원", radius_m=20000)
    assert [place.name for place in unknown][0] == "어라운드북"


def test_a_place_is_not_among_its_own_neighbours() -> None:
    """The anchor stands at zero metres from itself and would head every ranking it appears in."""

    provider = ContextMapProvider([NEARBY_CONTEXT])
    found = provider.nearby_search(_ref(provider, "평화시장"), query="서점", radius_m=20000)
    assert "평화시장" not in [place.name for place in found]


def test_a_type_word_does_not_resolve_to_the_place_that_ends_in_it() -> None:
    """Otherwise a place-type question answers itself: the option would find its own anchor."""

    provider = ContextMapProvider([
        "Information of <b>다모아편의점</b>:\n"
        "- Location: 주소 정보 없음(37.5658790, 127.0036030).\n"
        "- OSM ID: node/368948685.\n"
        "- Type: convenience_store.\n"
    ])
    assert provider.search_place("편의점") == []
    assert [place.name for place in provider.search_place("다모아편의점")] == ["다모아편의점"]


def test_a_brand_resolves_to_the_branch_that_extends_it() -> None:
    provider = ContextMapProvider([
        "Information of <b>CU 삼청점</b>:\n"
        "- Location: 서울특별시 종로구 삼청로 68(37.5828000, 126.9810000).\n"
        "- OSM ID: node/1.\n"
    ])
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

    provider = ContextMapProvider([
        "Information of <b>GS25</b>:\n"
        "- Location: 주소 정보 없음(37.5600000, 126.9300000).\n"
        "- OSM ID: node/2.\n"
    ])
    assert provider.search_place("GS25 합정프리미엄점") == []
    assert [place.name for place in provider.search_place("GS25")] == ["GS25"]


def test_a_recorded_route_is_returned_and_an_unrecorded_one_fails() -> None:
    provider = ContextMapProvider([ROUTE_CONTEXT])

    route = provider.directions(_ref(provider, "회기버거"), _ref(provider, "GS25 서대문남가좌점"))
    assert (route.distance_m, route.duration_s) == (4994, 420)
    with pytest.raises(RouteNotFoundError):
        provider.directions(_ref(provider, "GS25 서대문남가좌점"), _ref(provider, "회기버거"))
    with pytest.raises(UnsupportedTravelModeError):
        provider.directions(
            _ref(provider, "회기버거"), _ref(provider, "GS25 서대문남가좌점"), mode="walking"
        )
    assert provider.api_call_count == 0


def test_the_shared_tool_layer_reads_the_context_like_any_other_provider() -> None:
    """Both architectures reach the context only through the registry, unchanged."""

    provider = ContextMapProvider([NEARBY_CONTEXT])
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
    # The centre is the place id `batch_geocode` just handed back, not the name again: a place
    # argument is a reference the provider issued.
    nearby = registry.invoke(
        "nearby_places",
        {"center": execution.output[0]["place"]["place_id"], "query": "서점"},
    )
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


def test_a_place_the_agent_already_holds_is_addressable_by_id_and_by_coordinates() -> None:
    """Both are references this provider handed out itself, so it has to take them back.

    A ReAct run searched 커피바이, read the place_id off the result, asked what was near it, and
    got PlaceNotFoundError; it then tried the coordinates and got the same. Two of its remaining
    steps went on re-searching a name that was never a name, and the question was lost.
    """

    provider = ContextMapProvider([RADIUS_CONTEXT])

    by_id = provider.nearby_search("node/12192225021", query="은행", radius_m=500)
    by_point = provider.nearby_search("37.5021773,126.9877564", query="은행", radius_m=500)
    assert [place.name for place in by_id] == ["우리은행"]
    assert [place.name for place in by_point] == [place.name for place in by_id]
    assert [place.name for place in provider.search_place("node/12192225021")] == ["미소의집"]

    # A point the corpus does not describe is still a point, and what is near it is answerable.
    anonymous = provider.nearby_search("37.5021000,126.9870000", query="은행", radius_m=500)
    assert [place.name for place in anonymous] == ["우리은행"]
    assert provider.api_call_count == 0


def test_a_planners_place_shape_never_ends_a_run() -> None:
    """Execution is lenient about shape and strict only about evidence — in the tools too.

    The local operators already normalized these; the tool arguments did not, so the same planner
    artifact failed as a pydantic ValidationError before any tool ran.
    """

    provider = ContextMapProvider([NEARBY_CONTEXT])
    registry = ToolRegistry(provider)
    anchor = registry.invoke("batch_geocode", {"place_names": ["평화시장"]}).output[0]["place"]

    shapes: list[object] = [
        [anchor],  # the geocode result the planner forgot to index into
        {"place": anchor},  # the wrapper that carries the place
        dict(anchor, candidate_index=0, distance_m=12.5, rank=0),  # enriched by an earlier step
    ]
    for shape in shapes:
        execution = registry.invoke("nearby_places", {"center": shape, "query": "서점"})
        assert execution.status == "ok", execution.error
        assert execution.output[0]["name"] == "어라운드북"

    # A planner that writes the anchor's name where a place belongs is naming the same place.
    recovered = registry.invoke(
        "recover_option_places",
        {"options": ["소요서가"], "candidates": [], "anchor": "평화시장", "radius_m": 5000},
    )
    assert recovered.status == "ok", recovered.error
    assert [place["name"] for place in recovered.output] == ["소요서가"]


class StubLiveProvider(MapProvider):
    """Stands in for Kakao: whatever the corpus does not hold, this does."""

    def __init__(self) -> None:
        self.calls = 0
        self.place = Place(
            place_id="live/1", name="IBK기업은행", latitude=37.5000, longitude=126.9800
        )

    @property
    def api_call_count(self) -> int:
        return self.calls

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        self.calls += 1
        return [self.place.model_copy(update={"name": query})]

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        return self.search_place(address, limit=limit)

    def nearby_search(self, center, **kwargs) -> list[Place]:
        self.calls += 1
        return [self.place]

    def place_details(self, place_id: str) -> Place:
        self.calls += 1
        return self.place

    def directions(self, origin, destination, **kwargs) -> Route:
        self.calls += 1
        return Route(origin="a", destination="b", distance_m=1, duration_s=1)


def test_a_miss_falls_back_to_the_live_provider_the_way_upstream_does() -> None:
    """Upstream's operators query the cache first and call Google Maps when it has nothing.

    `query_local_place` returns None on a miss and the geocode API runs instead, which is what
    lets one corpus-wide cache be used without its coverage bounding the benchmark.
    """

    live = StubLiveProvider()
    provider = ContextMapProvider([RADIUS_CONTEXT], fallback=live)

    assert [place.name for place in provider.search_place("미소의집")] == ["미소의집"]
    assert live.calls == 0  # the corpus answered, so nothing was spent

    assert [place.name for place in provider.search_place("IBK기업은행")] == ["IBK기업은행"]
    assert live.calls == 1
    assert provider.api_call_count == 1
    assert provider.cache_miss_count == 1

    # Without a fallback the miss is the answer, and a missing place stays missing.
    alone = ContextMapProvider([RADIUS_CONTEXT])
    assert alone.search_place("IBK기업은행") == []
    assert alone.api_call_count == 0
