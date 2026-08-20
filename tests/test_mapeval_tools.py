"""The five MapEval-API tools, now served by Kakao.

What is pinned here is what the baseline is *measured on*: the tool names and argument schemas
the agent sees, the place-id threading between the tools, and the exact text each observation
returns. Upstream's formatters were kept verbatim, so a change to any of these strings is a
change to the task, not to the implementation.
"""

from __future__ import annotations

import httpx
import pytest

from src.kakao_maps import KakaoAuthError, KakaoMapsClient
from src.mapeval_api.FormattedTools import (
    DirectionsTool,
    NearbyPlacesTool,
    PlaceDetailsTool,
    PlaceSearchTool,
    TravelTimeTool,
    set_client,
)
from tests.test_kakao_maps import ROUTE, STARBUCKS, TWOSOME, build_client


@pytest.fixture(autouse=True)
def _no_client_leak():
    yield
    set_client(None)


def bind(routes) -> KakaoMapsClient:
    client, _ = build_client(routes)
    set_client(client)
    return client


# ------------------------------------------------------------------ the tool set


def test_the_agent_sees_exactly_the_five_tools_evaluator2_constructs() -> None:
    """`Evaluator2.py` line 33 instantiates these five and no others.

    `Tools.py` also defines a `PlaceIdTool` the evaluator never constructs, and
    `PlaceSearchTool` is itself documented as "Get place ID for a given location" — the two
    are one primitive under two names, not a sixth tool. Adding one here is a claim about
    upstream, so make it with that line in front of you.
    """

    tools = [
        PlaceSearchTool(),
        PlaceDetailsTool(),
        NearbyPlacesTool(),
        TravelTimeTool(),
        DirectionsTool(),
    ]

    assert [tool.name for tool in tools] == [
        "PlaceSearch",
        "PlaceDetails",
        "NearbyPlaces",
        "TravelTime",
        "Directions",
    ]
    assert [tool.description for tool in tools] == [
        "Get place ID for a given location.",
        "Get details for a given place ID.",
        "Get nearby places around a location.",
        "Estimate the travel time between two places.",
        "Get directions/routes between two places.",
    ]


def test_the_argument_schemas_are_upstreams() -> None:
    assert set(PlaceSearchTool().args) == {"placeName"}
    assert set(PlaceDetailsTool().args) == {"placeId"}
    assert set(NearbyPlacesTool().args) == {"placeId", "type", "rankby", "radius"}
    assert set(TravelTimeTool().args) == {"originId", "destinationId", "travelMode"}
    assert set(DirectionsTool().args) == {"originId", "destinationId", "travelMode"}


# ------------------------------------------------------------------ PlaceSearch


def test_place_search_returns_a_reference_and_nothing_else() -> None:
    """Upstream is `return data['results'][0]['place_id']`.

    A baseline that wants coordinates pays a second round trip through PlaceDetails, and that
    asymmetry against Spatial-Agent's richer tools is the paper's own arrangement. Returning
    the whole place here would quietly hand the baseline a capability upstream withheld.
    """

    bind([("search/keyword", {"documents": [STARBUCKS]})])

    assert PlaceSearchTool()._run("스타벅스 강남점") == "26338954"


def test_a_name_kakao_does_not_carry_gets_upstreams_own_message() -> None:
    bind([])

    assert (
        PlaceSearchTool()._run("존재하지않는가게")
        == "Incorrect place name. Please use the same name as in the question."
    )


# ------------------------------------------------------------------ PlaceDetails


def test_place_details_reports_what_kakao_publishes_and_no_more() -> None:
    """`place_to_context` prints a rating, a price level and opening hours when the place
    carries them. Kakao Local publishes none of the three, so those lines are absent — the
    formatter is unchanged and its conditions are simply false."""

    bind([("search/keyword", {"documents": [STARBUCKS]})])
    place_id = PlaceSearchTool()._run("스타벅스 강남점")

    text = PlaceDetailsTool()._run(place_id)

    assert "- Location: 서울 강남구 강남대로 390 (37.49794, 127.02758).\n" in text
    assert "- Phone Number: 02-555-1234.\n" in text
    assert "Rating" not in text
    assert "Price Level" not in text
    assert "Open:" not in text


def test_an_id_the_baseline_never_searched_for_is_refused() -> None:
    bind([])

    assert PlaceDetailsTool()._run("nope") == "Incorrect Place ID. Please use correct place id."


# ------------------------------------------------------------------ NearbyPlaces


def test_nearby_places_lists_each_place_with_the_id_the_next_tool_needs() -> None:
    """The listing is how the baseline threads a place into TravelTime or Directions."""

    bind(
        [
            ("search/keyword", {"documents": [STARBUCKS]}),
            ("search/category", {"documents": [TWOSOME]}),
        ]
    )
    anchor = PlaceSearchTool()._run("스타벅스 강남점")

    text = NearbyPlacesTool()._run(anchor, "cafe")

    assert text.startswith("Nearby Cafes are (sorted by distance in ascending order):\n")
    assert "1. 투썸플레이스 역삼점 (11111111)\n" in text
    assert "   - Address: 서울 강남구 테헤란로 152.\n" in text


def test_a_place_is_not_among_its_own_neighbours() -> None:
    """The anchor stands at zero metres from itself and would head every ranking it appears
    in — and a nearest-cafe question asked from a cafe lists that cafe."""

    bind(
        [
            ("search/keyword", {"documents": [STARBUCKS]}),
            ("search/category", {"documents": [STARBUCKS, TWOSOME]}),
        ]
    )
    anchor = PlaceSearchTool()._run("스타벅스 강남점")

    text = NearbyPlacesTool()._run(anchor, "cafe")

    assert "스타벅스 강남점" not in text
    assert "1. 투썸플레이스 역삼점" in text


def test_upstreams_rankby_and_radius_rule_is_kept() -> None:
    bind([])

    assert NearbyPlacesTool()._run("x", "cafe", "distance", 500).startswith(
        "When rankby is distance, radius is disallowed."
    )


# ------------------------------------------------------------------ TravelTime


def test_travel_time_reports_one_duration_and_one_distance() -> None:
    """Upstream's TravelTimeTool reports a duration and a distance; DirectionsTool is the one
    that prints guidance. The two differ by what they report, and that is preserved."""

    bind([("search/keyword", {"documents": [STARBUCKS, TWOSOME]}), ("directions", ROUTE)])
    origin = PlaceSearchTool()._run("스타벅스 강남점")

    text = TravelTimeTool()._run(origin, "11111111", "driving")

    assert text == "Travel Time by car is 15 mins (5.3 km)."
    assert "left" not in text


# ------------------------------------------------------------------ Directions


def test_directions_prints_every_step_the_way_upstream_does() -> None:
    bind([("search/keyword", {"documents": [STARBUCKS, TWOSOME]}), ("directions", ROUTE)])
    origin = PlaceSearchTool()._run("스타벅스 강남점")

    text = DirectionsTool()._run(origin, "11111111", "driving")

    assert text.startswith("There are 1  routes by car. They are:\n")
    assert "1. Via 강남대로 | 15 mins | 5.3 km\n" in text
    assert " - 왕십리로 방면으로 좌회전\n" in text
    assert " - 직진\n" in text


# ------------------------------------------------------------------ travel modes


@pytest.mark.parametrize("tool", [TravelTimeTool(), DirectionsTool()])
@pytest.mark.parametrize("mode", ["walking", "transit", "bicycling"])
def test_a_mode_kakao_cannot_serve_is_refused_in_words(tool, mode) -> None:
    """Kakao Mobility routes cars only. Answering a walking question with a driving route
    would answer a different question, so the tool says what is available instead."""

    bind([("directions", ROUTE)])

    text = tool._run("a", "b", mode)

    assert "Kakao Mobility serves driving routes only" in text


# ------------------------------------------------------------------ failures


def test_a_rejected_key_is_not_reported_as_a_missing_place() -> None:
    """Upstream's tools swallowed every failure into an "incorrect place" observation. Against
    a bad key that turns a whole run into confidently unanswerable questions instead of one
    loud failure, so a configuration error is allowed out."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errorType": "e"})

    set_client(
        KakaoMapsClient(
            "bad", client=httpx.Client(transport=httpx.MockTransport(handler)), cache_path=""
        )
    )

    with pytest.raises(KakaoAuthError):
        PlaceSearchTool()._run("스타벅스 강남점")
