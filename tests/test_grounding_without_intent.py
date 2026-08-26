"""Grounding reads what the question states, not what a classifier called the question.

`_ground_graph_literals` used to take `analysis["intent"]` and gate six branches on it: a radius
was read only from a question labelled `radius`, a compared pair only from one labelled
`distance`, a kind of place only from `nearby`/`direction`/`radius`, an anchor from a per-intent
pattern table, and the option splice from `nearby`/`direction`/`routing`.

That label is a guess, and the recorded runs show how often it misses. Over 849 v7_300 graphs the
Analysis stage put 21 of 90 `nearby_subtype_kth` questions under `poi`, 53 of 72
`routing_detour_cost` under `poi`, and 42 of 63 `routing_turn_count_via` under `trip`. Each of
those lost the branch its question needed -- silently, because grounding does not fail when it
declines to bind something.

`GroundingFacts` is that reading, taken once. Presence is the gate: a fact the question does not
state is `None` and its branch does not run.
"""

from __future__ import annotations

import pytest

from src.agent.spatial import (
    GroundingFacts,
    _extract_anchor,
    _extract_compared_places,
    _extract_radius_m,
    _extract_target_type,
    _ground_graph_literals,
    _ranks_the_options,
    extract_facts,
)


def test_the_facts_are_the_same_whatever_the_analysis_called_the_question() -> None:
    question = (
        "서울생활사박물관 별관동에서 직선거리 600m 이내에 있는 대형마트는 다음 중 어디인가요?"
    )
    expected = GroundingFacts(
        anchor="서울생활사박물관 별관동",
        target_type="대형마트",
        radius_m=600,
    )

    for mislabelled in ({}, {"intent": "poi"}, {"intent": "trip"}, {"intent": "radius"}):
        assert extract_facts(mislabelled, question) == expected


def test_a_question_that_states_nothing_states_nothing() -> None:
    facts = extract_facts({}, "다음 중 가장 분위기가 좋은 곳은 어디인가요?")

    assert facts.radius_m is None
    assert facts.direction is None
    assert facts.compared_pair is None
    assert facts.target_type is None
    assert facts.returns_to_start is False
    assert facts.stated_order is False


def test_the_inferred_kind_fills_in_only_where_the_question_names_none() -> None:
    """A need-shaped question names no category; that inference is the Analysis stage's job."""

    need = "지금 단막극장에 있습니다. 우산을 사야 합니다. 가장 가까운 곳은?"
    assert _extract_target_type(need) is None
    assert extract_facts({"target_type": "편의점"}, need).target_type == "편의점"

    stated = "서울역에서 가장 가까운 약국은 어디인가요?"
    assert extract_facts({"target_type": "편의점"}, stated).target_type == "약국"


def test_a_literal_the_scan_misses_can_still_be_recovered_from_the_concept_graph() -> None:
    """The recovery path, and its priority: the question wins when the question says it."""

    unknown_phrasing = "서울역 언저리 육백미터 남짓 안의 약국은?"
    assert _extract_radius_m(unknown_phrasing) is None
    analysis = {
        "concepts": [
            {"id": "c1", "role": "sub_condition", "attributes": {"radius_m": "600m"}},
            {"id": "c2", "role": "sub_condition", "attributes": {"direction": "북동쪽"}},
        ]
    }
    recovered = extract_facts(analysis, unknown_phrasing)
    assert recovered.radius_m == 600
    assert recovered.direction == "북동쪽"

    written = "서울역에서 직선거리 300m 이내에 있는 남쪽 약국은?"
    assert extract_facts(analysis, written).radius_m == 300
    assert extract_facts(analysis, written).direction == "남쪽"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("자양2동문고에서 초원책서점까지의 직선거리는 얼마인가요?", ("자양2동문고", "초원책서점")),
        ("A 및 B 사이의 직선거리는 몇 m인가요?", ("A", "B")),
        # Three places, two separations. Reading the first pair off this and binding
        # `place_names` to it deletes the third place -- the question then measures one of the
        # two distances it asked to compare.
        (
            "방도에서 GS더프레시 신길사러가점까지의 직선거리와 방도에서 노브랜드 광명소하점까지의 "
            "직선거리는 얼마나 차이가 나나요?",
            None,
        ),
        ("서울역에서 가장 가까운 약국은 어디인가요?", None),
    ],
)
def test_a_pair_is_two_places_and_two_places_only(
    question: str, expected: tuple[str, str] | None
) -> None:
    assert _extract_compared_places(question) == expected


def test_a_vicinity_word_is_not_part_of_the_anchors_name() -> None:
    """"헤이갤러리 근처에서 …" names 헤이갤러리.

    The per-intent tables never reached these questions, so the bare "에서" split that now runs
    for every question had to learn what they say. Binding "헤이갤러리 근처" geocodes a place
    that does not exist -- and writes it over the option the plan had already listed.
    """

    vicinity = "헤이갤러리 근처에서 분위기가 가장 좋은 카페는 어디인가요?"
    assert _extract_anchor(vicinity) == "헤이갤러리"
    assert _extract_anchor("동묘앞역 6호선 인근에서 평이 좋은 곳은?") == "동묘앞역 6호선"
    assert _extract_anchor("서울역에서 가장 가까운 약국은?") == "서울역"


_BATCH = {
    "id": "places",
    "operator": "batch_geocode",
    "arguments": {"place_names": ["가", "나", "다"]},
    "depends_on": [],
    "role": "extent",
}


def _with(consumer: str) -> list[dict[str, object]]:
    return [
        _BATCH,
        {
            "id": "next",
            "operator": consumer,
            "arguments": {"value": "$places"},
            "depends_on": ["places"],
            "role": "measure",
        },
    ]


@pytest.mark.parametrize(
    ("consumer", "ranks"),
    [
        ("match_options", True),
        ("nearest", True),
        ("filter_by_direction", True),
        ("recover_option_places", True),
        ("tsp_tw", False),
        ("calculate_finish_time", False),
        ("identity_measure", False),
    ],
)
def test_what_a_batch_of_names_is_for_is_read_off_the_graph(consumer: str, ranks: bool) -> None:
    assert _ranks_the_options(_with(consumer), "places") is ranks


def test_a_dependency_written_only_as_a_reference_still_counts() -> None:
    """Planners write `depends_on` and `$node` inconsistently; either one is the dataflow."""

    steps = [
        _BATCH,
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {"nodes": "$places"},
            "depends_on": [],
            "role": "measure",
        },
    ]

    assert _ranks_the_options(steps, "places") is False


def test_a_trip_plan_keeps_its_stays_whatever_the_question_was_labelled() -> None:
    """The A-class removal, stated as a behaviour rather than as a diff.

    `tsp_tw` appears only in a trip plan, so the `intent == "trip"` conjunct in front of it never
    admitted a node the operator alone would not have. It could only exclude one -- and on the
    recorded runs the Analysis stage labelled 42 of 63 `routing_turn_count_via` graphs `trip` and
    plenty of trip graphs something else.
    """

    steps = [
        {
            "id": "places",
            "operator": "batch_geocode",
            "arguments": {"place_names": ["가예", "A", "B"]},
            "depends_on": [],
            "role": "extent",
        },
        {
            "id": "tour",
            "operator": "tsp_tw",
            "arguments": {"nodes": "$places", "distance_matrix": "$legs"},
            "depends_on": ["places"],
            "role": "measure",
        },
    ]
    question = (
        "가예에서 출발해 A를 1시간, B를 1.5시간 동안 적힌 순서대로 둘러본 뒤 가예로 돌아옵니다. "
        "자동차 총 주행거리가 가장 짧은 방문 순서는?"
    )
    grounded = _ground_graph_literals(
        steps, question, ["A → B", "B → A"], extract_facts({}, question)
    )
    tour = grounded[1]["arguments"]

    assert tour["metric"] == "distance"
    assert tour["return_to_start"] is True
    assert tour["fixed_order"] is True
