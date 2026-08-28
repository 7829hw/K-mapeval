"""Reconciling an answer the core wrote in base units with options a person wrote.

The core is blind to the options by design and answers in metres, seconds and counts. The
options say `약 1.5km` and `세 곳`. Text matching cannot bridge that, and 49 of one hundred
questions in one run ended `answer_parse_failure` holding a correct answer -- `1518.07 m`
against an option list whose gold was `약 1.5km`.
"""

from __future__ import annotations

import pytest

from src.agent.answering import GroundedAnswer
from src.mcq_adapter import MCQAdapter, parse_quantity


def _answer(value: object, text: str) -> GroundedAnswer:
    return GroundedAnswer(value=value, text=text, confidence=0.9, reason="")


def test_metres_reconcile_with_kilometre_options() -> None:
    selection = MCQAdapter().select(
        _answer(1518.07, "CGV 여의도까지의 직선거리는 지민숲까지보다 약 1518.07m 더 깁니다."),
        ["약 1.9km", "약 1.8km", "약 1.7km", "약 1.5km"],
    )
    assert selection.index == 3
    assert selection.method == "grounded_quantity"


def test_a_counted_answer_reconciles_with_a_korean_numeral_option() -> None:
    selection = MCQAdapter().select(
        _answer(3, "반경 300m 이내에 있는 은행은 3곳입니다."),
        ["한 곳", "두 곳", "세 곳", "네 곳"],
    )
    assert selection.index == 2


def test_an_answer_that_rounds_to_nothing_offered_stays_unresolved() -> None:
    """No least-bad match: a question the core answered wrongly has to stay wrong."""

    selection = MCQAdapter().select(
        _answer(1518.07, "약 1518.07m"), ["약 5.0km", "약 9.0km"]
    )
    assert selection.index is None
    assert selection.method == "unresolved"


def test_an_answer_landing_between_two_options_stays_unresolved() -> None:
    selection = MCQAdapter().select(_answer(1550.0, "약 1550m"), ["약 1.5km", "약 1.6km"])
    assert selection.index is None


def test_a_dimension_mismatch_is_not_a_match() -> None:
    selection = MCQAdapter().select(_answer(3, "3곳"), ["약 3km", "약 4km"])
    assert selection.index is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("약 1.5km", 1500.0),
        ("약 25.4km", 25400.0),
        ("1518.07m", 1518.07),
        ("세 곳", 3.0),
        ("두 곳", 2.0),
        ("45분", 2700.0),
    ],
)
def test_quantities_read_in_their_dimension_base_unit(text: str, expected: float) -> None:
    quantity = parse_quantity(text)
    assert quantity is not None
    assert quantity.value == pytest.approx(expected)


def test_a_string_stating_two_amounts_names_no_single_amount() -> None:
    assert parse_quantity("6640m와 8158m") is None


def test_the_text_methods_still_win_before_any_arithmetic() -> None:
    selection = MCQAdapter().select(
        _answer(25500.0, "총 이동 거리는 약 25.5km입니다."),
        ["약 25.4km", "약 25.5km", "약 26.1km", "약 24.9km"],
    )
    assert selection.index == 1
    assert selection.method == "grounded_text_containment"


def test_a_named_answer_matches_the_option_it_holds() -> None:
    """`명이비인후과 (서울 광진구 용마산로 5)` names the option `명이비인후과`, and the value was
    only ever tried for equality."""

    selection = MCQAdapter().select(
        _answer("명이비인후과 (서울 광진구 용마산로 5)", "세 번째로 가까운 곳입니다."),
        ["아이코이비인후과의원", "유앤장이비인후과의원", "김진홍치과의원", "명이비인후과"],
    )
    assert selection.index == 3
    assert selection.method == "grounded_value_containment"


def test_a_numeric_value_is_never_matched_by_containment() -> None:
    """`3` is a substring of `약 3km` and of nothing else, which is a unique match and a wrong
    one. Numbers go through the quantity rule, which knows dimensions."""

    selection = MCQAdapter().select(_answer(3, "3곳"), ["약 3km", "약 4km"])
    assert selection.index is None


def test_the_bare_value_is_read_in_the_unit_the_answer_actually_used() -> None:
    """The core writes 7.26 for kilometres in one question and 6552.89 for metres in the next,
    so the bare number is read both ways -- but only in a dimension the answer's own text names.
    """

    selection = MCQAdapter().select(
        _answer(
            7.2631,
            "CGV 중계까지의 직선거리(약 14.75km)와 논현역까지(약 7.49km)의 차이는 약 7.26km입니다.",
        ),
        ["약 8.1km", "약 7.3km", "약 8.7km", "약 9.3km"],
    )
    assert selection.index == 1


def test_a_value_written_as_a_string_with_its_unit_is_a_quantity() -> None:
    """The core writes `12490.86541977818 m` as often as it writes a number, and reading it beats
    reading the prose -- which holds `신길역 5호선` and every other number the answer mentions."""

    selection = MCQAdapter().select(
        _answer(
            "12490.86541977818 m",
            "신길역 5호선에서 양재천 벚꽃길까지의 직선거리가 가장 멀며, 약 12490.87m입니다.",
        ),
        ["약 16.0km", "주어진 지도 정보로는 알 수 없음", "약 12.5km", "약 9.7km"],
    )
    assert selection.index == 2
    assert selection.method == "grounded_quantity"


def test_an_ordering_is_matched_by_the_order_the_answer_names_it_in() -> None:
    """An ordering question offers permutations of one set of places, so the separator is all
    that stands between the answer's prose and the option's `A → B → C`."""

    selection = MCQAdapter().select(
        _answer(
            "32620.0 m",
            "총 주행거리가 가장 짧은 순서는 호텔브릿지에서 시립보라매청소년센터 다이나믹홀, "
            "소요한남 바이 파르나스 갤러리, 서학당길을 순서대로 방문한 후 돌아오는 경로입니다.",
        ),
        [
            "서학당길 → 소요한남 바이 파르나스 갤러리 → 시립보라매청소년센터 다이나믹홀",
            "시립보라매청소년센터 다이나믹홀 → 서학당길 → 소요한남 바이 파르나스 갤러리",
            "시립보라매청소년센터 다이나믹홀 → 소요한남 바이 파르나스 갤러리 → 서학당길",
            "소요한남 바이 파르나스 갤러리 → 서학당길 → 시립보라매청소년센터 다이나믹홀",
        ],
    )
    assert selection.index == 2
    assert selection.method == "grounded_ordering"


def test_an_answer_that_names_no_order_matches_no_ordering() -> None:
    selection = MCQAdapter().select(
        _answer(None, "경로를 계산할 수 없습니다."),
        ["A → B → C", "B → C → A", "C → A → B"],
    )
    assert selection.index is None


def test_a_declined_answer_reads_against_the_option_that_says_the_same() -> None:
    """This port's `unanswerable` families have `주어진 지도 정보로는 알 수 없음` as their gold.
    The core is blind to the options and declines in prose, and there was nothing to match it
    to -- those families read 0 of 7 in every run of this stack."""

    selection = MCQAdapter().select(
        _answer(None, "제공된 증거에는 평점 정보가 포함되어 있지 않아 판단할 수 없습니다."),
        ["김가네 상암점", "삼성웰스토리", "예쁜돼지 상암점", "주어진 지도 정보로는 알 수 없음"],
    )
    assert selection.index == 3
    assert selection.method == "grounded_decline"


def test_a_run_that_broke_is_not_an_answer_of_cannot_know() -> None:
    """The guard that keeps this from becoming the escape hatch `AGENTS.md` records: a step that
    raised means the run broke, not that the map lacks the answer."""

    selection = MCQAdapter().select(
        _answer(None, "계산에 실패했습니다."),
        ["김가네 상암점", "삼성웰스토리", "예쁜돼지 상암점", "주어진 지도 정보로는 알 수 없음"],
        execution_errors=2,
    )
    assert selection.index is None


def test_a_measurement_that_matched_nothing_is_not_a_decline() -> None:
    """Only *no value at all* counts. A value that failed to round to any option stays
    unresolved rather than becoming `알 수 없음`."""

    selection = MCQAdapter().select(
        _answer(1518.07, "약 1518.07m"),
        ["약 5.0km", "약 9.0km", "주어진 지도 정보로는 알 수 없음"],
    )
    assert selection.index is None


def test_the_written_unit_does_not_override_the_number() -> None:
    """One answer read `약 14474.72 km` for a distance of 14.5 km, having labelled metres as
    kilometres. The number is the measurement; the label is prose."""

    selection = MCQAdapter().select(
        _answer("14474.72 km", "세 곳 중 가장 먼 곳까지의 거리는 약 14474.72 km입니다."),
        ["약 18.6km", "약 13.4km", "약 14.5km", "약 17.4km"],
    )
    assert selection.index == 2
