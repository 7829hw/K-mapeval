"""Several places written into one string are several names, for the argument that takes several.

Measured on `dataset/seoul_kmapeval_v9_mcq_300.jsonl`: five of the ten `provider_failure` rows
were a composite arriving where a name belongs -- three option lists
(`'CU 마곡아르디에점, CU 강서발산역점, GS25 강서등촌점, GS25 마곡루체점'`) and two itineraries
(`'쥬인스테이 → 삼모아트센터'`). `place_names` is the plural argument, so what was meant is not in
doubt; a single-name argument keeps failing, because there the composite is genuinely ambiguous.
"""

from __future__ import annotations

import pytest

from src.tools.registry import BatchGeocodeArgs


@pytest.mark.parametrize(
    ("written", "meant"),
    [
        (
            "CU 마곡아르디에점, CU 강서발산역점, GS25 강서등촌점, GS25 마곡루체점",
            ["CU 마곡아르디에점", "CU 강서발산역점", "GS25 강서등촌점", "GS25 마곡루체점"],
        ),
        ("쥬인스테이 → 삼모아트센터", ["쥬인스테이", "삼모아트센터"]),
        ("A -> B", ["A", "B"]),
    ],
)
def test_a_listed_string_becomes_the_names_it_lists(written: str, meant: list[str]) -> None:
    assert BatchGeocodeArgs(place_names=[written]).place_names == meant


@pytest.mark.parametrize(
    "name",
    [
        # Every Kakao name in `data/seoul_kakao_pool.json` that carries a comma. All seven write
        # it without a following space, which is what the separator requires.
        "종로5,6가동 주민센터",
        "종로1,2,3,4가동 주민센터",
        "금호2,3가동주민센터",
        "7,900파스타 용두역점",
        "중계2,3동주민센터",
        "상계6동1,2,3단지주변 노상공영주차장 거주자우선주차",
        "오후,네시",
    ],
)
def test_a_real_name_carrying_a_comma_survives(name: str) -> None:
    assert BatchGeocodeArgs(place_names=[name]).place_names == [name]


def test_an_ordinary_batch_is_untouched() -> None:
    names = ["극단 리틀드럼", "겸재한신길 골목형상점가"]
    assert BatchGeocodeArgs(place_names=names).place_names == names
