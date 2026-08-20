"""Shared machinery for building the Kakao-grounded Korean benchmark.

Every gold answer is computed from the same provider the agents query, so a wrong answer in a run
means the agent, never a source mismatch between the evidence and the grader.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data._toolkit.kakao import KakaoMapProvider  # noqa: E402
from data._toolkit.models import Place, Route, RouteStep  # noqa: E402
from data._toolkit.spatial import haversine_meters  # noqa: E402
from src.config import Settings  # noqa: E402

__all__ = [  # re-exported so the generator imports one module
    "CATEGORY_NOUNS",
    "Builder",
    "Place",
    "Route",
    "RouteStep",
    "cardinal",
    "distance_m",
    "eul",
    "eun",
    "euro",
    "iga",
    "load_pool",
    "plausible_name",
    "take_resolvable",
    "to_place",
    "wa",
]

POOL_PATH = Path(__file__).with_name("seoul_kakao_pool.json")

# A name an agent cannot look back up is not a usable option, however good the geometry is.
ROUND_TRIP_TOLERANCE_M = 200.0

CATEGORY_NOUNS = {
    "AT4": "관광명소",
    "CT1": "문화시설",
    "AD5": "숙박시설",
    "SW8": "지하철역",
    "CE7": "카페",
    "FD6": "음식점",
    "MT1": "대형마트",
    "CS2": "편의점",
    "BK9": "은행",
    "PO3": "공공기관",
    "SC4": "학교",
    "HP8": "병원",
    "PK6": "주차장",
}

# Pool names that are route descriptions, administrative labels or walking courses rather than
# places a question can send someone to.
NAME_REJECT_TOKENS = (
    "코스", "둘레길", "자락길", "산책로", "명예도로", "문화의거리", "먹자골목",
    "출입구", "주차장", "입구", "방면", "정류장", "환승", "본점영업부",
)


def load_pool() -> list[dict[str, Any]]:
    return json.loads(POOL_PATH.read_text(encoding="utf-8"))


def to_place(record: dict[str, Any]) -> Place:
    skip = ("category_code", "district")
    payload = {key: value for key, value in record.items() if key not in skip}
    return Place(**payload)


def distance_m(a: Place, b: Place) -> float:
    return haversine_meters(a.latitude, a.longitude, b.latitude, b.longitude)


def bearing_degrees(a: Place, b: Place) -> float:
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    delta = math.radians(b.longitude - a.longitude)
    y = math.sin(delta) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def cardinal(a: Place, b: Place) -> str:
    degrees = bearing_degrees(a, b)
    index = int((degrees + 22.5) % 360.0 // 45.0)
    return ("북", "북동", "동", "남동", "남", "남서", "서", "북서")[index]


def particle(word: str, with_final: str, without_final: str) -> str:
    """Pick the Korean particle a name's last syllable takes, so questions read naturally."""

    last = word.strip()[-1:]
    if not last or not ("\uac00" <= last <= "\ud7a3"):
        return without_final
    return with_final if (ord(last) - 0xAC00) % 28 else without_final


def eul(word: str) -> str:
    return word + particle(word, "\uc744", "\ub97c")


def eun(word: str) -> str:
    return word + particle(word, "\uc740", "\ub294")


def iga(word: str) -> str:
    return word + particle(word, "\uc774", "\uac00")


def wa(word: str) -> str:
    return word + particle(word, "\uacfc", "\uc640")


def euro(word: str) -> str:
    last = word.strip()[-1:]
    if last and "\uac00" <= last <= "\ud7a3" and (ord(last) - 0xAC00) % 28 in (0, 8):
        return word + "\ub85c"
    return word + "\uc73c\ub85c"


def plausible_name(name: str) -> bool:
    if not 2 <= len(name) <= 18:
        return False
    return not any(token in name for token in NAME_REJECT_TOKENS)


def take_resolvable(builder: Builder, places: Iterable[Place], wanted: int) -> list[Place]:
    """Take the first `wanted` names that survive round-trip resolution, and stop.

    Slicing after a comprehension (`[p for p in candidates if resolves(p)][:3]`) looks equivalent
    and is not: it round-trips every candidate before discarding all but three, which on a
    city-wide candidate list is thousands of Kakao calls for three names.
    """

    taken: list[Place] = []
    for place in places:
        if len(taken) == wanted:
            break
        if builder.resolves_to(place):
            taken.append(place)
    return taken


@dataclass
class Builder:
    """A Kakao session with memoized lookups, so a rejected candidate costs one call at most."""

    provider: KakaoMapProvider
    _resolved: dict[str, Place | None] = field(default_factory=dict)
    _routes: dict[tuple[str, str, tuple[str, ...], str], Route | None] = field(
        default_factory=dict
    )

    @classmethod
    def open(cls) -> Builder:
        settings = Settings()
        provider = KakaoMapProvider(
            settings.kakao_rest_api_key,
            timeout=settings.kakao_timeout_seconds,
            cache_path=settings.kakao_cache_db_path,
            cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
            search_center=settings.search_center(),
            search_radius_m=settings.kakao_search_radius_m,
        )
        return cls(provider=provider)

    def close(self) -> None:
        self.provider.close()

    def resolves_to(self, place: Place) -> bool:
        """True when searching the bare name lands back on this place.

        An option text is only usable if the agent can turn it back into the place the gold was
        computed from. Names that Kakao answers with a different branch are dropped here rather
        than being scored as agent failures later.
        """

        cached = self._resolved.get(place.name, "missing")
        if cached == "missing":
            try:
                found = self.provider.search_place(place.name, limit=1)
            except Exception:  # noqa: BLE001 - an unresolvable name is simply not usable
                found = []
            resolved = found[0] if found else None
            self._resolved[place.name] = resolved
            cached = resolved
        if cached is None:
            return False
        return distance_m(cached, place) <= ROUND_TRIP_TOLERANCE_M

    def as_resolved(self, place: Place) -> Place | None:
        """The place an agent gets when it searches this name, not the one the pool holds.

        The two may sit up to `ROUND_TRIP_TOLERANCE_M` apart, and a question asked about the pool's
        copy is then measured from a point no agent stands on: one 피부과 question put its gold at
        153 m from the pool's 신사동가로수길 while the resolved anchor had another clinic at 93 m.
        A question about an anchor is a question about whatever that name resolves to.
        """

        if not self.resolves_to(place):
            return None
        return self._resolved.get(place.name)

    def route(
        self,
        origin: Place,
        destination: Place,
        waypoints: tuple[Place, ...] = (),
        priority: str = "DISTANCE",
    ) -> Route | None:
        """Route a pair, choosing the priority by what the answer is made of.

        Only DISTANCE is traffic-invariant — a shortest-path over the road graph — so a distance
        gold is built from it and is a fact about the network. A duration cannot borrow that
        stability: the shortest route is not the one anyone drives, and on a four-leg chain its
        duration ran 160 minutes against TIME's 96. So a duration gold is built with the priority
        its consumer uses (`calculate_finish_time` defaults to TIME) and stays a live estimate;
        the same fixed route came back as 3,243 s and then 4,337 s. Questions resting on a
        duration must space their options wider than that spread.
        """

        key = (
            origin.place_id,
            destination.place_id,
            tuple(p.place_id for p in waypoints),
            priority,
        )
        if key not in self._routes:
            try:
                self._routes[key] = self.provider.directions(
                    origin,
                    destination,
                    waypoints=list(waypoints) or None,
                    include_steps=True,
                    priority=priority,
                )
            except Exception:  # noqa: BLE001 - an unroutable pair is simply not usable
                self._routes[key] = None
        return self._routes[key]

    def duration_s(self, origin: Place, destination: Place) -> int | None:
        found = self.route(origin, destination, priority="TIME")
        return None if found is None else found.duration_s

    def distance_m_driving(self, origin: Place, destination: Place) -> int | None:
        found = self.route(origin, destination, priority="DISTANCE")
        return None if found is None else found.distance_m
