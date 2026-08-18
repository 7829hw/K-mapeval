"""A map provider that answers from the benchmark's own contexts instead of a live API.

This is the port of upstream Spatial-Agent's local context cache. Upstream evaluates on
MapEval-API — multiple-choice questions with no context — and builds one SQLite database
(`data/build_cache.py` → `data/context_cache.db`) from the *whole* MapEval-Textual corpus, whose
rows carry the `context` field. Its operators query that database first and fall back to the Google
Maps API on a miss (`ContextManager.should_use_local_db` → `query_local_place` → geocode API). The
agent itself never sees the context text: `test_agent.py` does not mention it, and no agent module
reads it outside the database.

So the cache is a substitute *map database*, not a per-question oracle, and this module matches
that: one corpus built from every context in the dataset, shared by every question, optionally
backed by a live provider for what it does not hold. Scoping it per question would have made the
mere existence of a name an answer signal — a distractor invented for one question's options is
absent from that question's context but present in the corpus, exactly as a real map holds places
that are not the answer.

The evidence is loaded *behind* the tool layer either way, so both architectures still choose tools
and still read normalized `Place` / `Route` objects. What changes is only where those come from.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.models import Place, Route
from src.tools.map import (
    MapProvider,
    PlaceNotFoundError,
    RouteNotFoundError,
    UnsupportedTravelModeError,
)
from src.tools.spatial import (
    distinguishing_similarity,
    haversine_meters,
    parse_coordinate_literal,
    strip_location_qualifier,
)

# A context block writes "주소 정보 없음" where OSM carries no address for the place.
MISSING_ADDRESS = "주소 정보 없음"
# Same floor the Kakao path applies, for the same reason: a closed world of twenty places still has
# to refuse a name it does not hold rather than return whichever entry scored least badly.
NAME_MATCH_FLOOR = 0.55

INFO_HEADER = re.compile(r"^Information of <b>(?P<name>.+?)</b>:$")
NEARBY_HEADER = re.compile(
    r"^Nearby (?P<type>\S+) of (?P<anchor>.+?) are \((?P<qualifier>.*?)\):$"
)
ROUTE_HEADER = re.compile(r"^Route from (?P<origin>.+?) to (?P<destination>.+?) by (?P<mode>.+?):$")
NEARBY_ENTRY = re.compile(
    r"^\d+\.\s*<b>(?P<name>.+?)</b>\((?P<latitude>-?[\d.]+),\s*(?P<longitude>-?[\d.]+)\)$"
)
LOCATION_LINE = re.compile(
    r"^- Location:\s*(?P<address>.*?)\((?P<latitude>-?[\d.]+),\s*(?P<longitude>-?[\d.]+)\)\.?$"
)
ADDRESS_LINE = re.compile(r"^- Address:\s*(?P<address>.*?)\.?$")
OSM_ID_LINE = re.compile(r"^- OSM ID:\s*(?P<osm_id>.+?)\.?$")
TYPE_LINE = re.compile(r"^- Type:\s*(?P<type>.+?)\.?$")
DURATION_LINE = re.compile(r"^- Duration:\s*(?P<minutes>\d+)분\.?$")
DISTANCE_LINE = re.compile(r"^- Distance:\s*(?P<meters>\d+)m\.?$")

# The mode every context route is recorded under. Kept as a mapping rather than a constant so an
# unsupported mode fails as its own provider error, the way the Kakao provider fails one.
CONTEXT_TRAVEL_MODES = {"자동차": "driving"}


@dataclass(frozen=True)
class NearbyBlock:
    """One retrieval the context recorded for an anchor place."""

    anchor: str
    place_type: str
    places: tuple[Place, ...]
    # "in requested radius" blocks are already bounded by the radius the question asks about;
    # "sorted by distance" blocks are a k-nearest result with no radius at all.
    radius_bounded: bool


@dataclass(frozen=True)
class ContextDocument:
    """Everything one benchmark item states, parsed into provider-neutral objects."""

    places: tuple[Place, ...] = ()
    nearby: tuple[NearbyBlock, ...] = ()
    routes: tuple[Route, ...] = ()
    _by_id: dict[str, Place] = field(default_factory=dict)

    def all_places(self) -> list[Place]:
        seen: dict[str, Place] = {}
        for place in (*self.places, *(entry for block in self.nearby for entry in block.places)):
            seen.setdefault(place.place_id, place)
        return list(seen.values())


def _clean_address(value: str) -> str:
    address = value.strip().rstrip(".").strip()
    return "" if address in ("", MISSING_ADDRESS) else address


def _name_key(value: str) -> str:
    normalized = strip_location_qualifier(value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _place_key(latitude: float, longitude: float) -> str:
    return f"{latitude:.7f},{longitude:.7f}"


def parse_context(context: str) -> ContextDocument:
    """Parse one MapEval-style context block into places, nearby retrievals, and routes."""

    return parse_contexts([context])


def parse_contexts(contexts: Iterable[str]) -> ContextDocument:
    """Merge every context into one corpus, the way upstream's cache builder does.

    The format is fixed by the dataset generator: `Information of` blocks carry a place, `Nearby …`
    blocks carry one retrieval for an anchor, and `Route from … to … by …` blocks carry one route.
    Anything else in the text is ignored rather than guessed at. Two contexts describing the same
    coordinate describe the same place, and the fuller description wins.
    """

    places: dict[str, Place] = {}
    identifiers: dict[str, str] = {}
    blocks: list[NearbyBlock] = []
    routes: list[Route] = []

    def remember(
        name: str, latitude: float, longitude: float, address: str, category: str, osm_id: str
    ) -> Place:
        """Record a place, letting a fuller description of one coordinate replace a thinner one."""

        key = _place_key(latitude, longitude)
        place_id = identifiers.get(key) or osm_id or f"context/{len(identifiers) + 1}"
        identifiers[key] = place_id
        existing = places.get(place_id)
        place = Place(
            place_id=place_id,
            name=name or (existing.name if existing else name),
            address=address or (existing.address if existing else ""),
            latitude=latitude,
            longitude=longitude,
            category=category or (existing.category if existing else ""),
        )
        places[place_id] = place
        return place

    for context in contexts:
        _absorb(context, remember, blocks, routes)
    ordered = tuple(places[place_id] for place_id in dict.fromkeys(places))
    return ContextDocument(
        places=ordered,
        nearby=tuple(blocks),
        routes=tuple(routes),
        _by_id={place.place_id: place for place in ordered},
    )


def _absorb(
    context: str,
    remember: Callable[[str, float, float, str, str, str], Place],
    blocks: list[NearbyBlock],
    routes: list[Route],
) -> None:
    """Read one context block into the corpus under construction."""

    lines = context.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        info = INFO_HEADER.match(line)
        if info:
            latitude = longitude = None
            address = category = osm_id = ""
            while index < len(lines) and lines[index].strip().startswith("- "):
                detail = lines[index].strip()
                index += 1
                location = LOCATION_LINE.match(detail)
                if location:
                    latitude = float(location["latitude"])
                    longitude = float(location["longitude"])
                    address = _clean_address(location["address"])
                    continue
                identifier = OSM_ID_LINE.match(detail)
                if identifier:
                    osm_id = identifier["osm_id"].strip()
                    continue
                place_type = TYPE_LINE.match(detail)
                if place_type:
                    category = place_type["type"].strip()
            if latitude is not None and longitude is not None:
                remember(info["name"].strip(), latitude, longitude, address, category, osm_id)
            continue
        nearby = NEARBY_HEADER.match(line)
        if nearby:
            entries: list[Place] = []
            while index < len(lines):
                entry = NEARBY_ENTRY.match(lines[index].strip())
                if not entry:
                    if lines[index].strip():
                        break
                    index += 1
                    continue
                index += 1
                address = ""
                while index < len(lines) and lines[index].strip().startswith("- "):
                    matched = ADDRESS_LINE.match(lines[index].strip())
                    index += 1
                    if matched:
                        address = _clean_address(matched["address"])
                entries.append(
                    remember(
                        entry["name"].strip(),
                        float(entry["latitude"]),
                        float(entry["longitude"]),
                        address,
                        # A nearby block is retrieved by type, so every entry is of that type even
                        # though the block does not repeat it per place.
                        nearby["type"].strip().split("=")[-1],
                        "",
                    )
                )
            blocks.append(
                NearbyBlock(
                    anchor=nearby["anchor"].strip(),
                    place_type=nearby["type"].strip(),
                    places=tuple(entries),
                    radius_bounded="radius" in nearby["qualifier"],
                )
            )
            continue
        route = ROUTE_HEADER.match(line)
        if route:
            duration_s = distance_m = None
            while index < len(lines) and lines[index].strip().startswith("- "):
                detail = lines[index].strip()
                index += 1
                duration = DURATION_LINE.match(detail)
                if duration:
                    duration_s = int(duration["minutes"]) * 60
                    continue
                distance = DISTANCE_LINE.match(detail)
                if distance:
                    distance_m = int(distance["meters"])
            if duration_s is not None and distance_m is not None:
                routes.append(
                    Route(
                        origin=route["origin"].strip(),
                        destination=route["destination"].strip(),
                        distance_m=distance_m,
                        duration_s=duration_s,
                    )
                )


def _distance(anchor: Place, place: Place) -> float:
    return haversine_meters(anchor.latitude, anchor.longitude, place.latitude, place.longitude)


def _name_score(query: str, name: str) -> float:
    """How strongly a context name answers a query, on the Kakao path's evidence terms."""

    query_key = _name_key(query)
    name_key = _name_key(name)
    if not query_key or not name_key:
        return 0.0
    if query_key == name_key:
        return 1.0
    if query_key in name_key or name_key in query_key:
        # Containment is evidence in exactly one direction: a brand leads the branch that extends
        # it, so asking for CU may answer with CU 삼청점. The reverse does not hold — a context
        # that stores a bare GS25 is not the GS25 합정프리미엄점 an option names, it is whichever
        # GS25 the retrieval found. And a generic type word only ever *trails* the name it sits
        # inside, so allowing that would let a place-type question answer itself: the option
        # "편의점" would return the very 다모아편의점 being asked about.
        return 0.9 if name_key.startswith(query_key) and len(query_key) < len(name_key) else 0.0
    similarity = SequenceMatcher(None, query_key, name_key).ratio()
    if similarity < NAME_MATCH_FLOOR:
        return 0.0
    # Same guard the Kakao path needs: names of one kind share long generic affixes, so what
    # distinguishes them has to carry the match (오륜 vs 공릉, not 서울…초등학교).
    if distinguishing_similarity(query_key, name_key) < NAME_MATCH_FLOOR:
        return 0.0
    return similarity


class ContextMapProvider(MapProvider):
    """Serve the benchmark's contexts as the map provider, without any network call.

    One corpus, shared by every question, matching upstream's single `context_cache.db`. Every
    lookup the corpus can answer counts as a cache hit and none as an API call: the corpus *is* the
    cache. A lookup it cannot answer counts as a miss and is passed to `fallback` when one is
    configured — upstream falls back to the Google Maps API for exactly this — or fails with the
    same `ProviderError` the Kakao provider would raise when there is nothing to fall back to.
    """

    def __init__(
        self, contexts: Iterable[str] = (), *, fallback: MapProvider | None = None
    ) -> None:
        self._corpus = parse_contexts(contexts)
        self._fallback = fallback
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def api_call_count(self) -> int:
        return self._fallback.api_call_count if self._fallback is not None else 0

    @property
    def cache_hit_count(self) -> int:
        hits = self._fallback.cache_hit_count if self._fallback is not None else 0
        return self._cache_hits + hits

    @property
    def cache_miss_count(self) -> int:
        misses = self._fallback.cache_miss_count if self._fallback is not None else 0
        return self._cache_misses + misses

    @property
    def corpus(self) -> ContextDocument:
        return self._corpus

    def close(self) -> None:
        """Match the Kakao provider's lifecycle; the corpus itself holds no connection."""

        if self._fallback is not None and hasattr(self._fallback, "close"):
            self._fallback.close()

    def _ranked(self, query: str, candidates: list[Place], limit: int) -> list[Place]:
        scored = [
            (score, place)
            for place, score in ((place, _name_score(query, place.name)) for place in candidates)
            if score > 0
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [place for _, place in scored[: max(1, limit)]]

    def _dereference(self, value: str) -> Place | None:
        """A place the agent is already holding, addressed by id or by its coordinates.

        Both are references this provider handed out itself: an id it minted, or the latitude and
        longitude it printed on a place. An agent that reads them out of one tool result and passes
        them to the next is not naming a place, so sending them through the name search answers
        `PlaceNotFoundError` and a ReAct run spends its remaining steps re-searching a name that
        was never a name.
        """

        reference = value.strip()
        place = self._corpus._by_id.get(reference)
        if place is not None:
            return place
        coordinates = parse_coordinate_literal(reference)
        if coordinates is None:
            return None
        latitude, longitude = coordinates
        for candidate in self._corpus.all_places():
            if (
                haversine_meters(latitude, longitude, candidate.latitude, candidate.longitude)
                < 1.0
            ):
                return candidate
        # A point this context does not describe is still a point, and what is near it is a
        # question the context can answer.
        return Place(place_id=reference, name=reference, latitude=latitude, longitude=longitude)

    def _miss(self, call: Callable[[MapProvider], list[Place]]) -> list[Place]:
        """A lookup the corpus cannot answer, handed to the live provider when there is one.

        Upstream's operators do the same: `query_local_place` returns None on a miss and the
        Google Maps geocode call runs instead. Without a fallback the miss is the answer, and the
        caller turns it into the `PlaceNotFoundError` a missing POI deserves.
        """

        self._cache_misses += 1
        if self._fallback is None:
            return []
        return call(self._fallback)

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        referenced = self._dereference(query)
        if referenced is not None:
            self._cache_hits += 1
            return [referenced]
        found = self._ranked(query, self._corpus.all_places(), limit)
        if found:
            self._cache_hits += 1
            return found
        return self._miss(lambda provider: provider.search_place(query, limit=limit))

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        key = _name_key(address)
        by_address = [
            place
            for place in self._corpus.all_places()
            if place.address
            and (key in _name_key(place.address) or _name_key(place.address) in key)
        ]
        if by_address:
            self._cache_hits += 1
            return by_address[: max(1, limit)]
        return self.search_place(address, limit=limit)

    def reverse_geocode(
        self, latitude: float, longitude: float, *, limit: int = 5
    ) -> list[Place]:
        ordered = sorted(
            self._corpus.all_places(),
            key=lambda place: haversine_meters(
                latitude, longitude, place.latitude, place.longitude
            ),
        )
        if ordered:
            self._cache_hits += 1
            return ordered[: max(1, limit)]
        return self._miss(
            lambda provider: provider.reverse_geocode(latitude, longitude, limit=limit)
        )

    def place_details(self, place_id: str) -> Place:
        place = self._corpus._by_id.get(place_id)
        if place is not None:
            self._cache_hits += 1
            return place
        self._cache_misses += 1
        if self._fallback is None:
            raise PlaceNotFoundError(f"The corpus has no place with id {place_id!r}")
        return self._fallback.place_details(place_id)

    def _resolve_center(self, center: str | Place) -> Place:
        if isinstance(center, Place):
            return center
        referenced = self._dereference(center)
        if referenced is not None:
            return referenced
        matches = self._ranked(center, self._corpus.all_places(), 1)
        if matches:
            self._cache_hits += 1
            return matches[0]
        found = self._miss(lambda provider: provider.search_place(center, limit=1))
        if not found:
            raise PlaceNotFoundError(f"The corpus has no place named {center!r}")
        return found[0]

    def nearby_search(
        self,
        center: str | Place,
        *,
        query: str | None = None,
        category_code: str | None = None,
        radius_m: int = 2000,
        limit: int = 15,
    ) -> list[Place]:
        """Answer with the retrieval the corpus stored for this anchor.

        Upstream's `get_nearby_places` looks the reference place up in the `nearby_places` table
        and returns that block, re-ranked by distance — nothing else. A stored block *is* the
        provider's answer for its anchor, so the radius argument narrows it only when the block is
        itself radius-bounded (`in requested radius`); a k-nearest block carries no radius, and
        trimming it to whatever radius the agent guessed would report an absence the corpus never
        states. Kakao category codes do not filter it either: the block was retrieved by type
        already, and re-filtering by a code the context never carried can only drop evidence.

        Where upstream misses and calls the Google Maps API, an anchor with no stored block falls
        back to the places the corpus holds within the radius asked for — our direction and
        distance questions ship coordinates without a retrieval — and then to the live provider
        when one is configured.
        """

        anchor = self._resolve_center(center)
        stored: list[Place] = []
        for block in self._corpus.nearby:
            if _name_score(block.anchor, anchor.name) <= 0:
                continue
            for place in block.places:
                if block.radius_bounded and _distance(anchor, place) > radius_m:
                    continue
                stored.append(place)
        results = stored or [
            place
            for place in self._corpus.all_places()
            if place.place_id != anchor.place_id and _distance(anchor, place) <= radius_m
        ]
        if query and not stored:
            named = [place for place in results if _name_score(query, place.name) > 0]
            results = named or results
        results.sort(key=lambda place: _distance(anchor, place))
        if results:
            self._cache_hits += 1
            return results[: max(1, limit)]
        return self._miss(
            lambda provider: provider.nearby_search(
                anchor,
                query=query,
                category_code=category_code,
                radius_m=radius_m,
                limit=limit,
            )
        )

    def directions(
        self,
        origin: str | Place,
        destination: str | Place,
        *,
        mode: str = "driving",
        priority: str = "RECOMMEND",
        waypoints: list[str | Place] | None = None,
        include_steps: bool = False,
    ) -> Route:
        if mode not in ("driving", "car"):
            raise UnsupportedTravelModeError(f"Context routes are driving only, not {mode!r}")
        start = self._resolve_center(origin)
        end = self._resolve_center(destination)
        for route in self._corpus.routes:
            if _name_score(route.origin, start.name) > 0 and (
                _name_score(route.destination, end.name) > 0
            ):
                self._cache_hits += 1
                return route
        self._cache_misses += 1
        if self._fallback is None:
            raise RouteNotFoundError(
                f"The corpus has no route from {start.name!r} to {end.name!r}"
            )
        return self._fallback.directions(
            start,
            end,
            mode=mode,
            priority=priority,
            waypoints=waypoints,
            include_steps=include_steps,
        )
