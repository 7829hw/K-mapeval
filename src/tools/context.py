"""A map provider that answers from a benchmark item's own context instead of a live API.

Upstream Spatial-Agent evaluates on MapEval-Textual, where every question ships the retrieval
results it can be answered from and the agent never calls Google. Ported here, that context is not
handed to the agent — it is loaded *behind* the tool layer, so both architectures still have to
decide which tool to call and still read normalized `Place` / `Route` objects. What changes is only
where those objects come from: a per-question cache the dataset carries, instead of Kakao.

That keeps the independent variable intact (agent architecture) while removing the two things Kakao
adds to it — nationwide name ambiguity and a POI index that does not contain every OSM place the
questions were generated from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.models import Place, Route
from src.tools.map import (
    MapProvider,
    PlaceNotFoundError,
    RouteNotFoundError,
    UnsupportedTravelModeError,
)
from src.tools.spatial import distinguishing_similarity, haversine_meters, strip_location_qualifier

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


EMPTY_DOCUMENT = ContextDocument()


def _clean_address(value: str) -> str:
    address = value.strip().rstrip(".").strip()
    return "" if address in ("", MISSING_ADDRESS) else address


def _name_key(value: str) -> str:
    normalized = strip_location_qualifier(value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _place_key(latitude: float, longitude: float) -> str:
    return f"{latitude:.7f},{longitude:.7f}"


def parse_context(context: str) -> ContextDocument:
    """Parse a MapEval-style context block into places, nearby retrievals, and routes.

    The format is fixed by the dataset generator: `Information of` blocks carry a place, `Nearby …`
    blocks carry one retrieval for an anchor, and `Route from … to … by …` blocks carry one route.
    Anything else in the text is ignored rather than guessed at.
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
    ordered = tuple(places[place_id] for place_id in dict.fromkeys(places))
    return ContextDocument(
        places=ordered,
        nearby=tuple(blocks),
        routes=tuple(routes),
        _by_id={place.place_id: place for place in ordered},
    )


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
    """Serve one benchmark item's context as the map provider, without any network call.

    Every lookup the active context can answer counts as a cache hit and none as an API call: the
    context *is* the cache. A lookup it cannot answer counts as a miss and fails with the same
    `ProviderError` the Kakao provider would raise, so a place the dataset never recorded stays a
    place-not-found rather than a silently emptier answer.
    """

    def __init__(self) -> None:
        # Parsing is memoized by context text so a repeated question costs nothing to re-bind.
        self._documents: dict[str, ContextDocument] = {}
        self._active = EMPTY_DOCUMENT
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def api_call_count(self) -> int:
        return 0

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hits

    @property
    def cache_miss_count(self) -> int:
        return self._cache_misses

    @property
    def active_document(self) -> ContextDocument:
        return self._active

    def activate_context(self, context: str | None) -> None:
        """Point every subsequent lookup at this question's evidence.

        Called once per question, always — including with `None`, so one question's context can
        never answer the next one's lookups.
        """

        if not context:
            self._active = EMPTY_DOCUMENT
            return
        document = self._documents.get(context)
        if document is None:
            document = parse_context(context)
            self._documents[context] = document
        self._active = document

    def close(self) -> None:
        """Match the Kakao provider's lifecycle; a context provider holds no connection."""

    def _served(self, results: list[Place]) -> list[Place]:
        if results:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return results

    def _ranked(self, query: str, candidates: list[Place], limit: int) -> list[Place]:
        scored = [
            (score, place)
            for place, score in ((place, _name_score(query, place.name)) for place in candidates)
            if score > 0
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [place for _, place in scored[: max(1, limit)]]

    def search_place(self, query: str, *, limit: int = 5) -> list[Place]:
        return self._served(self._ranked(query, self._active.all_places(), limit))

    def geocode(self, address: str, *, limit: int = 5) -> list[Place]:
        key = _name_key(address)
        by_address = [
            place
            for place in self._active.all_places()
            if place.address
            and (key in _name_key(place.address) or _name_key(place.address) in key)
        ]
        if by_address:
            return self._served(by_address[: max(1, limit)])
        return self.search_place(address, limit=limit)

    def reverse_geocode(
        self, latitude: float, longitude: float, *, limit: int = 5
    ) -> list[Place]:
        ordered = sorted(
            self._active.all_places(),
            key=lambda place: haversine_meters(
                latitude, longitude, place.latitude, place.longitude
            ),
        )
        return self._served(ordered[: max(1, limit)])

    def place_details(self, place_id: str) -> Place:
        place = self._active._by_id.get(place_id)
        if place is None:
            self._cache_misses += 1
            raise PlaceNotFoundError(f"Context has no place with id {place_id!r}")
        self._cache_hits += 1
        return place

    def _resolve_center(self, center: str | Place) -> Place:
        if isinstance(center, Place):
            return center
        matches = self._ranked(center, self._active.all_places(), 1)
        if not matches:
            self._cache_misses += 1
            raise PlaceNotFoundError(f"Context has no place named {center!r}")
        return matches[0]

    def nearby_search(
        self,
        center: str | Place,
        *,
        query: str | None = None,
        category_code: str | None = None,
        radius_m: int = 2000,
        limit: int = 15,
    ) -> list[Place]:
        """Answer with what the context retrieved around this anchor, plus anything else it holds.

        A nearby block is the provider's stored result for its anchor, so it is returned as
        recorded. The radius argument only narrows it when the block is itself radius-bounded: a
        k-nearest block carries no radius, and truncating it by whatever radius the agent guessed
        would report an absence the context never states. Places the context mentions outside any
        block were not retrieved together, so those *are* filtered by the radius asked for.
        """

        anchor = self._resolve_center(center)
        results: list[Place] = []
        block_ids: set[str] = set()
        for block in self._active.nearby:
            if _name_score(block.anchor, anchor.name) <= 0:
                continue
            for place in block.places:
                distance = haversine_meters(
                    anchor.latitude, anchor.longitude, place.latitude, place.longitude
                )
                if block.radius_bounded and distance > radius_m:
                    continue
                results.append(place)
                block_ids.add(place.place_id)
        for place in self._active.all_places():
            if place.place_id in block_ids or place.place_id == anchor.place_id:
                continue
            distance = haversine_meters(
                anchor.latitude, anchor.longitude, place.latitude, place.longitude
            )
            if distance <= radius_m:
                results.append(place)
        if query:
            named = [place for place in results if _name_score(query, place.name) > 0]
            if named:
                results = named
        results.sort(
            key=lambda place: haversine_meters(
                anchor.latitude, anchor.longitude, place.latitude, place.longitude
            )
        )
        return self._served(results[: max(1, limit)])

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
        for route in self._active.routes:
            if _name_score(route.origin, start.name) > 0 and (
                _name_score(route.destination, end.name) > 0
            ):
                self._cache_hits += 1
                return route
        self._cache_misses += 1
        raise RouteNotFoundError(f"Context has no route from {start.name!r} to {end.name!r}")
