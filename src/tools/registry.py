from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models import Place
from src.tools.map import MapProvider

KakaoCategoryCode = Literal[
    "MT1",
    "CS2",
    "PS3",
    "SC4",
    "AC5",
    "PK6",
    "OL7",
    "SW8",
    "BK9",
    "CT1",
    "AG2",
    "PO3",
    "AT4",
    "AD5",
    "FD6",
    "CE7",
    "HP8",
    "PM9",
]

KAKAO_CATEGORY_ALIASES = {
    "대형마트": "MT1",
    "편의점": "CS2",
    "어린이집": "PS3",
    "유치원": "PS3",
    "학교": "SC4",
    "학원": "AC5",
    "주차장": "PK6",
    "주유소": "OL7",
    "충전소": "OL7",
    "역": "SW8",
    "지하철역": "SW8",
    "은행": "BK9",
    "문화시설": "CT1",
    "부동산": "AG2",
    "공공기관": "PO3",
    "관광명소": "AT4",
    "숙박": "AD5",
    "음식점": "FD6",
    "카페": "CE7",
    "병원": "HP8",
    "약국": "PM9",
}


class PlaceSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = None
    center: str | Place | None = None
    category_code: KakaoCategoryCode | None = None
    radius_m: int = Field(default=2000, ge=1, le=20000)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    open_now: bool | None = None
    limit: int = Field(default=5, ge=1, le=45)

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=45)

    @model_validator(mode="after")
    def require_selector(self) -> PlaceSearchArgs:
        if self.center is None and not self.query:
            raise ValueError("place_search requires query when center is omitted")
        if self.center is not None and not self.query and not self.category_code:
            raise ValueError("spatial place_search requires query or category_code")
        if self.center is None:
            self.limit = min(self.limit, 15)
        return self


class PlaceDetailsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    place_id: str = Field(description="ID returned by another place tool in this run")


class GeocodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    address: str = Field(description="Korean road-name or land-lot address")
    limit: int = Field(default=5, ge=1, le=15)

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=15)


class ReverseGeocodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    limit: int = Field(default=5, ge=1, le=15)


class BatchPlaceDetailsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    place_ids: list[str] = Field(min_length=1, max_length=45)


class NearbyPlacesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    center: str | Place = Field(
        description="Center place name, place_id, or normalized Place from an earlier step"
    )
    query: str | None = Field(default=None, description="Place type/name keyword, e.g. 지하철역")
    category_code: KakaoCategoryCode | None = Field(
        default=None,
        description=(
            "Official Kakao category group code. Use query for types without a group code."
        ),
    )
    radius_m: int = Field(default=2000, ge=1, le=20000)
    limit: int = Field(default=45, ge=1, le=45)

    @field_validator("radius_m", mode="before")
    @classmethod
    def clamp_radius(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=20_000)

    @field_validator("category_code", mode="before")
    @classmethod
    def normalize_category_code(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=45)

    @model_validator(mode="after")
    def require_search_selector(self) -> NearbyPlacesArgs:
        if not self.query and not self.category_code:
            raise ValueError("nearby search requires query or category_code")
        if self.query and not self.category_code:
            self.category_code = KAKAO_CATEGORY_ALIASES.get("".join(self.query.split()))
        return self


class DirectionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str | Place = Field(
        description="Origin name, place_id, or normalized Place from an earlier step"
    )
    destination: str | Place = Field(
        description="Destination name, place_id, or normalized Place from an earlier step"
    )
    mode: str = Field(default="driving", description="MVP supports driving")
    priority: str = Field(default="RECOMMEND", description="RECOMMEND, TIME, or DISTANCE")
    waypoints: list[str | Place] = Field(default_factory=list, max_length=30)
    include_steps: bool = False


class CalculateFinishTimeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_time: str
    locations: list[str | Place] = Field(min_length=1, max_length=30)
    stay_durations_s: list[float] = Field(default_factory=list, max_length=30)
    timezone: str = "Asia/Seoul"
    mode: str = "driving"
    priority: str = "TIME"

    @model_validator(mode="after")
    def validate_stays(self) -> CalculateFinishTimeArgs:
        if self.stay_durations_s and len(self.stay_durations_s) != len(self.locations):
            raise ValueError("stay_durations_s must be empty or match locations")
        return self


class BatchGeocodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    place_names: list[str] = Field(min_length=1, max_length=30)
    anchor: str | Place | None = Field(
        default=None,
        description="Optional anchor used to disambiguate nearby places",
    )
    radius_m: int = Field(default=20_000, ge=1, le=20_000)
    limit: int = Field(default=1, ge=1, le=15)

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=15)

    @field_validator("radius_m", mode="before")
    @classmethod
    def clamp_radius(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=20_000)


class RecoverOptionPlacesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    options: list[str] = Field(min_length=1, max_length=15)
    candidates: list[Place] = Field(default_factory=list)
    anchor: Place
    radius_m: int = Field(default=20_000, ge=1, le=20_000)

    @field_validator("radius_m", mode="before")
    @classmethod
    def clamp_radius(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=20_000)


class RoutePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str | Place | None
    destination: str | Place | None
    label: str | None = None


class DistanceMatrixArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origins: list[str | Place | None] | None = Field(default=None, max_length=15)
    destinations: list[str | Place | None] | None = Field(default=None, max_length=15)
    pairs: list[RoutePair] | None = Field(default=None, max_length=30)
    mode: str = "driving"
    priority: str = "RECOMMEND"

    @model_validator(mode="after")
    def require_matrix_or_pairs(self) -> DistanceMatrixArgs:
        if self.pairs:
            return self
        if self.origins and self.destinations:
            return self
        raise ValueError("distance_matrix requires pairs or non-empty origins and destinations")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


@dataclass(frozen=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    output: Any | None
    status: str
    error: str | None = None

    def observation(self) -> dict[str, Any]:
        if self.status == "ok":
            return {"status": "ok", "result": self.output}
        return {"status": "error", "error": self.error}


class ToolRegistry:
    """One provider-neutral tool surface shared by ReAct and Spatial-Agent."""

    def __init__(self, provider: MapProvider) -> None:
        self.provider = provider
        self.calls: list[ToolExecution] = []
        self._tools = {
            tool.name: tool
            for tool in (
                ToolDefinition(
                    "place_search",
                    "Search by keyword, or around center with radius/type/rating/open filters.",
                    PlaceSearchArgs,
                    self._place_search,
                ),
                ToolDefinition(
                    "geocode",
                    "Convert a Korean address into normalized coordinates and address fields.",
                    GeocodeArgs,
                    lambda args: self.provider.geocode(args.address, limit=args.limit),
                ),
                ToolDefinition(
                    "reverse_geocode",
                    "Convert WGS84 coordinates to a normalized Korean address.",
                    ReverseGeocodeArgs,
                    lambda args: self.provider.reverse_geocode(
                        args.latitude, args.longitude, limit=args.limit
                    ),
                ),
                ToolDefinition(
                    "batch_geocode",
                    "Resolve a list of place names in one GeoFlow operator. Results preserve input "
                    "order and expose each best match as the place field.",
                    BatchGeocodeArgs,
                    self._batch_geocode,
                ),
                ToolDefinition(
                    "place_details",
                    "Read normalized cached place metadata by place_id.",
                    PlaceDetailsArgs,
                    lambda args: self.provider.place_details(args.place_id),
                ),
                ToolDefinition(
                    "batch_place_details",
                    "Read cached metadata for several places in input order.",
                    BatchPlaceDetailsArgs,
                    lambda args: [
                        self.provider.place_details(place_id) for place_id in args.place_ids
                    ],
                ),
                ToolDefinition(
                    "nearby_places",
                    "Find places around a center with Kakao Local, sorted by distance. "
                    "Supply query or an official category_code; query may be combined with "
                    "category_code as a filter.",
                    NearbyPlacesArgs,
                    lambda args: self.provider.nearby_search(
                        args.center,
                        query=args.query,
                        category_code=args.category_code,
                        radius_m=args.radius_m,
                        limit=args.limit,
                    ),
                ),
                ToolDefinition(
                    "recover_option_places",
                    "Resolve only options absent from ranked nearby results, then merge them.",
                    RecoverOptionPlacesArgs,
                    self._recover_option_places,
                ),
                ToolDefinition(
                    "directions",
                    "Get a route with optional verified waypoints and navigation steps.",
                    DirectionsArgs,
                    lambda args: self.provider.directions(
                        args.origin,
                        args.destination,
                        mode=args.mode,
                        priority=args.priority,
                        waypoints=args.waypoints,
                        include_steps=args.include_steps,
                    ),
                ),
                ToolDefinition(
                    "travel_time",
                    "Get normalized driving time and distance. Same evidence schema as directions.",
                    DirectionsArgs,
                    lambda args: self.provider.directions(
                        args.origin,
                        args.destination,
                        mode=args.mode,
                        priority=args.priority,
                        waypoints=args.waypoints,
                        include_steps=args.include_steps,
                    ),
                ),
                ToolDefinition(
                    "distance_matrix",
                    "Compute driving distance/duration for an origin-destination matrix or an "
                    "explicit list of ordered route pairs. Individual route failures are isolated.",
                    DistanceMatrixArgs,
                    self._distance_matrix,
                ),
                ToolDefinition(
                    "calculate_finish_time",
                    "Compute a multi-stop finish time from live/cached travel times and stays.",
                    CalculateFinishTimeArgs,
                    self._calculate_finish_time,
                ),
            )
        }

    @property
    def tool_call_count(self) -> int:
        return len(self.calls)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        tool = self._tools.get(name)
        if tool is None:
            execution = ToolExecution(name, arguments, None, "error", f"Unknown tool: {name}")
            self.calls.append(execution)
            return execution
        try:
            parsed = tool.args_model.model_validate(arguments)
            output = _jsonable(tool.handler(parsed))
            execution = ToolExecution(name, parsed.model_dump(), output, "ok")
        except Exception as exc:  # failures are observations, not secret-bearing tracebacks
            execution = ToolExecution(
                name,
                arguments,
                None,
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        self.calls.append(execution)
        return execution

    def _batch_geocode(self, args: BatchGeocodeArgs) -> list[dict[str, Any]]:
        anchor_value = args.anchor or (args.place_names[0] if len(args.place_names) > 1 else None)
        anchor_place: Place | None = anchor_value if isinstance(anchor_value, Place) else None
        anchor_matches: list[Place] = []
        if isinstance(anchor_value, str):
            anchor_matches = _search_place_candidates(self.provider, anchor_value, limit=15)
            anchor_place = _best_place_match(anchor_value, anchor_matches)
        results: list[dict[str, Any]] = []
        for place_name in args.place_names:
            try:
                if anchor_place is not None and _same_search_text(place_name, anchor_value):
                    matches = anchor_matches or [anchor_place]
                elif anchor_place is not None:
                    matches = self.provider.nearby_search(
                        anchor_place,
                        query=place_name,
                        radius_m=args.radius_m,
                        limit=15,
                    )
                    if not matches:
                        matches = _search_place_candidates(self.provider, place_name, limit=15)
                else:
                    matches = _search_place_candidates(self.provider, place_name, limit=15)
                best_match = _best_place_match(place_name, matches)
                ordered_matches = (
                    [best_match, *(match for match in matches if match != best_match)]
                    if best_match
                    else []
                )
                results.append(
                    {
                        "query": place_name,
                        "place": _jsonable(best_match) if best_match else None,
                        "candidates": _jsonable(ordered_matches[: args.limit]),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "query": place_name,
                        "place": None,
                        "candidates": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results

    def _place_search(self, args: PlaceSearchArgs) -> list[Place]:
        places = (
            self.provider.search_place(str(args.query), limit=args.limit)
            if args.center is None
            else self.provider.nearby_search(
                args.center,
                query=args.query,
                category_code=args.category_code,
                radius_m=args.radius_m,
                limit=args.limit,
            )
        )
        if args.min_rating is not None:
            places = [
                place
                for place in places
                if place.rating is not None and place.rating >= args.min_rating
            ]
        if args.open_now is not None:
            places = [place for place in places if place.is_open is args.open_now]
        return places

    def _calculate_finish_time(self, args: CalculateFinishTimeArgs) -> dict[str, Any]:
        zone = ZoneInfo(args.timezone)
        start = datetime.fromisoformat(args.start_time)
        start = start.replace(tzinfo=zone) if start.tzinfo is None else start.astimezone(zone)
        stays = args.stay_durations_s or [0.0] * len(args.locations)
        route_evidence: list[dict[str, Any]] = []
        travel_seconds = 0
        for origin, destination in zip(args.locations, args.locations[1:], strict=False):
            route = self.provider.directions(
                origin, destination, mode=args.mode, priority=args.priority
            )
            travel_seconds += route.duration_s
            route_evidence.append(route.model_dump(mode="json"))
        stay_seconds = sum(float(value) for value in stays)
        finish = start + timedelta(seconds=travel_seconds + stay_seconds)
        return {
            "start_time": start.isoformat(),
            "finish_time": finish.isoformat(),
            "travel_duration_s": travel_seconds,
            "stay_duration_s": stay_seconds,
            "routes": route_evidence,
            "timezone": args.timezone,
        }

    def _distance_matrix(self, args: DistanceMatrixArgs) -> dict[str, Any]:
        pairs = list(args.pairs or [])
        if not pairs:
            pairs = [
                RoutePair(origin=origin, destination=destination)
                for origin in args.origins or []
                for destination in args.destinations or []
            ]
        routes: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs):
            if pair.origin is None or pair.destination is None:
                routes.append(
                    {
                        "pair_index": index,
                        "label": pair.label,
                        "origin": _place_label(pair.origin),
                        "destination": _place_label(pair.destination),
                        "status": "error",
                        "error": "PlaceNotFoundError: unresolved route endpoint",
                    }
                )
                continue
            try:
                route = self.provider.directions(
                    pair.origin,
                    pair.destination,
                    mode=args.mode,
                    priority=args.priority,
                )
                routes.append(
                    {
                        "pair_index": index,
                        "label": pair.label,
                        **_jsonable(route),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                routes.append(
                    {
                        "pair_index": index,
                        "label": pair.label,
                        "origin": _place_label(pair.origin),
                        "destination": _place_label(pair.destination),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {"routes": routes, "route_count": len(routes)}

    def _recover_option_places(self, args: RecoverOptionPlacesArgs) -> list[Place]:
        places = list(args.candidates)
        seen_place_ids = {place.place_id for place in places}
        for option in args.options:
            if any(_place_represents_option(option, place) for place in places):
                continue
            matches = self.provider.nearby_search(
                args.anchor,
                query=option,
                radius_m=args.radius_m,
                limit=15,
            )
            if not matches:
                matches = _search_place_candidates(self.provider, option, limit=15)
            match = _best_place_match(option, matches)
            if (
                match
                and _place_represents_option(option, match)
                and match.place_id not in seen_place_ids
            ):
                places.append(match)
                seen_place_ids.add(match.place_id)
        return places


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _clamp_int(value: Any, *, minimum: int, maximum: int) -> int:
    number = int(value)
    return max(minimum, min(number, maximum))


def _place_label(value: str | Place | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.name


def _same_search_text(place_name: str, anchor: str | Place | None) -> bool:
    anchor_name = anchor.name if isinstance(anchor, Place) else anchor
    if not anchor_name:
        return False
    return "".join(place_name.split()).casefold() == "".join(anchor_name.split()).casefold()


def _best_place_match(query: str, matches: list[Place]) -> Place | None:
    if not matches:
        return None
    normalized_query = _search_key(query)

    def score(place: Place) -> tuple[int, int, int, int, float]:
        normalized_name = _search_key(place.name)
        exact = int(normalized_query == normalized_name)
        containment = int(
            normalized_query in normalized_name or normalized_name in normalized_query
        )
        similarity = SequenceMatcher(None, normalized_query, normalized_name).ratio()
        return (
            exact,
            _branch_compatibility(query, place.name),
            _category_compatibility(query, place),
            containment,
            similarity,
        )

    return max(matches, key=score)


def _search_key(value: str) -> str:
    normalized = value.casefold()
    for source in ("공립작은도서관", "작은도서관", "새마을문고", "도서관"):
        normalized = normalized.replace(source, "문고")
    return "".join(character for character in normalized if character.isalnum())


def _place_represents_option(option: str, place: Place) -> bool:
    option_key = _search_key(option)
    place_key = _search_key(place.name)
    if not option_key or not place_key:
        return False
    containment = option_key in place_key or place_key in option_key
    fuzzy_match = (
        option_key[0] == place_key[0]
        and SequenceMatcher(None, option_key, place_key).ratio() >= 0.68
    )
    return containment or fuzzy_match


def _branch_compatibility(query: str, place_name: str) -> int:
    query_branch = _branch_token(query)
    place_branch = _branch_token(place_name)
    if not query_branch:
        return 0
    if place_branch == query_branch:
        return 2
    if place_branch and place_branch != query_branch:
        return -2
    place_key = _search_key(place_name)
    return 1 if _search_key(query_branch) in place_key else 0


def _branch_token(value: str) -> str | None:
    words = value.split()
    if not words:
        return None
    match = re.search(r"([^\s()]{1,20}?)(?:본점|지점|점)$", words[-1])
    return match.group(1) if match else None


def _category_compatibility(query: str, place: Place) -> int:
    query_key = _search_key(query)
    category_key = _search_key(place.category or "")
    name_key = _search_key(place.name)
    if any(term in query_key for term in ("은행", "새마을금고", "농협")):
        if "atm" in category_key or "365" in name_key:
            return -3
        if "은행" in category_key or "금융서비스" in category_key:
            return 3
    if "경찰서" in query_key or "파출소" in query_key or "치안센터" in query_key:
        return 2 if any(term in category_key for term in ("경찰서", "파출소")) else 0
    if "역" in query_key:
        return 2 if "지하철역" in category_key else 0
    return 0


def _search_place_candidates(provider: MapProvider, query: str, *, limit: int) -> list[Place]:
    """Progressively retry common POI-name variants, preserving provider/cache semantics."""

    matches: list[Place] = []
    seen_place_ids: set[str] = set()
    for variant in _query_variants(query):
        for place in provider.search_place(variant, limit=limit):
            if place.place_id not in seen_place_ids:
                matches.append(place)
                seen_place_ids.add(place.place_id)
        if _best_place_match(query, matches) is not None and any(
            _search_key(query) in _search_key(place.name)
            or _search_key(place.name) in _search_key(query)
            for place in matches
        ):
            break
    return matches


def _query_variants(query: str) -> list[str]:
    normalized = " ".join(query.split()).strip()
    variants = [normalized]
    without_company = normalized.replace("(주)", "").strip()
    if without_company != normalized:
        variants.append(without_company)
    for suffix in ("식품관", "푸드코트", "문화센터"):
        if without_company.endswith(suffix):
            variants.append(without_company[: -len(suffix)].strip())
    if "(" in without_company and ")" in without_company:
        outer = re.sub(r"\([^)]*\)", "", without_company).strip()
        inner = without_company.split("(", 1)[1].split(")", 1)[0].strip()
        variants.extend([outer, inner])
    punctuation_free = re.sub(r"[.·,_/-]+", " ", without_company)
    variants.append(" ".join(punctuation_free.split()))
    # Historical benchmark names may outlive a branch rename. Broader brand forms are only
    # attempted after the exact name misses, so live exact matches keep priority.
    branchless = re.sub(r"\s+\S{1,20}(?:본점|지점|점)$", "", without_company).strip()
    if branchless and branchless != without_company:
        variants.append(branchless)
    words = without_company.split()
    if len(words) > 1:
        variants.append(" ".join(words[:-1]))
        branch_token = words[-1]
        if branch_token.endswith("점") and len(branch_token) > 1:
            variants.append(f"{' '.join(words[:-1])} {branch_token[:-1]}")
            variants.append(branch_token[:-1])
    facility_suffixes = {
        "문고": "도서관",
        "도서관": "문고",
        "파출소": "경찰서",
        "치안센터": "경찰서",
    }
    for source, replacement in facility_suffixes.items():
        if without_company.endswith(source):
            variants.append(without_company[: -len(source)] + replacement)
    return list(dict.fromkeys(variant for variant in variants if variant))
