from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data._toolkit.map import MapProvider, PlaceNotFoundError
from data._toolkit.models import Place, Route
from data._toolkit.spatial import (
    SpatialOperatorRegistry,
    _cardinal_direction,
    _parse_datetime,
    build_duration_matrix,
    distinguishing_similarity,
    haversine_meters,
    strip_location_qualifier,
)

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


PLACE_FIELDS = frozenset(Place.model_fields)
# Wrappers a planner reaches a place through instead of referencing the place itself.
PLACE_WRAPPER_KEYS = ("place", "location", "nearest", "center", "anchor")


# RECOMMEND is Kakao's ordinary traffic-aware route, so every word that names *no* preference —
# "normal", "traffic", "the usual way" — means it. Those are not a fourth objective the provider
# lacks; they are a planner filling a required field for a question that never asked for a
# particular route, and refusing them failed all twenty-five legs of a matrix at once, which left
# `tsp_tw` nothing square to read and the generation stage guessing the answer.
PRIORITY_SYNONYMS = {
    "RECOMMEND": "RECOMMEND", "RECOMMENDED": "RECOMMEND", "DEFAULT": "RECOMMEND",
    "BALANCED": "RECOMMEND", "OPTIMAL": "RECOMMEND", "BEST": "RECOMMEND",
    "NORMAL": "RECOMMEND", "STANDARD": "RECOMMEND", "REGULAR": "RECOMMEND",
    "TRAFFIC": "RECOMMEND", "TRAFFIC_AWARE": "RECOMMEND", "REALTIME": "RECOMMEND",
    "REAL_TIME": "RECOMMEND", "LIVE": "RECOMMEND", "기본": "RECOMMEND", "실시간": "RECOMMEND",
    # Not a preference at all but an instruction to the router, which is the ordinary route.
    "RECALCULATE": "RECOMMEND", "RECOMPUTE": "RECOMMEND",
    "TIME": "TIME", "FASTEST": "TIME", "FAST": "TIME", "DURATION": "TIME", "QUICKEST": "TIME",
    "MINUTES": "TIME", "TRAVEL_TIME": "TIME",
    "DISTANCE": "DISTANCE", "SHORTEST": "DISTANCE", "SHORT": "DISTANCE",
}


def _as_priority(value: Any) -> Any:
    """Accept the words a planner reaches for, but only where the meaning is unambiguous.

    Kakao names its route priorities RECOMMEND/TIME/DISTANCE; an LLM writes "fastest" or
    "shortest" and the whole node failed on the spelling. A word that does not clearly mean one
    of the three is left alone so it still fails — this is leniency about wording, not meaning.
    A word that names no objective at all ("normal", "traffic") is not a fourth meaning; it is
    the ordinary route, which is what RECOMMEND is.
    """

    if isinstance(value, str):
        return PRIORITY_SYNONYMS.get(value.strip().upper(), value)
    return value


def _as_place_argument(value: Any) -> Any:
    """Accept the shapes a planner reaches a place through, not only a place.

    Execution is lenient about shape and strict only about evidence, and the local operators
    already normalize their inputs through `_as_place`. The tool arguments did not, so the same
    planner artifact that an operator shrugs off failed here as a pydantic ValidationError before
    any tool ran: a one-element list is the geocode result the planner forgot to index into, a
    wrapper carries the place under `place` or `location`, and a place an earlier operator enriched
    with `distance_m` or `candidate_index` is still that place — `Place` just forbids the extra
    keys. None of these is missing evidence, so none of them may end a run.
    """

    if isinstance(value, list) and len(value) == 1:
        return _as_place_argument(value[0])
    if isinstance(value, str):
        # A place an agent serialized back as JSON text. The ReAct baseline does this with the
        # place it just retrieved, and the string went to Kakao as a keyword query: twelve
        # HTTP 400s in one run, each one a retrieval the agent thought it had made. The
        # coordinates in it came from an earlier tool result, so reading them back adds nothing.
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            if isinstance(decoded, dict):
                return _as_place_argument(decoded)
        return value
    if not isinstance(value, dict):
        return value
    for key in PLACE_WRAPPER_KEYS:
        nested = value.get(key)
        if isinstance(nested, dict | list) and key not in PLACE_FIELDS:
            return _as_place_argument(nested)
    if "latitude" in value or "longitude" in value:
        extra = set(value) - PLACE_FIELDS
        if extra:
            return {key: item for key, item in value.items() if key in PLACE_FIELDS}
    return value


def _is_unresolved_place(value: Any) -> bool:
    """A geocode row that found nothing, or the marker of a step that failed.

    `batch_geocode` answers every name it was given, including the ones it could not resolve, and
    a planner passes the whole list on. One `{"query": …, "place": None}` among four good places
    failed the call with seven validation errors about the fields an unresolved row does not have,
    and the tool never ran. The row carries no place, so it is dropped — the places beside it are
    still evidence, and `recover_option_places` exists precisely to look up what is missing.
    """

    if not isinstance(value, dict):
        return False
    if "latitude" in value and "longitude" in value:
        return False
    return "error" in value or "query" in value


def _as_place_list_argument(value: Any) -> Any:
    if isinstance(value, list):
        # A list inside the list is the whole geocode node written where its places belong:
        # `locations: ["$places"]` resolves to `[[rec, rec, rec, rec]]`, and an itinerary of one
        # four-place list is not a shape any operator can read. This is the mirror of the
        # one-element unwrap below — the planner indexed one level too few instead of too many —
        # and it failed the clock before a single leg was routed.
        flattened: list[Any] = []
        for item in value:
            flattened.extend(item) if isinstance(item, list) else flattened.append(item)
        normalized = [_as_place_argument(item) for item in flattened]
        return [item for item in normalized if not _is_unresolved_place(item)]
    if isinstance(value, dict):
        # The mirror of `_as_place_argument`'s one-element list: a lone place-shaped value where
        # a list is expected is a list of one. A planner that referenced `$geocode.0` instead of
        # `$geocode` failed `locations` as a `list_type` validation error before the clock ran.
        return [_as_place_argument(value)]
    return value


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

    @field_validator("center", mode="before")
    @classmethod
    def normalize_center(cls, value: Any) -> Any:
        return _as_place_argument(value)


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

    @field_validator("center", mode="before")
    @classmethod
    def normalize_center(cls, value: Any) -> Any:
        return _as_place_argument(value)


class DirectionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str | Place = Field(
        description="Origin name, place_id, or normalized Place from an earlier step"
    )
    destination: str | Place = Field(
        description="Destination name, place_id, or normalized Place from an earlier step"
    )
    mode: str = Field(default="driving", description="MVP supports driving")
    # Kakao's own value list, the way upstream's TravelTime documents its travelMode: the accepted
    # values and nothing else. Glossing what each one optimizes was written after watching ReAct
    # read a question about the shortest route as RECOMMEND, which makes the baseline's vocabulary
    # a function of the test set. Which priority a question asks for is grounding, and grounding is
    # a Spatial-Agent stage under measurement (`_extract_route_priority`) — not a gap to close in
    # prose here.
    priority: str = Field(
        default="RECOMMEND", description="RECOMMEND, TIME, or DISTANCE"
    )

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Any) -> Any:
        return _as_priority(value)
    waypoints: list[str | Place] = Field(default_factory=list, max_length=30)
    # Upstream's `DirectionsTool` prints every step's instructions on every call, so a route and
    # its guidance are one observation there. Defaulting this to False made the guidance something
    # an agent had to know to ask for, which is a capability our port withheld from the baseline
    # rather than one MapEval's design withholds. TravelTime is the tool that answers without
    # steps -- see `TravelTimeArgs`.
    include_steps: bool = True

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def normalize_endpoint(cls, value: Any) -> Any:
        return _as_place_argument(value)

    @field_validator("waypoints", mode="before")
    @classmethod
    def normalize_waypoints(cls, value: Any) -> Any:
        return _as_place_list_argument(value)


class TravelTimeArgs(DirectionsArgs):
    """The same route request, reported as duration and distance.

    Upstream keeps TravelTime and Directions apart by what they return: `TravelTimeTool` reports
    one duration and one distance, `DirectionsTool` reports the routes with their steps. Same
    evidence, same provider call, different report -- so the only thing that differs here is the
    default.
    """

    include_steps: bool = False


class CalculateFinishTimeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Exactly one of these. An itinerary is asked about in both directions — when do I get back,
    # and when must I leave — and only running the clock forward left the reverse question to be
    # assembled by hand from a scalar the planner had to sum itself, which it under-counted.
    start_time: str | None = None
    arrival_time: str | None = None
    locations: list[str | Place] = Field(min_length=1, max_length=30)
    stay_durations_s: list[float] = Field(default_factory=list, max_length=30)
    timezone: str = "Asia/Seoul"
    mode: str = "driving"
    priority: str = "TIME"

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Any) -> Any:
        return _as_priority(value)

    @field_validator("locations", mode="before")
    @classmethod
    def normalize_locations(cls, value: Any) -> Any:
        return _as_place_list_argument(value)

    @model_validator(mode="after")
    def validate_stays(self) -> CalculateFinishTimeArgs:
        if self.stay_durations_s and len(self.stay_durations_s) != len(self.locations):
            raise ValueError("stay_durations_s must be empty or match locations")
        if bool(self.start_time) == bool(self.arrival_time):
            raise ValueError("give exactly one of start_time or arrival_time")
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
    strict_names: bool = Field(
        default=False,
        description=(
            "Require every name to match by name in the same script it was written in. Set "
            "when the names are POIs the question states precisely, rather than option shorthand "
            "that may be a transliteration of the Kakao entry."
        ),
    )

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
    # A planner routinely writes the anchor's name here rather than referencing the step that
    # resolved it. That is a shape artifact, not missing evidence, so the tool resolves the name.
    anchor: str | Place
    category_code: KakaoCategoryCode | None = Field(
        default=None,
        description=(
            "Kakao category the question asks about. Set it when the retrieval was categorised, "
            "so an option is only satisfied by a place of the kind being asked for."
        ),
    )
    radius_m: int = Field(default=20_000, ge=1, le=20_000)
    direction: str | None = Field(
        default=None,
        description=(
            "Cardinal direction the question asks about. Set it when the candidates were "
            "direction-filtered, so a recovered option is only accepted in the same sector."
        ),
    )

    @field_validator("radius_m", mode="before")
    @classmethod
    def clamp_radius(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=20_000)

    @field_validator("anchor", mode="before")
    @classmethod
    def normalize_anchor(cls, value: Any) -> Any:
        return _as_place_argument(value)

    @field_validator("candidates", mode="before")
    @classmethod
    def normalize_candidates(cls, value: Any) -> Any:
        return _as_place_list_argument(value)


class RoutePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str | Place | None
    destination: str | Place | None
    label: str | None = None

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def normalize_endpoint(cls, value: Any) -> Any:
        return _as_place_argument(value)


class DistanceMatrixArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origins: list[str | Place | None] | None = Field(default=None, max_length=15)
    destinations: list[str | Place | None] | None = Field(default=None, max_length=15)
    pairs: list[RoutePair] | None = Field(default=None, max_length=30)

    @field_validator("origins", "destinations", mode="before")
    @classmethod
    def normalize_endpoints(cls, value: Any) -> Any:
        # A planner naturally writes origins: "$places" and gets back batch_geocode's
        # {query, place, candidates} records. Rejecting that shape here is what stopped every
        # matrix — and with it every tsp_tw — on the trip questions.
        return _as_place_list_argument(value)
    mode: str = "driving"
    priority: str = "RECOMMEND"

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Any) -> Any:
        return _as_priority(value)

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

    #: The tool surface MapEval's own ReAct baseline is given
    #: (`mapeval-api/Tools.py`: PlaceSearch, PlaceId, PlaceDetails, NearbyPlaces, TravelTime,
    #: Directions). Everything this registry adds beyond it — `batch_geocode`,
    #: `batch_place_details`, `distance_matrix`, `calculate_finish_time`,
    #: `recover_option_places` — is an *aggregation* over those primitives, and aggregation is
    #: what GeoFlow's operator graph exists to express. Sharing them with the ReAct agent keeps
    #: our two agents comparable to each other, and makes neither comparable to the paper's
    #: baseline: a trip question the paper's ReAct must orchestrate over a dozen turns is two
    #: calls here. Restrict a ReAct registry to this set to reproduce the paper's comparison.
    # The five `mapeval-api/Evaluator2.py` (35d481a, line 33) actually instantiates:
    # PlaceSearchTool, PlaceDetailsTool, NearbyPlacesTool, TravelTimeTool, DirectionsTool.
    # `Tools.py` also defines a PlaceIdTool, but the evaluator never constructs it, and
    # `FormattedTools.PlaceSearchTool` is documented as "Get place ID for a given location" —
    # they are the same primitive under two names in two files, not a sixth tool.
    # `geocode` / `reverse_geocode` are excluded deliberately: upstream reaches every place
    # through a place id and converts between an address and coordinates nowhere, so both are
    # capabilities the paper's baseline was measured without. Name resolution stays reachable
    # through `place_search`, which is what PlaceSearch is.
    MAPEVAL_BASELINE_TOOLS = frozenset(
        {
            "place_search",
            "place_details",
            "nearby_places",
            "travel_time",
            "directions",
        }
    )

    def __init__(self, provider: MapProvider, *, allowed: Iterable[str] | None = None) -> None:
        self.provider = provider
        self.calls: list[ToolExecution] = []
        self._tools = {
            tool.name: tool
            for tool in (
                ToolDefinition(
                    "place_search",
                    "Find the place_id for a name. Returns place_id and name only; call "
                    "place_details with the id for coordinates, address and category.",
                    PlaceSearchArgs,
                    self._place_search,
                ),
                ToolDefinition(
                    "geocode",
                    "Convert a Korean address into normalized coordinates and address fields; "
                    "falls back to the place-name index when the address has no entry.",
                    GeocodeArgs,
                    self._geocode,
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
                        self._reference(args.center),
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
                    self._route,
                ),
                ToolDefinition(
                    "travel_time",
                    "Get normalized driving time and distance. Same evidence schema as directions.",
                    TravelTimeArgs,
                    self._route,
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

        if allowed is not None:
            permitted = set(allowed)
            unknown = permitted - set(self._tools)
            if unknown:
                raise ValueError(f"Unknown tools restricted on ToolRegistry: {sorted(unknown)}")
            self._tools = {
                name: tool for name, tool in self._tools.items() if name in permitted
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
            _reject_unresolved_places(name, arguments)
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
        results = self._resolve_batch(args, anchor_value, anchor_place)
        return self._reconcile_batch(args, results)

    def _reconcile_batch(
        self,
        args: BatchGeocodeArgs,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-resolve a batch that landed in two places at once.

        A question naming several POIs is asserting they sit in one neighbourhood, so a batch whose
        resolved places span more than `radius_m` has at least one name resolved wrong — and which
        one is not knowable from the first pass. Re-resolve the batch around each resolved place in
        turn and keep the reading that holds together. Re-ranking cannot substitute for this: a
        keyword search for an ambiguous short name fills every slot with one far-away city, so the
        intended place was never among the candidates to re-rank.
        """

        resolved = _resolved_places(results)
        # Only a pair. With three or more names the batch is an anchor plus option texts, the
        # anchored search has already put each option in the right neighbourhood, and "tightest
        # span" becomes the wrong objective: scattered option brands could out-vote a correct
        # anchor and drag the whole batch to another district. A wrongly distant option is
        # harmless — `nearest` simply never picks it — but a moved anchor invalidates everything.
        if len(resolved) != 2:
            return results
        best, best_span = results, _batch_span(resolved)
        if best_span <= min(args.radius_m, BATCH_LOCALITY_SPAN_M):
            return results
        for keeper_query, keeper_place in list(resolved.items())[:MAX_RECONCILE_KEEPERS]:
            trial = self._resolve_batch(args, keeper_query, keeper_place)
            trial_places = _resolved_places(trial)
            if len(trial_places) < len(resolved):
                continue  # a tighter batch is not worth losing a place over
            span = _batch_span(trial_places)
            if span < best_span:
                best, best_span = trial, span
        return best

    def _resolve_batch(
        self,
        args: BatchGeocodeArgs,
        anchor_value: str | Place | None,
        anchor_place: Place | None,
    ) -> list[dict[str, Any]]:
        anchor_matches = [anchor_place] if anchor_place is not None else []
        results: list[dict[str, Any]] = []
        for place_name in args.place_names:
            try:
                # Being near the anchor is not evidence of being the named place. Kakao's keyword
                # search is tolerant, so asking a neighbourhood for a name it does not contain
                # returns places of the same *kind* instead: 신사정육점 came back as 한아름축산,
                # 쌍문1치안센터 as 수유6치안센터. Both resolved, both were a different POI, and
                # every operator downstream then computed correctly over the wrong place. The
                # neighbourhood buys one thing only — the right to accept a name written in
                # another script ("A TWOSOME PLACE" for 투썸플레이스), where characters cannot
                # testify either way.
                anchored = False
                if anchor_place is not None and _same_search_text(place_name, anchor_value):
                    matches = anchor_matches or [anchor_place]
                elif anchor_place is not None:
                    matches = self.provider.nearby_search(
                        anchor_place,
                        query=place_name,
                        radius_m=args.radius_m,
                        limit=15,
                    )
                    anchored = bool(matches)
                    if not matches:
                        # Nothing within the anchor's radius: widen to a name search, but keep
                        # ranking against the anchor so a same-named place in another city cannot
                        # win on text alone.
                        matches = _search_place_candidates(self.provider, place_name, limit=15)
                else:
                    matches = _search_place_candidates(self.provider, place_name, limit=15)
                best_match = _best_place_match(
                    place_name,
                    matches,
                    anchor=anchor_place,
                    allow_cross_script=anchored and not args.strict_names,
                )
                if best_match is None and anchored:
                    # The neighbourhood has no place by this name. Widen before giving up: the POI
                    # may sit just outside the radius the question implied.
                    matches = _search_place_candidates(self.provider, place_name, limit=15)
                    best_match = _best_place_match(place_name, matches, anchor=anchor_place)
                if best_match is None:
                    raise PlaceNotFoundError(f"No place matched {place_name!r}")
                ordered_matches = [
                    best_match,
                    *(match for match in matches if match != best_match),
                ]
                results.append(
                    {
                        "query": place_name,
                        "place": _jsonable(best_match),
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

    @staticmethod
    def _search_view(places: list[Place]) -> list[dict[str, str]]:
        """What `mapeval-api`'s PlaceSearchTool returns: a reference, not a place.

        Upstream is `return data['results'][0]['place_id']` — an id and nothing else, so a baseline
        that wants coordinates calls `PlaceDetails` for them and pays a second round trip per
        place. Upstream *Spatial-Agent* is the opposite: `google_maps.nearby_search` and
        `get_place_details` both carry `lat`/`lng`, which is the tool-surface asymmetry the paper
        is built on. Ours handed the full normalized `Place` to whoever asked, and since
        Spatial-Agent never calls `place_search` at all (0 of 100 plans in the v4 run) every bit of
        that generosity went to the baseline: one call answered "how far apart are A and B".
        The name rides along because a nationwide keyword search needs to be checkable; the
        geometry does not.
        """

        return [{"place_id": place.place_id, "name": place.name} for place in places]

    def _place_search(self, args: PlaceSearchArgs) -> list[dict[str, str]]:
        places = (
            self.provider.search_place(str(args.query), limit=args.limit)
            if args.center is None
            else self.provider.nearby_search(
                self._reference(args.center),
                query=args.query,
                category_code=args.category_code,
                radius_m=args.radius_m,
                limit=args.limit,
            )
        )
        # An attribute filter over a field this provider never populates returns nothing rather
        # than filtering — see `evidence_carries`. Kakao Local publishes neither ratings nor
        # opening hours, so both of these are dropped there and both apply against a context
        # corpus that carries them.
        if args.min_rating is not None and any(place.rating is not None for place in places):
            places = [
                place
                for place in places
                if place.rating is not None and place.rating >= args.min_rating
            ]
        if args.open_now is not None and any(place.is_open is not None for place in places):
            places = [place for place in places if place.is_open is args.open_now]
        return self._search_view(places)

    def _reference(self, value: Any) -> Any:
        """A place argument on a baseline tool must be something the provider handed out.

        The rule is upstream's tool surface, so it belongs on the tool rather than inside one
        provider: `mapeval-api/FormattedTools.py` gives its baseline `PlaceSearchTool` to turn a
        name into a `place_id` and every other tool consumes that id. Reading a name here would
        run the search behind the call and delete the id-threading the paper's baseline is
        measured doing.
        """

        if not isinstance(value, str):
            return value
        found = self.provider.dereference(value)
        if found is not None:
            return found
        raise PlaceNotFoundError(
            f"{value!r} is a place name, not a place reference. Call place_search with it first "
            "and pass the place_id it returns."
        )

    def _resolved(self, value: Any) -> Any:
        """Turn a name into a place for the aggregation tools, which are name resolvers.

        The five baseline tools no longer do this: a place argument there must be a reference the
        provider issued, exactly as `mapeval-api/FormattedTools.py` requires. That rule is about
        the *baseline's* surface. `batch_geocode`, `distance_matrix` and `calculate_finish_time`
        exist to take the names a plan is holding and resolve them in one step — that is what makes
        them aggregations over PlaceSearch rather than new evidence — so they resolve here, once,
        through the same matcher `batch_geocode` uses.
        """

        if not isinstance(value, str):
            return value
        found = self.provider.dereference(value)
        if found is not None:
            return found
        return _best_place_match(value, _search_place_candidates(self.provider, value, limit=15))

    def _calculate_finish_time(self, args: CalculateFinishTimeArgs) -> dict[str, Any]:
        stays = args.stay_durations_s or [0.0] * len(args.locations)
        route_evidence: list[dict[str, Any]] = []
        travel_seconds = 0
        locations = [self._resolved(value) for value in args.locations]
        for origin, destination in zip(locations, locations[1:], strict=False):
            route = self.provider.directions(
                origin, destination, mode=args.mode, priority=args.priority
            )
            travel_seconds += route.duration_s
            route_evidence.append(route.model_dump(mode="json"))
        stay_seconds = sum(float(value) for value in stays)
        total = timedelta(seconds=travel_seconds + stay_seconds)
        if args.start_time:
            start = _parse_datetime(args.start_time, args.timezone)
            finish = start + total
        else:
            finish = _parse_datetime(str(args.arrival_time), args.timezone)
            start = finish - total
        return {
            "start_time": start.isoformat(),
            "finish_time": finish.isoformat(),
            # Both clocks are reported and only one of them was computed: run forwards and the
            # start is the question's, run backwards and the finish is. Nothing in the field
            # names says which, so say it here rather than let a reader guess — guessing by name
            # answered a "when must I leave" question with the deadline it was handed.
            "derived_clock": "finish_time" if args.start_time else "start_time",
            "travel_duration_s": travel_seconds,
            "stay_duration_s": stay_seconds,
            "routes": route_evidence,
            "timezone": args.timezone,
        }

    def _geocode(self, args: GeocodeArgs) -> list[Place]:
        found = self.provider.geocode(args.address, limit=args.limit)
        if not found:
            # Kakao keeps addresses and place names in two indexes, and a planner writes the
            # question's place name here as readily as an address. `대림동 우리 골목형상점가`
            # has no address entry and one exact place entry, and failing here cost the whole
            # question: the anchor never resolved, so no step after it had anything to work on.
            # This is the lookup `place_search` already performs, so it reaches no evidence the
            # tool surface does not carry.
            match = _best_place_match(
                args.address, _search_place_candidates(self.provider, args.address, limit=15)
            )
            found = [match] if match else []
        if not found:
            # Still nothing, and saying so here is the difference between one clear failure and a
            # cascade: returned as `[]` it became a `center: []` that failed the retrieval as a
            # pydantic type error, and the error dict that left behind then failed option
            # recovery with seven more.
            raise PlaceNotFoundError(f"Kakao has no coordinates for {args.address!r}")
        return found

    def _route(self, args: DirectionsArgs) -> Route:
        """One route, with the leg from a place to itself answered rather than asked for."""

        origin = self._reference(args.origin)
        destination = self._reference(args.destination)
        waypoints = [self._reference(value) for value in args.waypoints]
        if not waypoints and _same_endpoint(origin, destination):
            return _self_route(origin, destination)
        return self.provider.directions(
            origin,
            destination,
            mode=args.mode,
            priority=args.priority,
            waypoints=waypoints,
            include_steps=args.include_steps,
        )

    def _distance_matrix(self, args: DistanceMatrixArgs) -> dict[str, Any]:
        pairs = [
            RoutePair(
                origin=self._resolved(pair.origin),
                destination=self._resolved(pair.destination),
                label=pair.label,
            )
            for pair in (args.pairs or [])
        ]
        if not pairs:
            origins = [self._resolved(value) for value in args.origins or []]
            destinations = [self._resolved(value) for value in args.destinations or []]
            pairs = [
                RoutePair(origin=origin, destination=destination)
                for origin in origins
                for destination in destinations
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
            if _same_endpoint(pair.origin, pair.destination):
                # A place is no distance from itself, and Kakao refuses to route it: the
                # diagonal of an origins x destinations matrix came back as 750 route errors in
                # one run, each one an API call, and the generation stage read them as a matrix
                # that had failed. This is the one leg that may be filled — an *absent*
                # off-diagonal leg is still missing evidence, never a zero-cost hop.
                routes.append(
                    {
                        "pair_index": index,
                        "label": pair.label,
                        **_jsonable(_self_route(pair.origin, pair.destination)),
                        "status": "ok",
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
        # `tsp_tw` reads a square matrix, so emit one here as well: without it the trip path is
        # only reachable by a planner inventing the numbers it is supposed to look up.
        built = build_duration_matrix(routes)
        return {
            "routes": routes,
            "route_count": len(routes),
            "nodes": built["nodes"],
            "matrix": built["matrix"] if built["complete"] else None,
            "matrix_complete": built["complete"],
            "missing_legs": built["missing_legs"],
        }

    def _require_place(self, value: str | Place) -> Place:
        """Resolve a place reference a planner wrote as a name."""

        if isinstance(value, Place):
            return value
        match = _best_place_match(value, _search_place_candidates(self.provider, value, limit=15))
        if match is None:
            raise PlaceNotFoundError(f"No place matched {value!r}")
        return match

    def _recover_option_places(self, args: RecoverOptionPlacesArgs) -> list[Place]:
        anchor = self._require_place(args.anchor)
        places = list(args.candidates)
        seen_place_ids = {place.place_id for place in places}
        for option in args.options:
            if any(_place_represents_option(option, place) for place in places):
                continue
            matches = self.provider.nearby_search(
                anchor,
                query=option,
                category_code=args.category_code,
                radius_m=args.radius_m,
                limit=15,
            )
            if not matches and args.category_code is None:
                # A categorised question wants a place of that kind near the anchor. Widening to a
                # nationwide name search would answer "is there a 목동 anywhere", not "is there a
                # 목동 station here", so only an uncategorised recovery falls back.
                matches = _search_place_candidates(self.provider, option, limit=15)
            # The anchor is the question's reference point, never one of its answers. Recovering
            # "목동" around 교보문고 목동점 otherwise returns the bookstore itself, and a radius-set
            # question then reports the station 목동 as present when only 오목교 is.
            matches = [match for match in matches if match.place_id != anchor.place_id]
            match = _best_place_match(option, matches, anchor=anchor)
            if (
                match
                and _place_represents_option(option, match)
                and match.place_id not in seen_place_ids
                and _within_anchor_radius(anchor, match, args.radius_m)
                and _within_anchor_sector(anchor, match, args.direction)
            ):
                places.append(match)
                seen_place_ids.add(match.place_id)
        return places



def _within_anchor_sector(anchor: Place, place: Place, direction: str | None) -> bool:
    """Whether a recovered place lies in the sector the question asks about.

    Recovery runs after the candidates have been filtered, and it adds the option texts the
    retrieval did not surface. Without the constraint it adds them regardless of where they are:
    a "which mart north of here" question ranked a recovered mart 271 m *south* of the anchor
    above the northern one the filter had correctly found at 961 m. Its own constraint is what
    the recovered option has to satisfy, exactly as the radius already is.
    """

    if not direction:
        return True
    try:
        expected = _cardinal_direction(direction)
    except ValueError:
        return True
    bearing = SpatialOperatorRegistry.bearing_to_direction(
        {"latitude": anchor.latitude, "longitude": anchor.longitude},
        {"latitude": place.latitude, "longitude": place.longitude},
    )
    return bool(bearing["cardinal_direction"] == expected)


def _within_anchor_radius(anchor: Place, place: Place, radius_m: int) -> bool:
    """Whether a recovered place is close enough to the anchor to be one of the answers.

    The anchored search is already bounded, but the nationwide name fallback is not: it answered
    "is there a 꽃담공방 anywhere" with one 129 km away, and the option then entered the candidate
    set as if it were near the anchor. A proximity question cannot be answered by a place outside
    the radius it was asked about.
    """

    return (
        haversine_meters(anchor.latitude, anchor.longitude, place.latitude, place.longitude)
        <= radius_m
    )


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


# Only arguments whose models require a place. `anchor` is deliberately absent: it is optional
# everywhere it appears, so an explicit null there means "no anchor", not a failed lookup.
PLACE_ARGUMENT_NAMES = frozenset({"center", "origin", "destination", "place"})

# Each reconciliation attempt re-resolves the whole batch, so cap how many readings we try.
MAX_RECONCILE_KEEPERS = 3
# How far apart a batch's places may sit before we stop believing they are one neighbourhood. This
# is not the search radius: `radius_m` is how wide we are willing to look for a name, while this is
# how wide a single locality can plausibly be. Two POIs a question compares are a district apart at
# most, so a wider span means one of the names matched its twin in another part of the country.
BATCH_LOCALITY_SPAN_M = 5_000


def _resolved_places(results: list[dict[str, Any]]) -> dict[str, Place]:
    """Successfully geocoded rows, keyed by the query text that produced them."""

    return {
        str(row["query"]): Place.model_validate(row["place"])
        for row in results
        if row.get("place")
    }


def _batch_span(places: dict[str, Place]) -> float:
    """Widest distance between any two places in the batch."""

    values = list(places.values())
    return max(
        (
            -_proximity_score(first, second)
            for index, first in enumerate(values)
            for second in values[index + 1 :]
        ),
        default=0.0,
    )


def _reject_unresolved_places(name: str, arguments: dict[str, Any]) -> None:
    """Name an unresolved place for what it is, before pydantic calls it a type error.

    An upstream geocode that found nothing leaves `None` in a downstream argument. Letting that
    reach the args model turns a provider failure into a `ValidationError`, which the evaluator
    then files as agent reasoning.
    """

    for argument in PLACE_ARGUMENT_NAMES:
        if argument in arguments and arguments[argument] is None:
            raise PlaceNotFoundError(
                f"{name} received an unresolved place for {argument!r}; "
                "the upstream geocode found no match"
            )


def _self_route(origin: str | Place | None, destination: str | Place | None) -> Route:
    """A place is no distance from itself.

    Kakao refuses to route it — "출발지와 도착지가 5 m 이내로 설정된 경우 경로를 탐색할 수 없음" —
    and one run spent 750 matrix calls and 64 baseline calls collecting that refusal, which the
    generation stage then read as evidence that the legs had failed. This is the only leg either
    architecture may have filled in, and both get it identically: the evidence below the tools is
    the same for both, whatever tools they reach it through.
    """

    return Route(
        origin=_place_label(origin),
        destination=_place_label(destination),
        distance_m=0,
        duration_s=0,
    )


def _same_endpoint(origin: str | Place | None, destination: str | Place | None) -> bool:
    """Whether both ends of a leg are the same place — by id, by name, or by standing on it."""

    if origin is None or destination is None:
        return False
    if isinstance(origin, Place) and isinstance(destination, Place):
        if origin.place_id == destination.place_id:
            return True
        return haversine_meters(
            origin.latitude, origin.longitude, destination.latitude, destination.longitude
        ) < 5.0
    return _place_label(origin).strip().casefold() == _place_label(destination).strip().casefold()


def _place_label(value: str | Place | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.name


def _same_search_text(place_name: str, anchor: str | Place | None) -> bool:
    anchor_name = anchor.name if isinstance(anchor, Place) else anchor
    if not anchor_name:
        return False
    return "".join(place_name.split()).casefold() == "".join(anchor_name.split()).casefold()


def _best_place_match(
    query: str,
    matches: list[Place],
    *,
    anchor: Place | None = None,
    require_name_evidence: bool = True,
    allow_cross_script: bool = False,
) -> Place | None:
    """Pick the place a question means, preferring the one nearest a known anchor.

    A bare brand name ("맘스터치") matches every branch equally well on text, so without the anchor
    term the tiebreak collapses to name similarity — which favours the *shortest* branch name, not
    the branch the question is about. Proximity sits below the name-evidence terms, so an exact or
    branch-specific match still wins over a merely closer one.
    """

    if not matches:
        return None
    normalized_query = _search_key(query)

    def score(place: Place) -> tuple[int, int, int, int, float, float]:
        normalized_name = _search_key(place.name)
        exact = int(normalized_query == normalized_name)
        containment = int(
            normalized_query in normalized_name or normalized_name in normalized_query
        )
        similarity = SequenceMatcher(None, normalized_query, normalized_name).ratio()
        return (
            exact,
            # What kind of business this is outranks which outlet of it. "CU 가락센트럴점" has no
            # exact match, so every real CU branch scores -2 on the branch suffix while an unrelated
            # "센트럴타워" scores 0 for having no suffix at all — with branch ranked first, the
            # unrelated place won and then failed the evidence floor, resolving nothing.
            _category_compatibility(query, place),
            _branch_compatibility(query, place.name),
            containment,
            _proximity_score(anchor, place),
            similarity,
        )

    best = max(matches, key=score)
    if require_name_evidence and not _names_the_same_place(
        normalized_query, best, allow_cross_script=allow_cross_script
    ):
        return None
    return best


# Below this, two names share a few characters by coincidence rather than describing one place.
# Every genuine resolution observed in the benchmark clears it by containment alone.
NAME_EVIDENCE_FLOOR = 0.55


def _containment_is_evidence(left: str, right: str) -> bool:
    """Is one name inside the other because they name one place, or by coincidence?

    A short query is a substring of a great many long names. 압구정 sits inside
    해피냠냠라면가게한강버스압구정선착장점 while naming a different place entirely, and treating
    that as a match resolved a distance question to a POI 12 km from the one asked about. A brand
    followed by its branch (올리브영 / 올리브영 거여역점) leads the name it extends; anything else
    has to make up a real share of it.
    """

    shorter, longer = sorted((left, right), key=len)
    if not shorter or shorter not in longer:
        return False
    return longer.startswith(shorter) or len(shorter) / len(longer) >= 0.5


def _has_latin_letters(value: str) -> bool:
    return any("a" <= character <= "z" for character in value.casefold())


def _names_the_same_place(
    normalized_query: str, place: Place, *, allow_cross_script: bool = False
) -> bool:
    """Reject a nearest-but-unrelated match.

    `max` always yields a candidate, so a name Kakao simply does not have ("마천1치안센터") comes
    back as whatever scored least badly ("웅동파출소", 100 km away, zero characters in common). That
    silently swaps the question's POI for a different one, and every operator downstream then
    computes correctly over the wrong place — far worse than no match at all.
    """

    normalized_name = _search_key(place.name)
    if _containment_is_evidence(normalized_query, normalized_name):
        return True
    if allow_cross_script and _has_latin_letters(normalized_query) != _has_latin_letters(
        normalized_name
    ):
        # A brand transliterated across scripts shares no characters with its Kakao entry at all
        # ("A TWOSOME PLACE" / 투썸플레이스, "Lush" / 러쉬), so character similarity cannot speak
        # for or against it and whatever constrained the candidate set has to. Only the caller that
        # searched a bounded neighbourhood may ask for this.
        return True
    if SequenceMatcher(None, normalized_query, normalized_name).ratio() < NAME_EVIDENCE_FLOOR:
        return False
    # Two branches of one brand agree on everything except the part that names the branch, so the
    # bare-brand retry ("CU" for CU 구로소담점) scores above the floor while denoting a different
    # shop — under the region prior, whichever CU sits nearest the benchmark's centre. Judge the
    # residue between the shared affixes; a residue too short to distinguish anything
    # (CU 가락센트럴점 against Kakao's CU 가락센타점) is a spelling variant and is left alone.
    return distinguishing_similarity(normalized_query, normalized_name) >= NAME_EVIDENCE_FLOOR


def _proximity_score(anchor: Place | None, place: Place) -> float:
    """Negated distance to the anchor, so nearer sorts higher. 0.0 when there is no anchor."""

    if anchor is None:
        return 0.0
    return -haversine_meters(anchor.latitude, anchor.longitude, place.latitude, place.longitude)


def _search_key(value: str) -> str:
    normalized = strip_location_qualifier(value).casefold()
    for source in ("공립작은도서관", "작은도서관", "새마을문고", "도서관"):
        normalized = normalized.replace(source, "문고")
    # 치안센터 and 파출소 are the same institution under two names, and which one a source uses is
    # editorial: OSM writes 연남치안센터 where Kakao lists 연남파출소. Folding them together lets
    # the *distinguishing* part of the name (연남) decide, instead of the institution word.
    normalized = normalized.replace("파출소", "치안센터")
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
    convenience_brands = ("cu", "gs25", "세븐일레븐", "이마트24", "미니스톱")
    requested_brand = next((brand for brand in convenience_brands if brand in query_key), None)
    if requested_brand:
        if requested_brand in name_key:
            return 5
        if "편의점" in category_key:
            return 2
        return -4
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
    # A dataset that had to separate two same-named options appends the address to the name
    # ("버거킹 - 서울특별시 용산구 ..."). Kakao indexes names, not name-plus-address.
    unqualified = strip_location_qualifier(normalized)
    if unqualified != normalized:
        variants.append(unqualified)
    without_company = unqualified.replace("(주)", "").strip()
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
    words = without_company.split()
    deferred_branch_variants: list[str] = []
    if len(words) > 1:
        branch_token = words[-1]
        if branch_token.endswith("점") and len(branch_token) > 1:
            branch_stem = branch_token[:-1]
            descriptor_variant_added = False
            for descriptor in ("센트럴", "사거리", "로데오", "프라자", "타워"):
                if branch_stem.endswith(descriptor) and len(branch_stem) > len(descriptor):
                    variants.append(f"{' '.join(words[:-1])} {branch_stem}")
                    local_name = branch_stem[: -len(descriptor)]
                    variants.append(f"{' '.join(words[:-1])} {local_name}")
                    descriptor_variant_added = True
                    break
            if not descriptor_variant_added:
                deferred_branch_variants.append(f"{' '.join(words[:-1])} {branch_stem}")
            deferred_branch_variants.append(branch_stem)
    # Historical benchmark names may outlive a branch rename. Broader brand forms are only
    # attempted after specific shortened branch forms, so they cannot mask a local match.
    branchless = re.sub(r"\s+\S{1,20}(?:본점|지점|점)$", "", without_company).strip()
    if branchless and branchless != without_company:
        variants.append(branchless)
    variants.extend(deferred_branch_variants)
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
