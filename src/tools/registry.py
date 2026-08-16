from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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


class PlaceSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Korean place name, optionally with a region")
    limit: int = Field(default=5, ge=1, le=15)

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=15)


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
    limit: int = Field(default=15, ge=1, le=15)

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
        return _clamp_int(value, minimum=1, maximum=15)

    @model_validator(mode="after")
    def require_search_selector(self) -> NearbyPlacesArgs:
        if not self.query and not self.category_code:
            raise ValueError("nearby search requires query or category_code")
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
                    "Find normalized Korean places by name. Returns IDs, addresses, "
                    "coordinates and categories.",
                    PlaceSearchArgs,
                    lambda args: self.provider.search_place(args.query, limit=args.limit),
                ),
                ToolDefinition(
                    "geocode",
                    "Convert a Korean address into normalized coordinates and address fields.",
                    GeocodeArgs,
                    lambda args: self.provider.geocode(args.address, limit=args.limit),
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
                    "Read a normalized place previously retrieved in this run by place_id.",
                    PlaceDetailsArgs,
                    lambda args: self.provider.place_details(args.place_id),
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
                    "directions",
                    "Get a normalized driving-route distance and duration summary.",
                    DirectionsArgs,
                    lambda args: self.provider.directions(
                        args.origin,
                        args.destination,
                        mode=args.mode,
                        priority=args.priority,
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
                    ),
                ),
                ToolDefinition(
                    "distance_matrix",
                    "Compute driving distance/duration for an origin-destination matrix or an "
                    "explicit list of ordered route pairs. Individual route failures are isolated.",
                    DistanceMatrixArgs,
                    self._distance_matrix,
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
            anchor_matches = self.provider.search_place(anchor_value, limit=max(args.limit, 5))
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
                        limit=max(args.limit, 5),
                    )
                    if not matches:
                        matches = self.provider.search_place(place_name, limit=max(args.limit, 5))
                else:
                    matches = self.provider.search_place(place_name, limit=max(args.limit, 5))
                best_match = _best_place_match(place_name, matches)
                results.append(
                    {
                        "query": place_name,
                        "place": _jsonable(best_match) if best_match else None,
                        "candidates": _jsonable(matches[: args.limit]),
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

    def score(place: Place) -> tuple[int, float]:
        normalized_name = _search_key(place.name)
        exact = int(normalized_query == normalized_name)
        containment = int(
            normalized_query in normalized_name or normalized_name in normalized_query
        )
        overlap = len(set(normalized_query) & set(normalized_name)) / max(
            len(set(normalized_query)), 1
        )
        return exact * 2 + containment, overlap

    return max(matches, key=score)


def _search_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
