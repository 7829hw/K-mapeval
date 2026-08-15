from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models import Place
from src.tools.map import MapProvider


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
    category_code: str | None = Field(
        default=None,
        description="Optional Kakao category code such as SW8 (subway) or FD6 (food)",
    )
    radius_m: int = Field(default=2000, ge=1, le=20000)
    limit: int = Field(default=15, ge=1, le=15)

    @field_validator("radius_m", mode="before")
    @classmethod
    def clamp_radius(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=20_000)

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
                    "place_details",
                    "Read a normalized place previously retrieved in this run by place_id.",
                    PlaceDetailsArgs,
                    lambda args: self.provider.place_details(args.place_id),
                ),
                ToolDefinition(
                    "nearby_places",
                    "Find places around a center, sorted by distance. Supply query "
                    "or category_code.",
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
                    "Get a normalized driving route with distance, duration and guidance steps.",
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
