from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Place(BaseModel):
    """Provider-neutral place returned to both agent architectures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    place_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str = ""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    category: str = ""
    phone: str = ""
    place_url: str = ""
    rating: float | None = Field(default=None, ge=0, le=5)
    price_level: str | None = None
    opening_hours: dict[str, Any] | None = None
    timezone: str | None = None
    is_open: bool | None = None


class RouteStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str
    road_name: str = ""
    distance_m: int = Field(default=0, ge=0)
    duration_s: int = Field(default=0, ge=0)


class Route(BaseModel):
    """Provider-neutral route returned to both agent architectures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: str
    destination: str
    distance_m: int = Field(ge=0)
    duration_s: int = Field(ge=0)
    steps: tuple[RouteStep, ...] = ()
    waypoints: tuple[str, ...] = ()
