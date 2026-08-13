from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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

