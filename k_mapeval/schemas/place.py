from __future__ import annotations

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

