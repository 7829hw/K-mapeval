from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models import Place, Route
from src.tools.spatial import parse_coordinate_literal


def canonical_retrieval_specs(place_type: str) -> list[dict[str, Any]]:
    """Search a canonical place type by its own name, with no provider vocabulary at all.

    A category code is provider knowledge: `PM9` means 약국 to Kakao and nothing to anyone else.
    This is what a provider with no category scheme uses, and what the reasoning core falls back
    to so it can be exercised -- in tests and in the offline grounding replay -- with no provider
    present.
    """

    return [{"query": place_type}]


class ProviderError(RuntimeError):
    """Base class for failures in an external map provider."""


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class PlaceNotFoundError(ProviderError):
    pass


class RouteNotFoundError(ProviderError):
    pass


class UnsupportedTravelModeError(ProviderError):
    pass


class MapProvider(ABC):
    """Provider-neutral interface used by every agent and tool."""

    @property
    @abstractmethod
    def api_call_count(self) -> int: ...

    @property
    def cache_hit_count(self) -> int:
        return 0

    @property
    def cache_miss_count(self) -> int:
        return 0

    @abstractmethod
    def search_place(self, query: str, *, limit: int = 5) -> list[Place]: ...

    @abstractmethod
    def geocode(self, address: str, *, limit: int = 5) -> list[Place]: ...

    def reverse_geocode(
        self, latitude: float, longitude: float, *, limit: int = 5
    ) -> list[Place]:
        raise NotImplementedError("This map provider does not implement reverse_geocode")

    @abstractmethod
    def nearby_search(
        self,
        center: str | Place,
        *,
        query: str | None = None,
        category_code: str | None = None,
        radius_m: int = 2000,
        limit: int = 15,
    ) -> list[Place]: ...

    @abstractmethod
    def place_details(self, place_id: str) -> Place: ...

    def retrieval_specs(self, place_type: str) -> list[dict[str, Any]]:
        """Which searches cover a canonical place type, in this provider's own vocabulary.

        A category code is provider knowledge, not spatial reasoning: `PM9` means 약국 to Kakao
        and nothing to anyone else, and "빵집 also needs a 베이커리 query" is a fact about one
        provider's POI names. The default is the canonical type as a plain query, so a provider
        that has no category scheme needs no ontology of its own.
        """

        return canonical_retrieval_specs(place_type)

    def dereference(self, value: str | Place) -> Place | None:
        """The place this provider already handed out under that reference, or None.

        A `place_id` it minted, coordinates it printed, or the place itself. A *name* is not a
        reference: turning one into a place is what `search_place` is for, and doing it silently
        inside `nearby_search` and `directions` excused the ReAct baseline from the id-threading
        that `mapeval-api/FormattedTools.py` makes unavoidable. Adding a provider means
        implementing this alongside the searches.
        """

        if isinstance(value, Place):
            return value
        # A provider that keeps no index of what it handed out still recognises the coordinates it
        # printed; a name falls through to None, which is the whole point of the contract.
        coordinates = parse_coordinate_literal(value)
        if coordinates is None:
            return None
        latitude, longitude = coordinates
        return Place(
            place_id=f"{latitude},{longitude}",
            name=str(value),
            address="",
            latitude=latitude,
            longitude=longitude,
            category="COORDINATE",
        )

    @abstractmethod
    def directions(
        self,
        origin: str | Place,
        destination: str | Place,
        *,
        mode: str = "driving",
        priority: str = "RECOMMEND",
        waypoints: list[str | Place] | None = None,
        include_steps: bool = False,
    ) -> Route: ...
