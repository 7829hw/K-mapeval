from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Place, Route


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

    @abstractmethod
    def directions(
        self,
        origin: str | Place,
        destination: str | Place,
        *,
        mode: str = "driving",
        priority: str = "RECOMMEND",
    ) -> Route: ...
