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

    def activate_context(self, context: str | None) -> None:
        """Bind the evidence the next question is answered from.

        A dataset may ship the retrieval results for each question, which a provider can serve
        instead of calling an API. Providers that answer from a live API ignore this.
        """

        return None

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
