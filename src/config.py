from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str | None = None
    # A ReAct question carries its whole growing trace into every call, and a busy self-hosted
    # endpoint can take minutes to answer one. A short timeout does not make the answer arrive
    # sooner; it only turns a slow answer into a lost question.
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    llm_max_retries: int = Field(default=8, ge=0, le=20)
    llm_retry_backoff_seconds: float = Field(default=5.0, gt=0)
    max_reasoning_steps: int = Field(default=8, ge=1, le=30)
    benchmark_concurrency: int = Field(default=4, ge=1, le=32)
    # Extra attempts for a single question the endpoint failed to serve. The client already retries
    # each request; this catches the case where the endpoint stayed down for a whole question, so
    # one blip does not cost a data point that says nothing about the architecture. 0 disables it.
    benchmark_question_retries: int = Field(default=2, ge=0, le=5)
    benchmark_question_retry_backoff_seconds: float = Field(default=10.0, gt=0)

    kakao_rest_api_key: str = ""
    # Region prior for name lookups, as "lat,lng". Korean POI names repeat across cities, so a
    # nationwide keyword search resolves 선진약국 to 대전 with no way to know better. Biasing every
    # lookup to the benchmark's region is a deployment setting applied identically to both agents,
    # not gold metadata: it says where to look, never which option is right. Blank disables it.
    kakao_search_center: str = ""
    kakao_search_radius_m: int = Field(default=20_000, ge=0, le=20_000)
    kakao_timeout_seconds: float = Field(default=15.0, gt=0)
    kakao_cache_db_path: str = "data/kakao_cache.db"
    kakao_cache_ttl_seconds: int = Field(default=86_400, ge=0)

    @model_validator(mode="after")
    def normalize_blanks(self) -> Settings:
        if self.llm_base_url is not None and not self.llm_base_url.strip():
            self.llm_base_url = None
        return self

    def search_center(self) -> tuple[float, float] | None:
        """`KAKAO_SEARCH_CENTER` as (latitude, longitude), or None when unset."""

        raw = self.kakao_search_center.strip()
        if not raw or self.kakao_search_radius_m <= 0:
            return None
        parts = raw.split(",")
        if len(parts) != 2:
            raise ValueError('KAKAO_SEARCH_CENTER must be "latitude,longitude"')
        try:
            latitude, longitude = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError('KAKAO_SEARCH_CENTER must be "latitude,longitude"') from exc
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("KAKAO_SEARCH_CENTER is outside valid coordinate ranges")
        return latitude, longitude

    def require_llm(self) -> None:
        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", self.llm_api_key),
                ("LLM_MODEL", self.llm_model),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required LLM settings: {', '.join(missing)}")

    def require_kakao(self) -> None:
        if not self.kakao_rest_api_key:
            raise ValueError("Missing required setting: KAKAO_REST_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
