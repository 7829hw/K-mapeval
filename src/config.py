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
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0, le=10)
    llm_retry_backoff_seconds: float = Field(default=2.0, gt=0)
    max_reasoning_steps: int = Field(default=8, ge=1, le=30)
    benchmark_concurrency: int = Field(default=4, ge=1, le=32)
    # Stop a batch once this many questions in a row die on the LLM endpoint. A run that keeps
    # going through an outage produces a 0% report that reads like a result.
    benchmark_abort_after_llm_failures: int = Field(default=10, ge=0)

    kakao_rest_api_key: str = ""
    kakao_timeout_seconds: float = Field(default=15.0, gt=0)
    kakao_cache_db_path: str = "data/kakao_cache.db"
    kakao_cache_ttl_seconds: int = Field(default=86_400, ge=0)

    @model_validator(mode="after")
    def normalize_blanks(self) -> Settings:
        if self.llm_base_url is not None and not self.llm_base_url.strip():
            self.llm_base_url = None
        return self

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
