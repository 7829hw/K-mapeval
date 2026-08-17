from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from src.models import Place, Route

SCHEMA_VERSION = 1
_SCHEMA_LOCK = Lock()


class SQLiteMapCache:
    """Persistent cache for normalized provider responses.

    Only canonical Place and Route payloads are stored. API keys, raw Kakao responses,
    prompts, and agent traces are never written to this database.
    """

    def __init__(self, path: str | Path, *, ttl_seconds: int = 86_400) -> None:
        self.path = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        self.ttl_seconds = ttl_seconds
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.execute("PRAGMA busy_timeout=30000")
        with _SCHEMA_LOCK:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def get_places(self, operation: str, arguments: Mapping[str, Any]) -> list[Place] | None:
        payload = self._get(operation, arguments)
        if payload is None:
            return None
        places = [Place.model_validate(item) for item in payload]
        self.store_places(places)
        return places

    def set_places(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        places: list[Place],
    ) -> None:
        payload = [place.model_dump(mode="json") for place in places]
        self._set(operation, arguments, payload)
        self.store_places(places)

    def get_place(self, place_id: str) -> Place | None:
        row = self._connection.execute(
            "SELECT payload_json, expires_at FROM places WHERE place_id = ?",
            (place_id,),
        ).fetchone()
        if row is None:
            return None
        if self._expired(row[1]):
            with self._connection:
                self._connection.execute("DELETE FROM places WHERE place_id = ?", (place_id,))
            return None
        return Place.model_validate_json(row[0])

    def store_places(self, places: list[Place]) -> None:
        now = int(time.time())
        expires_at = self._expires_at(now)
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO places(place_id, payload_json, updated_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(place_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                [
                    (place.place_id, place.model_dump_json(), now, expires_at)
                    for place in places
                ],
            )

    def get_route(self, operation: str, arguments: Mapping[str, Any]) -> Route | None:
        payload = self._get(operation, arguments)
        return Route.model_validate(payload) if payload is not None else None

    def set_route(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        route: Route,
    ) -> None:
        self._set(operation, arguments, route.model_dump(mode="json"))

    def clear_expired(self) -> int:
        if self.ttl_seconds == 0:
            return 0
        now = int(time.time())
        with self._connection:
            entries = self._connection.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            ).rowcount
            places = self._connection.execute(
                "DELETE FROM places WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            ).rowcount
        return entries + places

    def _get(self, operation: str, arguments: Mapping[str, Any]) -> Any | None:
        cache_key, _ = _cache_key(operation, arguments)
        row = self._connection.execute(
            "SELECT response_json, expires_at FROM cache_entries WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        if self._expired(row[1]):
            with self._connection:
                self._connection.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,)
                )
            return None
        return json.loads(row[0])

    def _set(self, operation: str, arguments: Mapping[str, Any], payload: Any) -> None:
        cache_key, request_json = _cache_key(operation, arguments)
        now = int(time.time())
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO cache_entries(
                    cache_key, operation, request_json, response_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    operation,
                    request_json,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    now,
                    self._expires_at(now),
                ),
            )

    def _expired(self, expires_at: int | None) -> bool:
        return expires_at is not None and expires_at <= int(time.time())

    def _expires_at(self, now: int) -> int | None:
        return None if self.ttl_seconds == 0 else now + self.ttl_seconds

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_cache_entries_expiry
                    ON cache_entries(expires_at);

                CREATE TABLE IF NOT EXISTS places (
                    place_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    expires_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_places_expiry ON places(expires_at);
                """
            )
            self._connection.execute(
                """
                INSERT INTO cache_metadata(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )


def _cache_key(operation: str, arguments: Mapping[str, Any]) -> tuple[str, str]:
    request_json = json.dumps(
        _canonicalize(arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{operation}:{request_json}".encode()).hexdigest()
    return digest, request_json


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return round(value, 7)
    return value
