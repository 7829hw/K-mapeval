from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from src.agent.base import AgentResult

LOGGER_NAME = "spatial_agent"
LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
_CONFIGURE_LOCK = Lock()


def configure_logging() -> logging.Logger:
    """Configure the logger like the upstream Spatial-Agent implementation."""

    logger = logging.getLogger(LOGGER_NAME)
    with _CONFIGURE_LOCK:
        if logger.handlers:
            return logger
        level_name = os.getenv("SPATIAL_AGENT_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(console_handler)
        logger.propagate = False
    return logger


def normalise_filename(text: str, max_length: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text[:max_length] or "query"


@contextmanager
def query_log(
    question: str,
    question_id: str | int | None,
    *,
    log_dir: str | Path = "logs",
    option_count: int = 0,
) -> Iterator[logging.Logger]:
    """Attach the upstream-style UTC timestamped per-query file handler."""

    base_logger = configure_logging()
    # A global logger with multiple temporary handlers would copy concurrent queries into
    # one another's files. Each query therefore gets an isolated logger at the same level.
    logger = logging.Logger(
        f"{LOGGER_NAME}.query.{question_id}.{datetime.now(UTC).timestamp()}",
        level=base_logger.level,
    )
    logger.propagate = False
    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = normalise_filename(" ".join(question.strip().split()[:5]))
    filename = (
        f"{timestamp}_id{question_id}_{slug}.log"
        if question_id is not None
        else f"{timestamp}_{slug}.log"
    )
    path = destination / filename
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logger.level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(handler)
    logger.info("Attached per-query log handler: %s", path)
    logger.info("=" * 80)
    id_text = f"ID: {question_id}" if question_id is not None else "Interactive"
    logger.info("FAST WORKFLOW STARTED | %s | OPTIONS: %s", id_text, option_count)
    logger.info("=" * 80)
    logger.info("Question: %s%s", question[:100], "..." if len(question) > 100 else "")
    logger.info("Timestamp: %s", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("")
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()


def log_agent_result(
    logger: logging.Logger,
    result: AgentResult,
    *,
    correct_answer: int,
) -> None:
    """Write the agent's phases and upstream-style workflow summary."""

    for entry in result.trace:
        stage = str(entry.get("stage", "trace")).upper()
        logger.info("[%s] %s", stage, json.dumps(entry, ensure_ascii=False, default=str))
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "OK WORKFLOW COMPLETED | Total: %.2fs | LLM: %s calls | Kakao API: %s calls",
        result.latency_ms / 1000,
        result.reasoning_steps,
        result.api_calls,
    )
    logger.info(
        "TOKENS | prompt %s | completion %s | total %s | reasoning %s | reasoning text %s chars",
        result.prompt_tokens,
        result.completion_tokens,
        result.total_tokens,
        # A server that does not split the completion reports nothing here, and nothing is what
        # gets written: an estimate printed beside three measured counts reads as a fourth.
        "n/a" if result.reasoning_tokens is None else result.reasoning_tokens,
        result.reasoning_chars,
    )
    logger.info("=" * 80)
    logger.info("Result:")
    logger.info("  - Intent: %s", result.predicted_intent)
    logger.info("  - Predicted: %s", result.predicted_answer)
    logger.info("  - Correct: %s", correct_answer)
    logger.info(
        "  - Status: %s",
        "OK MATCH" if result.predicted_answer == correct_answer else "ERROR MISMATCH",
    )
    if result.failure_message:
        logger.error("  - Error: %s", result.failure_message)
