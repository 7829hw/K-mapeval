from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.agent.base import BenchmarkAgent
from src.dataset import BenchmarkItem
from src.llm import LLMUnavailableError
from src.logging import log_agent_result, query_log

INFRASTRUCTURE_FAILURE = "llm_unavailable"
ABORTED_FAILURE = "run_aborted"

# Straight-line-distance answers are graded with a tolerance, because the dataset's coordinates come
# from OSM while the tool layer measures against Kakao's. A few metres of provenance drift is not a
# reasoning error. Applied by the evaluator to both agents alike, never inside an agent.
DISTANCE_TOLERANCE_M = 20.0
_DISTANCE_VALUE = re.compile(r"([\d,.]+)\s*(km|m)?", re.IGNORECASE)


def _distance_option_metres(text: str | None) -> float | None:
    match = _DISTANCE_VALUE.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return value * 1000 if (match.group(2) or "m").lower() == "km" else value


def _within_distance_tolerance(
    classification: str,
    predicted_text: str | None,
    correct_text: str,
    *,
    tolerance_m: float,
) -> bool:
    if tolerance_m <= 0 or classification != "distance" or predicted_text is None:
        return False
    predicted = _distance_option_metres(predicted_text)
    correct = _distance_option_metres(correct_text)
    if predicted is None or correct is None:
        return False
    return abs(predicted - correct) <= tolerance_m


class EvaluationReport(BaseModel):
    """The three-section report emitted by upstream Spatial-Agent."""

    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any]
    statistics: dict[str, Any]
    results: list[dict[str, Any]]


class Evaluator:
    def __init__(
        self,
        agent: BenchmarkAgent | None,
        dataset: list[BenchmarkItem],
        *,
        agent_factory: Callable[[], AbstractContextManager[BenchmarkAgent]] | None = None,
        max_workers: int = 1,
        output_dir: str | Path | None = "reports",
        dataset_path: str | Path | None = None,
        log_dir: str | Path = "logs",
        test_mode: str = "full",
        agent_type: str | None = None,
        llm_profile: dict[str, Any] | None = None,
        abort_after_llm_failures: int = 0,
        distance_tolerance_m: float = 0.0,
    ) -> None:
        if agent is None and agent_factory is None:
            raise ValueError("Evaluator requires agent or agent_factory")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_workers > 1 and agent_factory is None:
            raise ValueError("Parallel evaluation requires an isolated agent_factory")
        self.agent = agent
        self.agent_factory = agent_factory
        self.max_workers = max_workers
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.dataset_path = str(dataset_path) if dataset_path else None
        self.log_dir = Path(log_dir)
        self.test_mode = test_mode
        self.agent_type = agent_type
        self.llm_profile = dict(llm_profile) if llm_profile else {}
        self.abort_after_llm_failures = abort_after_llm_failures
        self.distance_tolerance_m = distance_tolerance_m
        self.report_path: Path | None = None
        self._failure_lock = Lock()
        self._consecutive_llm_failures = 0
        self._aborted = False

    def run(self) -> EvaluationReport:
        total = len(self.dataset)
        worker_count = min(self.max_workers, total) if total else 1
        self._consecutive_llm_failures = 0
        self._aborted = False
        print("=" * 80)
        print(f"Running evaluation on {total} samples")
        print(f"Concurrent LLM sessions: {worker_count}")
        print("=" * 80)
        if worker_count == 1:
            results = self._run_sequential(total)
        else:
            results = self._run_parallel(total, worker_count)
        print()

        statistics = calculate_statistics(results, aborted=self._aborted)
        metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "test_mode": self.test_mode,
            "sample_ratio": None,
            "random_seed": None,
            "total_samples": total,
            "concurrency": worker_count,
            "dataset_source": self.dataset_path,
            "agent_type": self.agent_type,
            "distance_tolerance_m": self.distance_tolerance_m,
            **self.llm_profile,
        }
        report = EvaluationReport(metadata=metadata, statistics=statistics, results=results)
        self.report_path = self._write_report(report)
        print_summary(statistics)
        if self.report_path is not None:
            print(f"Report saved to: {self.report_path}")
        return report

    def _run_sequential(self, total: int) -> list[dict[str, Any]]:
        if self.agent_factory is not None:
            with self.agent_factory() as agent:
                return self._consume(agent, enumerate(self.dataset, 1), total)
        if self.agent is None:  # guarded by __init__; keeps type narrowing explicit
            raise RuntimeError("Sequential evaluator has no agent")
        return self._consume(self.agent, enumerate(self.dataset, 1), total)

    def _consume(
        self,
        agent: BenchmarkAgent,
        jobs: Any,
        total: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, item in jobs:
            if self._aborted:
                results.append(self._skipped_row(item, index=index, total=total))
                continue
            results.append(self._run_single(agent, item, index=index, total=total))
        return results

    def _run_parallel(self, total: int, worker_count: int) -> list[dict[str, Any]]:
        if self.agent_factory is None:  # guarded by __init__
            raise RuntimeError("Parallel evaluator has no agent factory")

        jobs: Queue[tuple[int, BenchmarkItem] | None] = Queue()
        for index, item in enumerate(self.dataset, 1):
            jobs.put((index, item))
        for _ in range(worker_count):
            jobs.put(None)

        ordered_results: list[dict[str, Any] | None] = [None] * total
        worker_errors: list[BaseException] = []
        error_lock = Lock()

        def worker() -> None:
            try:
                with self.agent_factory() as agent:
                    while True:
                        job = jobs.get()
                        if job is None:
                            return
                        index, item = job
                        ordered_results[index - 1] = (
                            self._skipped_row(item, index=index, total=total)
                            if self._aborted
                            else self._run_single(agent, item, index=index, total=total)
                        )
            except BaseException as exc:
                with error_lock:
                    worker_errors.append(exc)

        threads = [
            Thread(target=worker, name=f"benchmark-worker-{index + 1}")
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if worker_errors:
            raise RuntimeError(f"Benchmark worker failed: {worker_errors[0]}") from worker_errors[0]
        if any(result is None for result in ordered_results):
            raise RuntimeError("Parallel benchmark finished with missing results")
        return [result for result in ordered_results if result is not None]

    def _run_single(
        self,
        agent: BenchmarkAgent,
        item: BenchmarkItem,
        *,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        question, options = item.agent_input()
        correct_option = item.answer
        correct_text = options[correct_option].strip()
        started = time.time()
        try:
            with query_log(
                question,
                item.id,
                log_dir=self.log_dir,
                option_count=len(options),
            ) as logger:
                result = agent.answer(question, options)
                log_agent_result(logger, result, correct_answer=item.answer)
            elapsed = time.time() - started
            predicted_option = result.predicted_answer
            predicted_text = (
                options[predicted_option]
                if predicted_option is not None and 0 <= predicted_option < len(options)
                else None
            )
            error = result.failure_message
            intent_correct = result.predicted_intent == item.classification
            strict_correct = predicted_option == correct_option
            answer_correct = strict_correct or _within_distance_tolerance(
                item.classification,
                predicted_text,
                correct_text,
                tolerance_m=self.distance_tolerance_m,
            )
            self._print_result(
                item,
                index=index,
                total=total,
                predicted_intent=result.predicted_intent,
                predicted_text=predicted_text,
                predicted_option=predicted_option,
                correct_text=correct_text,
                intent_correct=intent_correct,
                answer_correct=answer_correct,
                elapsed=elapsed,
            )
            self._note_failure(result.failure_type)
            return {
                "id": item.id,
                "question": question,
                "expected_classification": item.classification,
                "predicted_intent": result.predicted_intent,
                "correct_answer": correct_option,
                "correct_answer_text": correct_text,
                "predicted_option": predicted_option,
                "predicted_answer": predicted_text,
                "intent_correct": intent_correct,
                "answer_correct": answer_correct,
                "strict_answer_correct": strict_correct,
                "time": elapsed,
                "error": error,
                "failure_type": result.failure_type,
            }
        except Exception as exc:
            elapsed = time.time() - started
            print(
                f"[{index}/{total}] ID {item.id!s:>3} | ERROR: {str(exc)[:80]} | {elapsed:.1f}s",
                flush=True,
            )
            failure_type = (
                INFRASTRUCTURE_FAILURE
                if isinstance(exc, LLMUnavailableError)
                else "agent_reasoning_failure"
            )
            self._note_failure(failure_type)
            return self._failed_row(
                item,
                question,
                correct_text,
                error=f"{type(exc).__name__}: {exc}",
                failure_type=failure_type,
                elapsed=elapsed,
            )

    def _skipped_row(self, item: BenchmarkItem, *, index: int, total: int) -> dict[str, Any]:
        question, options = item.agent_input()
        print(
            f"[{index}/{total}] ID {item.id!s:>3} | SKIPPED: run aborted (LLM endpoint down)",
            flush=True,
        )
        return self._failed_row(
            item,
            question,
            options[item.answer].strip(),
            error="Run aborted before this question: the LLM endpoint was unavailable",
            failure_type=ABORTED_FAILURE,
            elapsed=0.0,
        )

    @staticmethod
    def _failed_row(
        item: BenchmarkItem,
        question: str,
        correct_text: str,
        *,
        error: str,
        failure_type: str,
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "id": item.id,
            "question": question,
            "expected_classification": item.classification,
            "predicted_intent": None,
            "correct_answer": item.answer,
            "correct_answer_text": correct_text,
            "predicted_option": None,
            "predicted_answer": None,
            "intent_correct": False,
            "answer_correct": False,
            "strict_answer_correct": False,
            "time": elapsed,
            "error": error,
            "failure_type": failure_type,
        }

    def _note_failure(self, failure_type: str | None) -> None:
        """Trip the circuit breaker once the endpoint has failed enough questions in a row."""

        if self.abort_after_llm_failures <= 0:
            return
        with self._failure_lock:
            if failure_type == INFRASTRUCTURE_FAILURE:
                self._consecutive_llm_failures += 1
            else:
                self._consecutive_llm_failures = 0
                return
            if self._aborted or self._consecutive_llm_failures < self.abort_after_llm_failures:
                return
            self._aborted = True
        print(
            f"\nAborting run: {self.abort_after_llm_failures} consecutive questions failed on the "
            "LLM endpoint. Remaining questions are skipped; this report is not a benchmark "
            "result.\n",
            flush=True,
        )

    @staticmethod
    def _print_result(
        item: BenchmarkItem,
        *,
        index: int,
        total: int,
        predicted_intent: str | None,
        predicted_text: str | None,
        predicted_option: int | None,
        correct_text: str,
        intent_correct: bool,
        answer_correct: bool,
        elapsed: float,
    ) -> None:
        intent_mark = "OK" if intent_correct else "NO"
        answer_mark = "OK" if answer_correct else "NO"
        shown_pred = predicted_text if predicted_text is not None else f"idx={predicted_option}"
        print(
            f"[{index}/{total}] ID {item.id!s:>3} | "
            f"{item.classification:10s} -> {predicted_intent or 'None':10s} {intent_mark} | "
            f"pred={str(shown_pred)[:24]!r}, correct={correct_text[:24]!r} {answer_mark} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    def _write_report(self, report: EvaluationReport) -> Path | None:
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"test_{timestamp}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        return path


def calculate_statistics(results: list[dict[str, Any]], *, aborted: bool = False) -> dict[str, Any]:
    total = len(results)
    intent_correct = sum(1 for row in results if row.get("intent_correct"))
    answer_correct = sum(1 for row in results if row.get("answer_correct"))
    strict_correct = sum(1 for row in results if row.get("strict_answer_correct"))
    failed = [row["id"] for row in results if row.get("error")]
    by_intent: dict[str, dict[str, Any]] = {}
    for row in results:
        classification = str(row.get("expected_classification", "unknown"))
        stats = by_intent.setdefault(classification, {"total": 0, "correct": 0})
        stats["total"] += 1
        if row.get("answer_correct"):
            stats["correct"] += 1
    for stats in by_intent.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0
    average_time = sum(float(row.get("time", 0.0)) for row in results) / total if total else 0.0
    failure_types = Counter(str(row["failure_type"]) for row in results if row.get("failure_type"))
    infrastructure = failure_types[INFRASTRUCTURE_FAILURE] + failure_types[ABORTED_FAILURE]
    return {
        "intent_classification_accuracy": {
            "correct": intent_correct,
            "total": total,
            "accuracy": round(intent_correct / total, 4) if total else 0.0,
        },
        "answer_accuracy_by_intent": by_intent,
        "overall_answer_accuracy": {
            "correct": answer_correct,
            "total": total,
            "accuracy": round(answer_correct / total, 4) if total else 0.0,
        },
        # Exact option match, ungraded by any tolerance. Kept beside the headline number so a
        # tolerant run stays comparable with a strict one instead of quietly replacing it.
        "strict_answer_accuracy": {
            "correct": strict_correct,
            "total": total,
            "accuracy": round(strict_correct / total, 4) if total else 0.0,
        },
        "performance": {
            "average_time_seconds": round(average_time, 3),
            "failed_count": len(failed),
            "failed_ids": failed,
            "failure_types": dict(failure_types),
        },
        # Accuracy only means something when the questions were actually answered. An outage on the
        # LLM endpoint scores 0% exactly like a wrong answer does, so say so here rather than
        # letting a downed deployment be read as an architecture comparison.
        "run_validity": {
            "valid": infrastructure == 0,
            "llm_unavailable_count": infrastructure,
            "aborted": aborted,
        },
    }


def print_summary(statistics: dict[str, Any]) -> None:
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    intent = statistics["intent_classification_accuracy"]
    overall = statistics["overall_answer_accuracy"]
    performance = statistics["performance"]
    print(
        f"Intent accuracy: {intent['correct']}/{intent['total']} ({intent['accuracy'] * 100:.1f}%)"
    )
    print("Answer accuracy by intent:")
    for classification, stats in sorted(statistics["answer_accuracy_by_intent"].items()):
        print(
            f"  {classification:10s}: {stats['correct']:3d}/{stats['total']:3d} "
            f"({stats['accuracy'] * 100:.1f}%)"
        )
    print(
        f"Overall answer accuracy: {overall['correct']}/{overall['total']} "
        f"({overall['accuracy'] * 100:.1f}%)"
    )
    strict = statistics.get("strict_answer_accuracy")
    if strict and strict["correct"] != overall["correct"]:
        print(
            f"Strict answer accuracy (exact option): {strict['correct']}/{strict['total']} "
            f"({strict['accuracy'] * 100:.1f}%)"
        )
    print(f"Average time: {performance['average_time_seconds']:.2f}s")
    if performance["failed_count"]:
        print(f"Failed samples: {performance['failed_ids']}")
    if performance.get("failure_types"):
        print(f"Failure types: {performance['failure_types']}")
    validity = statistics.get("run_validity", {})
    if not validity.get("valid", True):
        print()
        print("!" * 80)
        print(
            f"INVALID RUN: {validity['llm_unavailable_count']}/{overall['total']} questions never "
            "reached the LLM endpoint."
        )
        print("Accuracy above is not a benchmark result. Fix the endpoint and re-run.")
        print("!" * 80)
    print()
