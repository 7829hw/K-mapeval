from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.agent.base import BenchmarkAgent
from src.dataset import BenchmarkItem
from src.logging import log_agent_result, query_log


class EvaluationReport(BaseModel):
    """The three-section report emitted by upstream Spatial-Agent."""

    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any]
    statistics: dict[str, Any]
    results: list[dict[str, Any]]


class Evaluator:
    def __init__(
        self,
        agent: BenchmarkAgent,
        dataset: list[BenchmarkItem],
        *,
        output_dir: str | Path | None = "reports",
        dataset_path: str | Path | None = None,
        log_dir: str | Path = "logs",
        test_mode: str = "full",
    ) -> None:
        self.agent = agent
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.dataset_path = str(dataset_path) if dataset_path else None
        self.log_dir = Path(log_dir)
        self.test_mode = test_mode
        self.report_path: Path | None = None

    def run(self) -> EvaluationReport:
        results: list[dict[str, Any]] = []
        total = len(self.dataset)
        print("=" * 80)
        print(f"Running evaluation on {total} samples")
        print("=" * 80)
        for index, item in enumerate(self.dataset, 1):
            results.append(self._run_single(item, index=index, total=total))
        print()

        statistics = calculate_statistics(results)
        metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "test_mode": self.test_mode,
            "sample_ratio": None,
            "random_seed": None,
            "total_samples": total,
            "dataset_source": self.dataset_path,
        }
        report = EvaluationReport(metadata=metadata, statistics=statistics, results=results)
        self.report_path = self._write_report(report)
        print_summary(statistics)
        if self.report_path is not None:
            print(f"Report saved to: {self.report_path}")
        return report

    def _run_single(
        self,
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
                result = self.agent.answer(question, options)
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
            answer_correct = predicted_option == correct_option
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
                "time": elapsed,
                "error": error,
            }
        except Exception as exc:
            elapsed = time.time() - started
            print(f"[{index}/{total}] ID {item.id!s:>3} | ERROR: {str(exc)[:80]} | {elapsed:.1f}s")
            return {
                "id": item.id,
                "question": question,
                "expected_classification": item.classification,
                "predicted_intent": None,
                "correct_answer": correct_option,
                "correct_answer_text": correct_text,
                "predicted_option": None,
                "predicted_answer": None,
                "intent_correct": False,
                "answer_correct": False,
                "time": elapsed,
                "error": str(exc),
            }

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
            f"{elapsed:.1f}s"
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


def calculate_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    intent_correct = sum(1 for row in results if row.get("intent_correct"))
    answer_correct = sum(1 for row in results if row.get("answer_correct"))
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
        "performance": {
            "average_time_seconds": round(average_time, 3),
            "failed_count": len(failed),
            "failed_ids": failed,
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
    print(f"Average time: {performance['average_time_seconds']:.2f}s")
    if performance["failed_count"]:
        print(f"Failed samples: {performance['failed_ids']}")
    print()
