from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.agent.base import BenchmarkAgent
from src.dataset import BenchmarkItem
from src.metrics import summarize


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_type: str
    dataset_path: str | None = None
    completed_questions: int
    total_questions: int
    is_complete: bool
    summary: dict[str, Any]
    records: list[dict[str, Any]]


class Evaluator:
    def __init__(
        self,
        agent: BenchmarkAgent,
        dataset: list[BenchmarkItem],
        *,
        output_dir: str | Path | None = None,
        dataset_path: str | Path | None = None,
    ) -> None:
        self.agent = agent
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.dataset_path = str(dataset_path) if dataset_path else None

    def run(self) -> EvaluationReport:
        records: list[dict[str, Any]] = []
        total = len(self.dataset)
        report = self._build_report(records, total=total)
        for index, item in enumerate(self.dataset, 1):
            print(
                f"[{self.agent.agent_type}] QA {index}/{total} 실행 중 | id={item.id}",
                flush=True,
            )
            question, options = item.agent_input()
            result = self.agent.answer(question, options)
            record = {
                "question_id": item.id,
                "question": item.question,
                "options": item.options,
                "classification": item.classification,
                "agent_type": result.agent_type,
                "predicted_answer": result.predicted_answer,
                "gold_answer": item.answer,
                "correct": result.predicted_answer == item.answer,
                "tool_calls": result.tool_calls,
                "api_calls": result.api_calls,
                "cache_hits": result.cache_hits,
                "cache_misses": result.cache_misses,
                "reasoning_steps": result.reasoning_steps,
                "latency_ms": result.latency_ms,
                "failure_type": result.failure_type,
                "failure_message": result.failure_message,
                "response": result.response,
                "trace": result.trace,
            }
            records.append(record)
            self._write_question_log(record)
            report = self._build_report(records, total=total)
            self._write_report(report)
            outcome = (
                f"실패({result.failure_type})"
                if result.failure_type
                else "정답"
                if record["correct"]
                else "오답"
            )
            print(
                f"[{self.agent.agent_type}] QA {index}/{total} 완료 "
                f"({index / total:.1%}) | id={item.id} | {outcome} | "
                f"{result.latency_ms:.0f}ms",
                flush=True,
            )
        return report

    def _build_report(
        self,
        records: list[dict[str, Any]],
        *,
        total: int,
    ) -> EvaluationReport:
        completed = len(records)
        return EvaluationReport(
            agent_type=self.agent.agent_type,
            dataset_path=self.dataset_path,
            completed_questions=completed,
            total_questions=total,
            is_complete=completed == total,
            summary=summarize(records),
            records=list(records),
        )

    def _write_question_log(self, record: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        log_dir = self.output_dir / "questions" / str(record["agent_type"])
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{record['question_id']}.json"
        _atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2))

    def _write_report(self, report: EvaluationReport) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{report.agent_type}_report.json"
        _atomic_write(path, report.model_dump_json(indent=2))


def _atomic_write(path: Path, content: str) -> None:
    """Replace a log only after its complete contents have reached a temporary file."""

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
