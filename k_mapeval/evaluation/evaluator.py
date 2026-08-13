from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from k_mapeval.agents.base import BenchmarkAgent
from k_mapeval.evaluation.dataset import BenchmarkItem
from k_mapeval.evaluation.metrics import summarize


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_type: str
    summary: dict[str, Any]
    records: list[dict[str, Any]]


class Evaluator:
    def __init__(
        self,
        agent: BenchmarkAgent,
        dataset: list[BenchmarkItem],
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.agent = agent
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None

    def run(self) -> EvaluationReport:
        records: list[dict[str, Any]] = []
        for item in self.dataset:
            question, options = item.agent_input()
            result = self.agent.answer(question, options)
            record = {
                "question_id": item.id,
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
        report = EvaluationReport(
            agent_type=self.agent.agent_type,
            summary=summarize(records),
            records=records,
        )
        self._write_report(report)
        return report

    def _write_question_log(self, record: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        log_dir = self.output_dir / "questions" / str(record["agent_type"])
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{record['question_id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_report(self, report: EvaluationReport) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{report.agent_type}_report.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
