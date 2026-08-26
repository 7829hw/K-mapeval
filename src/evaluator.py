from __future__ import annotations

import json
import random
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
from src.dataset import BenchmarkItem, resolve_mapeval_class
from src.llm import LLMContextOverflowError, LLMOutputTruncatedError, LLMUnavailableError
from src.logging import log_agent_result, log_trace_entry, query_log

INFRASTRUCTURE_FAILURE = "llm_unavailable"
PROVIDER_FAILURE = "provider_failure"
# A token ceiling cut the completion off. Never retried: the ceiling is a setting, so asking the
# same question again under it only spends the tokens again.
TRUNCATION_FAILURE = "llm_output_truncated"
# The prompt outgrew the context window. Never retried either: the same prompt is the same length.
CONTEXT_OVERFLOW_FAILURE = "llm_context_overflow"
# The loop used its whole step budget without reaching an answer. A miss, the way upstream counts
# it, but not an unreadable answer and not the map's fault.
ITERATION_LIMIT_FAILURE = "iteration_limit"
# Spatial-Agent drafted, repaired and re-validated, and no graph passed the paper's constraints.
# Nothing executed. Kept apart from `agent_reasoning_failure`, which is this port raising while
# running one: the first is a result about the architecture, the second is a defect, and pooling
# them made the pooled count read as the first.
GRAPH_VALIDATION_FAILURE = "graph_validation_failure"
# Provider errors that say the API could not answer right now, as opposed to answering that the
# place does not exist. Only these are worth asking again for; a PlaceNotFoundError is evidence.
TRANSIENT_PROVIDER_ERRORS = ("ProviderTimeoutError", "ProviderRateLimitError")


def is_transient_failure(row: dict[str, Any]) -> bool:
    """Whether a question failed because an API could not answer, not because of its answer.

    Retrying anything else would re-roll the architecture under measurement: a question the agent
    reasoned its way to the wrong end of, or one whose place genuinely is not in Kakao, has to keep
    the result it earned.
    """

    failure_type = row.get("failure_type")
    if failure_type == INFRASTRUCTURE_FAILURE:
        return True
    if failure_type != PROVIDER_FAILURE:
        return False
    return str(row.get("error") or "").startswith(TRANSIENT_PROVIDER_ERRORS)


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
        question_retries: int = 0,
        question_retry_backoff_seconds: float = 10.0,
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
        self.question_retries = max(0, question_retries)
        self.question_retry_backoff_seconds = question_retry_backoff_seconds
        self.report_path: Path | None = None

    def run(self) -> EvaluationReport:
        total = len(self.dataset)
        worker_count = min(self.max_workers, total) if total else 1
        print("=" * 80)
        print(f"Running evaluation on {total} samples")
        print(f"Concurrent LLM sessions: {worker_count}")
        print("=" * 80)
        if worker_count == 1:
            results = self._run_sequential(total)
        else:
            results = self._run_parallel(total, worker_count)
        print()

        statistics = calculate_statistics(results)
        metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "test_mode": self.test_mode,
            "sample_ratio": None,
            "random_seed": None,
            "total_samples": total,
            "concurrency": worker_count,
            "dataset_source": self.dataset_path,
            "agent_type": self.agent_type,
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
                        ordered_results[index - 1] = self._run_single(
                            agent, item, index=index, total=total
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
        """Answer one question, asking again when an API — not the agent — is what failed."""

        attempt = 1
        while True:
            row = self._attempt_single(agent, item, index=index, total=total)
            if attempt > self.question_retries or not is_transient_failure(row):
                break
            delay = self._retry_delay(attempt)
            print(
                f"[{index}/{total}] ID {item.id!s:>3} | RETRY {attempt}/{self.question_retries} "
                f"in {delay:.1f}s: {str(row.get('error'))[:60]}",
                flush=True,
            )
            time.sleep(delay)
            attempt += 1
        row["attempts"] = attempt
        return row

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, so concurrent workers do not retry in lockstep."""

        return self.question_retry_backoff_seconds * (2 ** (attempt - 1)) * (0.5 + random.random())

    def _attempt_single(
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
                agent_type=agent.agent_type,
            ) as logger:
                # Written as the agent works, not once it is done: a question can run for
                # minutes, and a log that appears only on return says nothing about the one
                # currently in flight and nothing at all about a question that never returns.
                streamed = 0

                def emit(entry: dict[str, Any], logger: Any = logger) -> None:
                    nonlocal streamed
                    streamed += 1
                    log_trace_entry(logger, entry)

                agent.trace_sink = emit
                try:
                    result = agent.answer(question, options)
                finally:
                    agent.trace_sink = None
                log_agent_result(
                    logger, result, correct_answer=item.answer, already_logged=streamed
                )
            elapsed = time.time() - started
            predicted_option = result.predicted_answer
            predicted_text = (
                options[predicted_option]
                if predicted_option is not None and 0 <= predicted_option < len(options)
                else None
            )
            error = result.failure_message
            answer_correct = predicted_option == correct_option
            self._print_result(
                item,
                index=index,
                total=total,
                predicted_text=predicted_text,
                predicted_option=predicted_option,
                correct_text=correct_text,
                answer_correct=answer_correct,
                elapsed=elapsed,
            )
            return {
                "id": item.id,
                "question": question,
                "expected_classification": item.classification,
                # Three labels, three axes, one property of the question each: the MapEval task
                # category the paper's table is indexed by, the measurement type, and the
                # generator template. None of them is derivable from another.
                "expected_mapeval_class": resolve_mapeval_class(item),
                "template_id": item.template_id,
                "predicted_intent": result.predicted_intent,
                "correct_answer": correct_option,
                "correct_answer_text": correct_text,
                "predicted_option": predicted_option,
                "predicted_answer": predicted_text,
                "answer_correct": answer_correct,
                "time": elapsed,
                # Deltas read off the shared registry/provider around this question. Without them
                # a report cannot say whether an answer came from the tools or from the model.
                "tool_calls": result.tool_calls,
                "api_calls": result.api_calls,
                "cache_hits": result.cache_hits,
                "cache_misses": result.cache_misses,
                "reasoning_steps": result.reasoning_steps,
                # What the question cost at the endpoint. `reasoning_tokens` is null on a server
                # that does not split the completion, and `reasoning_chars` is the thinking text
                # that came back and never reached the parser -- measurable everywhere.
                "llm_calls": result.llm_calls,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "reasoning_chars": result.reasoning_chars,
                # Intermediate failures are not question failures: an architecture can recover
                # or its evaluator can answer from another branch.  Preserve both the count and
                # the compact details so aggregate accuracy cannot hide a broken execution path.
                "execution_error_count": len(result.execution_errors),
                "execution_errors": result.execution_errors,
                "error": error,
                "failure_type": result.failure_type,
                "attempts": 1,
            }
        except Exception as exc:
            elapsed = time.time() - started
            print(
                f"[{index}/{total}] ID {item.id!s:>3} | ERROR: {str(exc)[:80]} | {elapsed:.1f}s",
                flush=True,
            )
            if isinstance(exc, LLMUnavailableError):
                failure_type = INFRASTRUCTURE_FAILURE
            elif isinstance(exc, LLMOutputTruncatedError):
                failure_type = TRUNCATION_FAILURE
            elif isinstance(exc, LLMContextOverflowError):
                failure_type = CONTEXT_OVERFLOW_FAILURE
            else:
                failure_type = "agent_reasoning_failure"
            return self._failed_row(
                item,
                question,
                correct_text,
                error=f"{type(exc).__name__}: {exc}",
                failure_type=failure_type,
                elapsed=elapsed,
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
        attempts: int = 1,
    ) -> dict[str, Any]:
        return {
            "id": item.id,
            "question": question,
            "expected_classification": item.classification,
            "expected_mapeval_class": resolve_mapeval_class(item),
            "template_id": item.template_id,
            "predicted_intent": None,
            "correct_answer": item.answer,
            "correct_answer_text": correct_text,
            "predicted_option": None,
            "predicted_answer": None,
            "answer_correct": False,
            "time": elapsed,
            "tool_calls": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "reasoning_steps": 0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": None,
            "reasoning_chars": 0,
            "execution_error_count": 0,
            "execution_errors": [],
            "error": error,
            "failure_type": failure_type,
            "attempts": attempts,
        }

    @staticmethod
    def _print_result(
        item: BenchmarkItem,
        *,
        index: int,
        total: int,
        predicted_text: str | None,
        predicted_option: int | None,
        correct_text: str,
        answer_correct: bool,
        elapsed: float,
    ) -> None:
        answer_mark = "OK" if answer_correct else "NO"
        shown_pred = predicted_text if predicted_text is not None else f"idx={predicted_option}"
        print(
            f"[{index}/{total}] ID {item.id!s:>3} | "
            f"{item.classification:10s} | "
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


def _accuracy_by(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """Answer accuracy grouped by one dataset label, in the label's own sort order."""

    grouped: dict[str, dict[str, Any]] = {}
    for row in results:
        # A v1/v2/v3 row carries no `template_id`, and a row whose label the evaluator never
        # wrote reads as "unknown" rather than vanishing from the total.
        key = str(row.get(field) or "unknown")
        stats = grouped.setdefault(key, {"total": 0, "correct": 0})
        stats["total"] += 1
        if row.get("answer_correct"):
            stats["correct"] += 1
    for stats in grouped.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0
    return dict(sorted(grouped.items()))


def calculate_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    answer_correct = sum(1 for row in results if row.get("answer_correct"))
    failed = [row["id"] for row in results if row.get("error")]
    # Answer accuracy split by the *dataset's* labels, which are properties of the question and
    # not of anything an agent predicted. It was called "by intent", which is how scoring an
    # agent's intent against it came to look like a measurement in the first place.
    #
    # Three axes over the same rows, so the extra ones cost one pass each and no LLM calls:
    #   mapeval_class -- MapEval-API's four task categories, the axis the paper's table is
    #     indexed by, plus this port's own `unanswerable`. Report that one as its own row: it is
    #     an addition to the paper's four, not a member of them, and folding it in would compare
    #     a five-category number against a four-category one.
    #   classification -- what is measured. Two `nearby` questions can be a nearest-of-a-kind and
    #     a count-within-600m, and only this tells them apart.
    #   template_id  -- the generator family, which is the grain error analysis actually needs.
    by_class = _accuracy_by(results, "expected_classification")
    by_mapeval_class = _accuracy_by(results, "expected_mapeval_class")
    by_template = _accuracy_by(results, "template_id")
    average_time = sum(float(row.get("time", 0.0)) for row in results) / total if total else 0.0
    failure_types = Counter(str(row["failure_type"]) for row in results if row.get("failure_type"))
    retried = [row for row in results if int(row.get("attempts") or 1) > 1]
    recovered = [row["id"] for row in retried if not is_transient_failure(row)]
    execution_error_rows = [row for row in results if int(row.get("execution_error_count") or 0)]
    execution_error_operators = Counter(
        str(error.get("operator") or "unknown")
        for row in results
        for error in (row.get("execution_errors") or [])
        if isinstance(error, dict)
    )
    return {
        "answer_accuracy_by_class": by_class,
        "answer_accuracy_by_mapeval_class": by_mapeval_class,
        "answer_accuracy_by_template": by_template,
        "overall_answer_accuracy": {
            "correct": answer_correct,
            "total": total,
            "accuracy": round(answer_correct / total, 4) if total else 0.0,
        },
        "performance": {
            "average_time_seconds": round(average_time, 3),
            "failed_count": len(failed),
            "failed_ids": failed,
            "failure_types": dict(failure_types),
            # A question the endpoint failed and we asked again for. Recovered ones carry a real
            # result; the rest were unanswerable however many times we asked.
            "retried_question_count": len(retried),
            "retry_recovered_ids": recovered,
            # Questions that ran out of patience rather than out of evidence. Reported as a plain
            # count: it belongs in the write-up next to the accuracy, not in a verdict here.
            "llm_unavailable_count": failure_types[INFRASTRUCTURE_FAILURE],
            # Questions the serving side's output limit ended rather than the map or the model.
            # An accuracy with any of these in it is partly a measurement of that limit.
            "llm_output_truncated_count": failure_types[TRUNCATION_FAILURE],
            # Questions whose prompt outgrew the context window before the model could answer.
            "llm_context_overflow_count": failure_types[CONTEXT_OVERFLOW_FAILURE],
            # Questions that ran out of steps before answering. Read beside the step budget in the
            # metadata: a family whose questions need more tool calls than the budget allows is
            # measuring the budget, not the architecture.
            "iteration_limit_count": failure_types[ITERATION_LIMIT_FAILURE],
            # Questions no valid GeoFlow graph was produced for, after the draft and the repair.
            "graph_validation_failure_count": failure_types[GRAPH_VALIDATION_FAILURE],
            # A question can still answer after one branch failed.  Keep this beside accuracy so
            # those failures do not disappear merely because the final option happened to match.
            "execution_error_question_count": len(execution_error_rows),
            "execution_error_question_ids": [row["id"] for row in execution_error_rows],
            "execution_error_step_count": sum(
                int(row.get("execution_error_count") or 0) for row in results
            ),
            "execution_error_operators": dict(execution_error_operators),
            # Summed over the questions, so a run can be read as "how much did it look up".
            "tool_calls": sum(int(row.get("tool_calls") or 0) for row in results),
            "api_calls": sum(int(row.get("api_calls") or 0) for row in results),
            "cache_hits": sum(int(row.get("cache_hits") or 0) for row in results),
            "cache_misses": sum(int(row.get("cache_misses") or 0) for row in results),
            "reasoning_steps": sum(int(row.get("reasoning_steps") or 0) for row in results),
            **_token_totals(results),
        },
    }


def _token_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """What the run cost at the endpoint, and how much of it was thinking.

    Two agents that score the same are not the same result if one spent three times the tokens
    getting there, and a reasoning model spends most of a completion on text the parser never
    sees. `reasoning_tokens` stays null unless the server reported it on every row that had a
    completion: summing a column the server filled in sometimes would read as a total.
    """

    reported = [row.get("reasoning_tokens") for row in results]
    complete = results and all(value is not None for value in reported)
    return {
        "llm_calls": sum(int(row.get("llm_calls") or 0) for row in results),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in results),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in results),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in results),
        "reasoning_tokens": (
            sum(int(value or 0) for value in reported) if complete else None
        ),
        "reasoning_chars": sum(int(row.get("reasoning_chars") or 0) for row in results),
    }


def _print_accuracy_rows(rows: dict[str, Any], *, width: int = 12) -> None:
    for label, stats in sorted(rows.items()):
        print(
            f"  {label:{width}s}: {stats['correct']:3d}/{stats['total']:3d} "
            f"({stats['accuracy'] * 100:.1f}%)"
        )


def print_summary(statistics: dict[str, Any]) -> None:
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    overall = statistics["overall_answer_accuracy"]
    performance = statistics["performance"]
    # `unanswerable` is printed under its own heading rather than as a fifth MapEval category:
    # it is this port's addition to the paper's four, and a mean over five categories is not
    # comparable to a mean over four.
    mapeval = dict(statistics.get("answer_accuracy_by_mapeval_class") or {})
    added = {name: mapeval.pop(name) for name in list(mapeval) if name == "unanswerable"}
    if mapeval:
        print("Answer accuracy by MapEval task category:")
        _print_accuracy_rows(mapeval)
    if added:
        print("Answer accuracy on this port's added category (not one of MapEval's four):")
        _print_accuracy_rows(added)
    print("Answer accuracy by measurement type:")
    _print_accuracy_rows(statistics["answer_accuracy_by_class"])
    by_template = statistics.get("answer_accuracy_by_template") or {}
    if by_template:
        print("Answer accuracy by generator template:")
        _print_accuracy_rows(by_template, width=32)
    print(
        f"Overall answer accuracy: {overall['correct']}/{overall['total']} "
        f"({overall['accuracy'] * 100:.1f}%)"
    )
    print(f"Average time: {performance['average_time_seconds']:.2f}s")
    total_tokens = performance.get("total_tokens") or 0
    if total_tokens:
        answered = overall["total"] or 1
        reasoning = performance.get("reasoning_tokens")
        thinking = (
            f"{reasoning} reasoning"
            if reasoning is not None
            else f"{performance.get('reasoning_chars') or 0} reasoning chars (server reports "
            "no reasoning-token split)"
        )
        print(
            f"Tokens: {total_tokens} total "
            f"({performance.get('prompt_tokens') or 0} prompt + "
            f"{performance.get('completion_tokens') or 0} completion) over "
            f"{performance.get('llm_calls') or 0} LLM calls, "
            f"{total_tokens / answered:.0f} per question | {thinking}"
        )
    truncated = performance.get("llm_output_truncated_count") or 0
    if truncated:
        print(
            f"The endpoint's output limit cut off {truncated} question(s): their answers were "
            "never written, so this accuracy is partly a measurement of that limit. Raise it on "
            "the serving side."
        )
    overflowed = performance.get("llm_context_overflow_count") or 0
    if overflowed:
        print(
            f"The prompt outgrew the context window on {overflowed} question(s): the model never "
            "saw the whole question. Widen the window on the serving side or shorten what the "
            "agent carries into a call."
        )
    stopped = performance.get("iteration_limit_count") or 0
    if stopped:
        print(
            f"Ran out of steps on {stopped} question(s) without answering. Upstream counts these "
            "as misses; check whether the family can be answered within the step budget at all."
        )
    unbuildable = performance.get("graph_validation_failure_count") or 0
    if unbuildable:
        print(
            f"No valid GeoFlow graph on {unbuildable} question(s), after the draft and the "
            "repair. Nothing executed on those: read them as the planner's ceiling, not as a "
            "wrong answer."
        )
    execution_error_steps = performance.get("execution_error_step_count") or 0
    if execution_error_steps:
        print(
            f"Intermediate execution errors: {execution_error_steps} step(s) across "
            f"{performance.get('execution_error_question_count') or 0} question(s) | "
            f"{performance.get('execution_error_operators') or {}}"
        )
    if performance["failed_count"]:
        print(f"Failed samples: {performance['failed_ids']}")
    if performance.get("failure_types"):
        print(f"Failure types: {performance['failure_types']}")
    if performance.get("retried_question_count"):
        recovered = performance.get("retry_recovered_ids") or []
        print(
            f"Retried after an API failure: {performance['retried_question_count']} "
            f"({len(recovered)} recovered)"
        )
    unavailable = performance.get("llm_unavailable_count", 0)
    if unavailable:
        print(
            f"Never answered by the LLM endpoint: {unavailable}/{overall['total']} "
            "(counted as incorrect above)"
        )
    print()
