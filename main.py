from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.agent import ReactAgent, SpatialAgent
from src.config import Settings
from src.dataset import load_dataset
from src.evaluator import Evaluator
from src.llm import OpenAIChatClient
from src.tools import ContextMapProvider, KakaoMapProvider, MapProvider, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the K-MapEval benchmark")
    result.add_argument(
        "--agent",
        choices=("react", "spatial", "both"),
        default="both",
        help="Agent architecture to evaluate (default: both)",
    )
    result.add_argument("--dataset", default="dataset/seoul_mapeval_v1_mcq_100.jsonl")
    result.add_argument(
        "--provider",
        choices=("auto", "context", "hybrid", "kakao"),
        default="auto",
        help=(
            "Where tools get their evidence: the corpus built from the dataset's contexts, that "
            "corpus with a live Kakao fallback for what it does not hold (hybrid, upstream "
            "Spatial-Agent's arrangement), or Kakao alone (default: auto, which picks context "
            "when every row carries one)"
        ),
    )
    result.add_argument("--output-dir", default="reports")
    result.add_argument("--ids", nargs="*", help="Optional question IDs")
    result.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrent question/LLM sessions (default: BENCHMARK_CONCURRENCY or 4)",
    )
    return result


def build_provider(
    provider_kind: str, settings: Settings, contexts: list[str]
) -> MapProvider:
    """Build the evidence source both architectures share for this run.

    `context` answers from the benchmark's own corpus alone; `hybrid` adds the Kakao fallback
    upstream Spatial-Agent uses for what its cache does not hold.
    """

    if provider_kind == "context":
        return ContextMapProvider(contexts)
    kakao = KakaoMapProvider(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
        search_center=settings.search_center(),
        search_radius_m=settings.kakao_search_radius_m,
    )
    if provider_kind == "hybrid":
        return ContextMapProvider(contexts, fallback=kakao)
    return kakao


@contextmanager
def create_agent_session(
    agent_type: str, settings: Settings, provider_kind: str, contexts: list[str]
) -> Iterator[ReactAgent | SpatialAgent]:
    """Create resources owned by exactly one benchmark worker thread."""

    provider = build_provider(provider_kind, settings, contexts)
    llm: OpenAIChatClient | None = None
    try:
        llm = OpenAIChatClient(settings)
        tools = ToolRegistry(provider)
        agent = (
            ReactAgent(llm, tools, max_steps=settings.max_reasoning_steps)
            if agent_type == "react"
            else SpatialAgent(llm, tools, max_steps=settings.max_reasoning_steps)
        )
        yield agent
    finally:
        try:
            if llm is not None:
                llm.close()
        finally:
            provider.close()


def resolve_provider_kind(requested: str, dataset: list) -> str:
    """Choose the evidence source, and refuse to silently answer from the wrong one."""

    with_context = sum(1 for item in dataset if item.context)
    if requested in ("context", "hybrid"):
        if with_context != len(dataset):
            raise ValueError(
                f"--provider {requested} needs a context on every row; "
                f"{with_context}/{len(dataset)} have one"
            )
        return requested
    if requested == "kakao":
        return "kakao"
    return "context" if with_context == len(dataset) else "kakao"


def run(agent_type: str, args: argparse.Namespace) -> dict:
    settings = Settings()
    settings.require_llm()
    dataset = load_dataset(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        dataset = [item for item in dataset if item.id in wanted]
        if not dataset:
            raise ValueError("None of --ids were found in the dataset")
    provider_kind = resolve_provider_kind(args.provider, dataset)
    if provider_kind in ("kakao", "hybrid"):
        settings.require_kakao()
    contexts = [item.context for item in dataset if item.context]
    print(f"Evidence source: {provider_kind} ({len(contexts)} contexts in the corpus)")
    concurrency = settings.benchmark_concurrency if args.concurrency is None else args.concurrency
    if not 1 <= concurrency <= 32:
        raise ValueError("--concurrency must be between 1 and 32")
    report = Evaluator(
        None,
        dataset,
        agent_factory=lambda: create_agent_session(
            agent_type, settings, provider_kind, contexts
        ),
        max_workers=concurrency,
        output_dir=Path(args.output_dir),
        dataset_path=args.dataset,
        test_mode="ids" if args.ids else "full",
        agent_type=agent_type,
        llm_profile={
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "provider": provider_kind,
        },
        question_retries=settings.benchmark_question_retries,
        question_retry_backoff_seconds=settings.benchmark_question_retry_backoff_seconds,
    ).run()
    return report.statistics


def main() -> None:
    args = build_parser().parse_args()
    agent_types = (
        ("react", "spatial_agent")
        if args.agent == "both"
        else ("spatial_agent" if args.agent == "spatial" else "react",)
    )
    summaries: dict[str, dict] = {agent_type: run(agent_type, args) for agent_type in agent_types}
    if len(summaries) == 2:
        print(
            "ReAct accuracy="
            f"{summaries['react']['overall_answer_accuracy']['accuracy']:.3f} | "
            "Spatial-Agent accuracy="
            f"{summaries['spatial_agent']['overall_answer_accuracy']['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
