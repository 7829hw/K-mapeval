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
    result.add_argument(
        "--dataset",
        default="dataset/seoul_kmapeval_v2_mcq_100.jsonl",
        help=(
            "Benchmark to evaluate. The default is the Kakao-grounded reproduction benchmark; "
            "the context-cache benchmark is dataset/seoul_mapeval_v1_mcq_100.jsonl."
        ),
    )
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
    result.add_argument(
        "--react-tools",
        choices=("mapeval", "full"),
        default="mapeval",
        help=(
            "Tool surface for the ReAct baseline. 'mapeval' (default) gives ReAct the five "
            "primitives MapEval's own baseline has, which is the comparison the paper reports. "
            "'full' shares this repository's whole registry with Spatial-Agent — an ablation "
            "asking whether the graph adds anything on top of strong aggregation tools, not the "
            "paper's question. Recorded in the report metadata either way."
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
    agent_type: str,
    settings: Settings,
    provider_kind: str,
    contexts: list[str],
    react_tools: str = "mapeval",
) -> Iterator[ReactAgent | SpatialAgent]:
    """Create resources owned by exactly one benchmark worker thread.

    The two agents get different tool surfaces, because the tool surface is part of the
    architecture under test. ReAct gets the five primitives MapEval's own baseline is given;
    Spatial-Agent gets this registry's aggregations and its local operators, which is the
    arrangement upstream has (`spatial-agent/src/tools/google_maps.py` carries a distance matrix
    that `mapeval-api/Evaluator2.py` never hands its baseline). `react_tools="full"` restores the
    shared surface as an ablation. Report which was used.
    """

    provider = build_provider(provider_kind, settings, contexts)
    llm: OpenAIChatClient | None = None
    try:
        llm = OpenAIChatClient(settings)
        allowed = (
            ToolRegistry.MAPEVAL_BASELINE_TOOLS
            if agent_type == "react" and react_tools != "full"
            else None
        )
        tools = ToolRegistry(provider, allowed=allowed)
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
            agent_type, settings, provider_kind, contexts, args.react_tools
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
            # Which tool surface the ReAct baseline had. A run is only comparable to the paper
            # when this says "mapeval", and only comparable across our own agents when "full".
            "react_tools": args.react_tools,
            # Which code answered. A night of fixes produces a shelf of reports whose accuracies
            # differ for reasons no field records, and "which commit was this?" is not
            # reconstructable from the timestamp once two runs overlap. Read once at import, not
            # per run: the process holds the code it loaded at startup, and a commit landing
            # while `--agent both` is halfway through would otherwise be recorded as the code
            # that answered the second half.
            "code_revision": CODE_REVISION,
        },
        question_retries=settings.benchmark_question_retries,
        question_retry_backoff_seconds=settings.benchmark_question_retry_backoff_seconds,
    ).run()
    return report.statistics


def _code_revision() -> str | None:
    """The commit this run's code came from, read from the git directory, never a subprocess."""

    head = Path(__file__).resolve().parent / ".git" / "HEAD"
    try:
        pointer = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not pointer.startswith("ref:"):
        return pointer[:12] or None
    reference = head.parent / pointer.split(" ", 1)[1].strip()
    try:
        return reference.read_text(encoding="utf-8").strip()[:12] or None
    except OSError:
        pass
    packed = head.parent / "packed-refs"
    try:
        lines = packed.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    wanted = pointer.split(" ", 1)[1].strip()
    for line in lines:
        commit, _, name = line.partition(" ")
        if name.strip() == wanted:
            return commit[:12]
    return None


# Read once, at import: this is the revision whose code this process is running.
CODE_REVISION = _code_revision()


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
