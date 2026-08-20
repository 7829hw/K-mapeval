"""Run the K-MapEval benchmark over the two upstream agents on Kakao Map.

Both architectures are vendored, not reimplemented: `src/spatial_agent/` is
`ecerybao/Spatial-Agent@6876bba` and the ReAct baseline is `MapEval/MapEval-API@35d481a`'s
five tools driven by its own `initialize_agent` call. The only thing swapped underneath them
is the map API — `src/kakao_maps.py` puts Kakao Local and Kakao Mobility behind Google Maps'
client surface, and both agents read that one client, so a difference between them cannot
come from having been shown different evidence.

`docs/UPSTREAM_MAPPING.md` records every deviation from the two upstreams.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agent import ReactAgent, SpatialAgent
from src.config import Settings
from src.dataset import load_dataset
from src.evaluator import Evaluator
from src.kakao_maps import KakaoMapsClient


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the K-MapEval benchmark on Kakao Map")
    result.add_argument(
        "--agent",
        choices=("react", "spatial", "both"),
        default="both",
        help="Agent architecture to evaluate (default: both)",
    )
    result.add_argument(
        "--dataset",
        default="dataset/seoul_kmapeval_v4_mcq_100.jsonl",
        help=(
            "Benchmark to evaluate. The default is the MapEval-method reproduction benchmark; "
            "seoul_kmapeval_v3 is the compositional one and seoul_kmapeval_v2 the superseded "
            "first reproduction benchmark."
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
    result.add_argument(
        "--verbose-agent",
        action="store_true",
        help="Print the ReAct baseline's chain of thought, as upstream's evaluator does",
    )
    return result


def build_llm(settings: Settings) -> ChatOpenAI:
    """The chat model both architectures use.

    Upstream Spatial-Agent constructs `ChatOpenAI(temperature=0)` from `OPENAI_*`, and the
    MapEval-API baseline takes whatever `LLM.load_model` built. Building one here from this
    repository's `LLM_*` settings is what lets a single run configure both, so an accuracy
    difference is not a difference in decoding.

    The retry budget is deliberately generous. The endpoint is a self-hosted deployment
    behind a reverse proxy: it answers 502/503 while it reloads and takes minutes to answer a
    ReAct call carrying a long trace. Waiting is the only thing that makes an answer arrive.
    """

    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def export_upstream_environment(settings: Settings) -> None:
    """Give the vendored code the environment variable names it reads.

    `spatial_agent/agent/spatial_agent.py` reads `OPENAI_MODEL` at construction time and
    `kakao_maps.py` reads `KAKAO_REST_API_KEY` and the region prior. Exporting this
    repository's settings under those names is how the vendored files stay unmodified.
    """

    os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)
    os.environ["OPENAI_MODEL"] = settings.llm_model
    if settings.llm_base_url:
        os.environ["OPENAI_BASE_URL"] = settings.llm_base_url
    os.environ["KAKAO_REST_API_KEY"] = settings.kakao_rest_api_key
    os.environ["KAKAO_SEARCH_CENTER"] = settings.kakao_search_center
    os.environ["KAKAO_SEARCH_RADIUS_M"] = str(settings.kakao_search_radius_m)
    os.environ["KAKAO_CACHE_DB_PATH"] = settings.kakao_cache_db_path
    os.environ["KAKAO_CACHE_TTL_SECONDS"] = str(settings.kakao_cache_ttl_seconds)
    os.environ["KAKAO_TIMEOUT_SECONDS"] = str(settings.kakao_timeout_seconds)


@contextmanager
def create_agent_session(
    agent_type: str,
    settings: Settings,
    *,
    verbose: bool = False,
) -> Iterator[ReactAgent | SpatialAgent]:
    """Create the resources owned by exactly one benchmark worker thread.

    Each worker gets its own Kakao client and its own agent, so the per-question API and
    cache counters are that worker's and two workers never share an HTTP connection. The
    SQLite response cache underneath is shared and safe to share.
    """

    client = KakaoMapsClient(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
        search_center=settings.search_center(),
        search_radius_m=settings.kakao_search_radius_m,
    )
    try:
        llm = build_llm(settings)
        if agent_type == "react":
            yield ReactAgent(
                llm,
                client,
                verbose=verbose,
                max_iterations=settings.max_reasoning_steps,
            )
        else:
            yield SpatialAgent(client, llm=llm)
    finally:
        client.close()


def run(agent_type: str, args: argparse.Namespace) -> dict:
    settings = Settings()
    settings.require_llm()
    settings.require_kakao()
    export_upstream_environment(settings)

    dataset = load_dataset(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        dataset = [item for item in dataset if item.id in wanted]
        if not dataset:
            raise ValueError("None of --ids were found in the dataset")

    concurrency = settings.benchmark_concurrency if args.concurrency is None else args.concurrency
    if not 1 <= concurrency <= 32:
        raise ValueError("--concurrency must be between 1 and 32")

    report = Evaluator(
        None,
        dataset,
        agent_factory=lambda: create_agent_session(
            agent_type, settings, verbose=args.verbose_agent
        ),
        max_workers=concurrency,
        output_dir=Path(args.output_dir),
        dataset_path=args.dataset,
        test_mode="ids" if args.ids else "full",
        agent_type=agent_type,
        llm_profile={
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "provider": "kakao",
            # Which upstream each side of the run came from, so a report stays attributable
            # after the vendored trees are updated.
            "upstream_spatial_agent": "ecerybao/Spatial-Agent@6876bba",
            "upstream_mapeval_api": "MapEval/MapEval-API@35d481a",
            # Which code answered. A night of fixes produces a shelf of reports whose
            # accuracies differ for reasons no field records, and "which commit was this?" is
            # not reconstructable from the timestamp once two runs overlap. Read once at
            # import, not per run.
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
        ("react", "spatial")
        if args.agent == "both"
        else ("spatial" if args.agent == "spatial" else "react",)
    )
    summaries: dict[str, dict] = {agent_type: run(agent_type, args) for agent_type in agent_types}
    if len(summaries) == 2:
        print(
            "ReAct accuracy="
            f"{summaries['react']['overall_answer_accuracy']['accuracy']:.3f} | "
            "Spatial-Agent accuracy="
            f"{summaries['spatial']['overall_answer_accuracy']['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
