"""The benchmark's no-tool floor: the same model and prompt with the map taken away.

An accuracy means nothing on its own. A benchmark whose questions a model can answer from its own
knowledge scores high for a reason that has nothing to do with the agent, and the only way to know
is to measure the floor: same model, same options, no tools, closed book. What the map explains is
`accuracy - floor`, and a report that omits the floor cannot be compared with anyone else's.

Offline diagnostic tooling, like `data/verify_benchmark.py`: it imports `src/` and nothing in
`src/` imports it, and no benchmark run depends on it having been executed. Costs LLM tokens
(one call per question, two when the first answer carries no option) and zero provider quota.

    PYTHONPATH=data python data/measure_no_tool_floor.py dataset/seoul_kmapeval_v2_mcq_100.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.base import format_question  # noqa: E402
from src.config import Settings  # noqa: E402
from src.llm import (  # noqa: E402
    LLMContextOverflowError,
    LLMOutputTruncatedError,
    LLMUnavailableError,
    OpenAIChatClient,
)
from src.parsing import parse_answer  # noqa: E402

# Not `REACT_SYSTEM_PROMPT`: that one tells the agent to use the map tools, and with none bound 22
# of 100 runs answered by writing tool calls in prose and never selected an option -- a harness
# artifact scored as a wrong answer. Saying outright that there are no tools moved the floor from
# 18/100 with 22 unparsed to 31/100 with none.
CLOSED_BOOK_PROMPT = (
    "You are answering a Korean multiple-choice question about places in Seoul. "
    "No tools are available: answer from your own knowledge. "
    "Select one 0-based option and answer exactly as ^^Option_Number^^."
)


def answer_one(row: dict) -> dict:
    llm = OpenAIChatClient(Settings())
    messages = [
        {"role": "system", "content": CLOSED_BOOK_PROMPT},
        {"role": "user", "content": format_question(row["question"], row["options"])},
    ]
    try:
        text = llm.chat(messages).content
        if parse_answer(text, option_count=len(row["options"])) is None:
            # ReAct is nudged once when its step budget runs out; the floor gets the same nudge.
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {"role": "user", "content": "Select the best option now, exactly as ^^N^^."}
            )
            text = llm.chat(messages).content
        failure = None
    # The three ways a call ends without an answer, all of them the endpoint's and none of them
    # this question's fault. They are the same three the evaluator separates for an agent run, and
    # they are caught for the same reason: one closed-book question that sends the model into a
    # reasoning spiral used to raise out of the thread pool and take the whole floor with it --
    # 282 questions' worth of answers discarded because one of them wrote 65,304 tokens. A floor
    # is a measurement; it reports what it could not measure rather than refusing to report.
    except (LLMUnavailableError, LLMOutputTruncatedError, LLMContextOverflowError) as exc:
        text, failure = "", f"{type(exc).__name__}: {exc}"
    finally:
        llm.close()
    predicted = parse_answer(text, option_count=len(row["options"]))
    return {
        "id": row["id"],
        "template_id": row.get("template_id"),
        "classification": row.get("classification"),
        "gold": row["answer"],
        "predicted": predicted,
        "correct": predicted == row["answer"],
        "failure": failure,
    }


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset/seoul_kmapeval_v2_mcq_100.jsonl"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(answer_one, rows))
    correct = sum(row["correct"] for row in results)
    print(f"\n=== no-tool floor on {dataset}")
    failed = [row for row in results if row["failure"]]
    print(
        f"overall {correct}/{len(results)}   "
        f"unparsed {sum(row['predicted'] is None for row in results)}   "
        f"failures {len(failed)}"
    )
    # A floor with rows the endpoint never answered is a floor over fewer rows than it claims, so
    # say which and how many rather than folding them into the denominator silently.
    for row in failed:
        print(f"  ! {row['id']} {row['template_id']}: {row['failure'][:120]}")
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for row in results:
        key = row["template_id"] or row["classification"] or "?"
        totals[key] += 1
        hits[key] += row["correct"]
    for key in sorted(totals):
        print(f"  {key:28s} {hits[key]:2d}/{totals[key]:2d}")
    # A floor at chance with a flat histogram is a model guessing; a skewed one is a position prior.
    chosen = Counter(row["predicted"] for row in results)
    print("  chosen-option histogram:", dict(sorted(chosen.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    main()
