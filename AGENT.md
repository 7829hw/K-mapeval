# AGENT.md

Canonical working guidance for coding agents on the `upstream-kakao` branch. `CLAUDE.md`
imports this file; edit here, not there.

## What this branch is

A comparison of a **MapEval-style ReAct** agent against **Spatial-Agent (GeoFlow)** on Korean
multiple-choice map questions, where **both agents are the upstream implementations, vendored
unmodified, and the only thing swapped is the map API**.

That is the whole premise, and it is the one thing to protect:

- `src/spatial_agent/` is [`ecerybao/Spatial-Agent`](https://github.com/ecerybao/Spatial-Agent)
  @ `6876bba`, byte-identical apart from a mechanical rename.
- `src/mapeval_api/` is [`MapEval/MapEval-API`](https://github.com/MapEval/MapEval-API)
  @ `35d481a`; `Evaluator2.py` is vendored untouched as the reference, `FormattedTools.py` is
  the port.
- `src/kakao_maps.py` puts Kakao Local + Kakao Mobility behind Google Maps' client surface.

An accuracy this branch reports is attributable to the upstream architectures, not to a
reimplementation of them. **`main` is the other experiment**: a from-scratch port with its own
tool registry, operator library and grounding stage. The two branches' numbers answer different
questions and must never be pooled.

`docs/UPSTREAM_MAPPING.md` records **every** deviation from the two upstreams. Read it before
changing anything under `src/spatial_agent/`, `src/mapeval_api/` or `src/kakao_maps.py`, and add
to it when you make another one. If a deviation is not in that file, it is a bug.

`K-MapEval_PRD.md` is the project spec. `docs/REFERENCE_MAPPING.md` documents the `main` branch
port and the benchmark measurements; it is the datasets' provenance, not this branch's design.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp example.env .env          # .env.example has the same content; either works

pytest                       # full suite; Kakao and the LLM are stubbed, no keys or network
ruff check .                 # line-length 100, rules E,F,I,UP,B

# The benchmark run. Costs real LLM tokens and real Kakao quota.
python main.py --agent both                   # dataset/seoul_kmapeval_v4_mcq_100.jsonl
python main.py --agent react
python main.py --agent both --ids seoul_kmapeval_v4_000 seoul_kmapeval_v4_024
python main.py --agent react --verbose-agent  # upstream's own chain-of-thought printing
python main.py --agent spatial --concurrency 4

# Verify the vendored trees have not drifted from upstream.
diff -r ~/spatial-agent/src/agent src/spatial_agent/agent   # only the renames
diff ~/mapeval-api/Evaluator2.py src/mapeval_api/Evaluator2.py
```

Every test in `tests/` fakes both Kakao (`httpx.MockTransport`) and the LLM
(`FakeListChatModel`), so verify with `pytest` first and only run the benchmark when the user
asks for live numbers. Keep it that way: any live-API test stays separate and optional.

## Layering

```
main.py → Evaluator → src/agent/react.py    → src/mapeval_api/FormattedTools.py ─┐
                    → src/agent/spatial.py  → src/spatial_agent/  ───────────────┤
                                                                                 └→ src/kakao_maps.py → Kakao
```

- **`src/agent/react.py` and `src/agent/spatial.py` are adapters, not agents.** They translate
  between the harness's `AgentResult` and what the vendored code returns. Logic that decides an
  answer does not belong in them; it belongs upstream, where it already is.
- **`src/kakao_maps.py` is the only place Kakao HTTP calls exist.** Both architectures read the
  same client instance per worker, so an accuracy difference between them can never be a
  difference in what they were shown.
- **`data/_toolkit/` is offline dataset tooling and nothing in `src/` may import it.** It is
  `main`'s tool layer, kept so the dataset builders and verifiers still run. It is a second
  Kakao implementation, which is only acceptable because it never touches an agent.

## The rules that break silently if violated

- **Do not edit `src/spatial_agent/` or `src/mapeval_api/Evaluator2.py`.** Their value is that
  they are upstream's. A bug you find in them is a finding about upstream, and belongs in the
  write-up rather than in a patch — including the dead branch at `operators.py:1110`, which
  reads `client.geocode(wp)` in a shape `geocode` never returned on either provider. If a change
  is genuinely forced by the API swap, it goes in `docs/UPSTREAM_MAPPING.md` first.
- **`src/kakao_maps.py` keeps Google's shape, method for method.** ~9,900 vendored lines index
  into what it returns. `tests/test_kakao_maps.py` pins the contract — the matrix shape
  `operators.tsp_tw` walks, the `html_instructions` the formatters print, the four keys every
  geocode caller reads. A method that changes shape breaks upstream code a long way from here.
- **0-based option indices everywhere in the harness** — dataset `answer`,
  `AgentResult.predicted_answer`, logs, reports. Upstream MapEval is 1-based and upstream
  Spatial-Agent is 0-based, so `react.py` converts once on the way out and `spatial.py` converts
  nothing. Both are pinned by tests. The final-answer wire format is `^^N^^`.
- **The ReAct baseline's prompt is upstream's, character for character.** `build_prompt`
  reproduces `Evaluator2.py` lines 58-76 including its one-based `Option1: …` list. Rewriting it
  makes the baseline a function of this repository rather than of MapEval, and any gap it then
  shows is unattributable. Same for `parse_upstream_answer` and the five-tool set: adding a
  sixth is a claim that `Evaluator2.py` line 33 constructs it.
- **Gold and eval-only metadata never reach an agent.** `BenchmarkItem.agent_input()` returns
  `(question, options)`; `answer`, `classification`, `region`, `difficulty` and `verified_at`
  stay in the evaluator. `process_question` accepts a `correct_answer` for its own logging — the
  adapter does not pass it, and `tests/test_spatial_adapter.py` pins that.
- **No per-question special-casing.** Heuristics must be generic over a place type or a
  question shape, never keyed to a question id or an option string, and answers are never
  hardcoded. This applies to `TYPE_VOCABULARY` too: add terms from Kakao category paths you have
  actually observed, never from what a category ought to be called.
- **A filter over a field the evidence does not carry is not a filter.** Kakao publishes no
  rating, price level or opening hours, so `min_rating` and `open_now` are accepted and ignored.
  Applying them would delete every candidate, and an empty list is what a generation stage
  guesses over. A source that publishes no ratings is not a source in which every place is
  unrated.
- **A place is not near itself.** `NearbyPlacesTool` drops the anchor from its own neighbour
  list, and `get_distance_matrix` answers the diagonal locally. The anchor stands at 0 m from
  itself and wins every distance ranking it appears in; Kakao additionally refuses to route a
  leg under 5 m. The diagonal is the *only* leg that may be filled — a missing off-diagonal leg
  is `ZERO_RESULTS`, never a free hop, because an absent leg is missing evidence.
- **A ranking never invents evidence.** `geocode` keeps upstream's 100 km / 200 km rejection
  thresholds: the nearest candidate is always *some* candidate, and a name Kakao does not carry
  must fail rather than resolve to whatever scored least badly.
- **The region prior is deployment configuration, not evidence.** `KAKAO_SEARCH_CENTER` /
  `KAKAO_SEARCH_RADIUS_M` bias the first keyword query toward the benchmark's region, because
  Korean POI names repeat across cities. A name with no match in the region still resolves
  nationwide, so the prior can never hide a place. It applies to whichever agent queries, reads
  nothing from `BenchmarkItem`, and a report must say which it was.
- **Provider failures and reasoning failures are distinct `failure_type`s.** `KakaoAuthError`,
  `KakaoRateLimitError` and `KakaoTimeoutError` are separate classes because
  `is_transient_failure` decides retryability from the class name a message starts with. Only a
  timeout or a rate limit is worth asking again for; a rejected key and a place Kakao does not
  carry are not.
- **Do not add code that judges the LLM endpoint's health and changes control flow on the
  verdict** — no preflight ping, no circuit breaker, no run-level "invalid" stamp. The endpoint
  is a self-hosted deployment behind a reverse proxy: it answers 502/503 while it reloads and
  reports 404 for a model it serves again a minute later. Every one of those turns a slow
  endpoint into lost questions and none makes an answer arrive sooner. Wait instead.
- **Never retry a wrong answer.** `Evaluator._run_single` retries only what
  `is_transient_failure` accepts. An `agent_reasoning_failure` or an `answer_parse_failure` is
  the result the architecture earned, and re-rolling it measures luck. Retries are counted, not
  hidden: each row carries `attempts`.
- **`MAX_REASONING_STEPS` is 30 on purpose**, against `initialize_agent`'s default of 15. On
  five primitives a four-stop itinerary needs four PlaceSearch turns plus four TravelTime turns
  before any arithmetic, and being generous to the baseline is the conservative direction for
  the claim under test. Do not cut it to make a gap appear.
- **Counters are deltas around each question**, read off the worker's own `KakaoMapsClient`.
  Each worker owns one client and one agent (`Evaluator` rejects a shared `agent` when
  `max_workers > 1`); never share an agent across workers or introduce module-level mutable
  agent state. `FormattedTools` holds its client in a `threading.local` for exactly this reason.
- Do not create persistent dumps of raw Kakao responses beyond the cache; usage rights are not
  established for them.

## Adding to the type vocabulary

`TYPE_VOCABULARY` in `src/kakao_maps.py` maps a requested place type onto Kakao's own. It has
to speak three vocabularies, because three kinds of caller reach it:

- a **Google place type** (`convenience_store`), because the vendored prompts were written
  against Google and a planner still emits one;
- the **Korean noun** a question asks by (`편의점`), so a planner that copies the question's own
  word reaches the same retrieval;
- a **bare Kakao code** (`CS2`), because a planner that has seen one will write it.

Keep the terms to words Kakao uses for a *type*. A bare `마트` also matches
`가정,생활 > 편의점 > 이마트24`, which lets a convenience-store brand answer a 대형마트 question.
Add terms from category paths you have actually observed, never from what a category ought to be
called.

## Concurrency, cache, and outputs

`Evaluator` runs `BENCHMARK_CONCURRENCY` (default 4) worker threads, each entering its own
`create_agent_session` in `main.py` with a private `KakaoMapsClient` and agent. Result order is
restored by index, not completion. `src/logging.py` builds a fresh `logging.Logger` per question
so concurrent traces cannot cross-write.

The SQLite cache underneath is shared and safe to share. It stores Kakao responses plus the
place table `get_place_details` is served from — Kakao has no details endpoint, so losing that
table would break the baseline's whole PlaceSearch-then-PlaceDetails idiom. Its tables are named
`kakao_responses` / `kakao_places` so a `data/kakao_cache.db` written by `main`'s implementation
cannot be read as this one's. `KAKAO_CACHE_DB_PATH=` (blank) disables it.

`logs/`, `reports/` and `data/*.db` are generated and gitignored: per-question traces at
`logs/<UTC>_id<id>_<slug>.log`, one `reports/test_<UTC>.json` per batch with `metadata` /
`statistics` / `results`. Report `metadata` carries `agent_type`, `llm_model`, `llm_base_url`,
`code_revision` and the two `upstream_*` revisions, so a report stays attributable after the
vendored trees are updated. Primary metric is overall MCQ accuracy; per-classification accuracy,
tool calls, API calls, cache hits/misses, latency and failures are reported alongside it.

## Datasets

JSONL, one `BenchmarkItem` per line, unique ids, 2–5 options, `answer` a 0-based index, and
`classification` from `nearby | poi | routing | trip | type | direction | distance | radius`.
Extra fields are allowed and every one of them is evaluation-only.

The benchmarks were built and verified on `main`; this branch consumes them as data.
`dataset/seoul_kmapeval_v4_mcq_100.jsonl` is the default — the MapEval-method reproduction
benchmark, built by evidence-first construction the way MapQaTor does. `docs/REFERENCE_MAPPING.md`
carries the full construction rules and measurements for all four, including why v2 is
superseded and why v3 is the compositional one.

`data/build_*.py` and `data/verify_*.py` still work, through `data/_toolkit/`. Note that the
toolkit is `main`'s Kakao path and normalizes differently from `src/kakao_maps.py`, so **a
verifier disagreement is not automatically an agent result** — re-verify against `main` before
concluding one.

## What this branch cannot measure

Say so in any write-up rather than reporting a low number as an architecture result:

- **Ratings, price levels, opening hours.** Kakao Local publishes none, so MapEval's whole
  attribute half is unanswerable here by nature.
- **Walking, cycling and transit routes.** Kakao Mobility routes cars only; those modes are
  refused in words rather than answered with a driving route.
- **Anything outside Korea.**
- **An LLM outage on the Spatial-Agent side.** `process_question` swallows every exception, so
  `llm_unavailable_count` under-reports there. The ReAct side has no such gap.

Because this repository runs the prompting-only path — no SFT, no DPO, no embedding retrieval —
reports must be labeled prompting-only and must not be presented as reproducing either paper's
headline numbers.
