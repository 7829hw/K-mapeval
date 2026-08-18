# AGENT.md

Canonical working guidance for coding agents in this repository. `CLAUDE.md` imports this file;
edit here, not there.

## What this is

A research MVP that compares a MapEval-style **ReAct** agent against a **Spatial-Agent** (GeoFlow)
port on Korean multiple-choice map questions, both driven by the same Kakao-backed tool layer. The
research question is whether Spatial-Agent's reported gains reproduce on Korean geography and POI
data. The independent variable is agent architecture — everything below the agent (provider, tools,
cache, normalized schemas, evaluator) must stay identical for both.

`K-MapEval_PRD.md` is the full spec. `docs/REFERENCE_MAPPING.md` records every deliberate deviation
from the upstream MapEval / Spatial-Agent implementations — update it when you add another one.

Upstream references, when porting or checking behavior:

- `MapEval/MapEval-API` — the ReAct baseline. `Evaluator2.py` for the agent loop and MCQ
  evaluation, `FormattedTools.py` / `Tools.py` for tool interfaces and formatting.
- `ecerybao/Spatial-Agent` — the Spatial-Agent side. `src/agent/spatial_agent.py`,
  `src/agent/operators.py`, `src/tools/google_maps.py`, `test_agent.py`.
- `MapQaTor/mapqator-backend` — architectural reference only; do not reproduce its HTTP backend.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp example.env .env          # .env.example has the same content; either works

pytest                       # full suite; Kakao + LLM are stubbed, no keys/network needed
pytest tests/test_tools_and_agents.py::test_react_executes_common_tool_then_parses_answer
pytest -k geoflow
ruff check .                 # line-length 100, rules E,F,I,UP,B

python main.py --agent react
python main.py --agent spatial
python main.py --agent both --dataset dataset/seoul_mapqa_kr_mcq_100.jsonl
python main.py --agent both --dataset dataset/test.jsonl --ids nearby_001 poi_001
python main.py --agent spatial --concurrency 4
```

Running `main.py` costs real LLM tokens and Kakao API quota. Every test in `tests/` fakes both, so
verify with `pytest` first and only run the benchmark when the user asks for live numbers. Keep unit
tests mocked (`httpx.MockTransport` for Kakao, a queued fake for the LLM); any live-API test stays
separate and optional.

## Layering (do not shortcut it)

```
main.py → Evaluator → ReactAgent | SpatialAgent → ToolRegistry → MapProvider (KakaoMapProvider)
                                                              → SQLiteMapCache → Kakao HTTP APIs
Spatial-Agent additionally → SpatialOperatorRegistry (pure local computation, zero API calls)
```

- Kakao HTTP calls exist **only** in `src/tools/kakao.py` — Kakao Local for place/category/address/
  coordinate queries, Kakao Mobility for driving directions. Agents, evaluator, dataset, and tests
  talk to the abstract `MapProvider` in `src/tools/map.py`, and providers are injected, never
  constructed inside an agent.
- Raw Kakao JSON never reaches an agent. Everything is normalized to the frozen `Place` / `Route`
  models in `src/models.py`; adding a field there means adding it to the provider normalizers and
  the cache payload.
- Deterministic math (haversine, bearing, sorting, min/max, route comparison, TSP-TW, time
  arithmetic) belongs in `src/tools/spatial.py` and must never spend an API call.
- ReAct and Spatial-Agent share one `ToolRegistry`; never give one agent evidence, a tool, or a
  Kakao implementation the other cannot reach. Tool wrappers stay thin and delegate to the provider.
- **Place disambiguation is anchor-relative, and both agents get it.** Korean POI names repeat
  across branches and cities, so `_best_place_match` (`src/tools/registry.py`) scores proximity to a
  known anchor *below* the name-evidence terms (exact / branch / category / containment) and *above*
  string similarity — without that term a bare brand name resolves to whichever branch has the
  shortest name, anywhere in the country. `_batch_geocode` then reconciles the batch against itself:
  when the anchor ends up farther than `radius_m` from every other resolved name, it re-searches the
  anchor in the peers' neighbourhood, because a nationwide keyword search for an ambiguous short
  name never surfaces the intended place as a candidate at all. An argument in
  `PLACE_ARGUMENT_NAMES` that arrives as `None` raises `PlaceNotFoundError` before pydantic can
  report it as a validation error.
- **The region prior is deployment configuration, not evidence.** `KAKAO_SEARCH_CENTER` /
  `KAKAO_SEARCH_RADIUS_M` bias the first Kakao keyword query toward the benchmark's region inside
  `KakaoMapProvider`, because Kakao searches nationwide and Korean POI names repeat across cities. A
  name with no match in the region still falls back to the unbiased nationwide search, so the prior
  can never hide a place; the cache key carries it so biased and unbiased runs cannot share entries.
  It applies to both agents identically and reads nothing from `BenchmarkItem` — deriving it from
  `region` would leak eval-only metadata. Blank disables it, and reports must say which it was.
- **Reconciliation is for pairs only, and the anchor is authoritative.** `_reconcile_batch` runs
  when exactly two names resolved. With three or more the batch is an anchor plus option texts, the
  anchored search has already placed each option, and "tightest span" lets scattered option brands
  out-vote a correct anchor and drag the batch to another district. A wrongly distant option is
  harmless — `nearest` never picks it — but a moved anchor invalidates every operator after it.
- **Option recovery answers the question's category, and never returns the anchor.**
  `recover_option_places` takes the `category_code` the retrieval used (bound by
  `_ground_graph_literals` from `_nearby_retrieval_specs`) and skips the nationwide fallback when
  it is set, so an option is satisfied by the kind of place being asked for rather than any
  namesake — `목동` in a station question otherwise matched 교보문고 목동점, the anchor itself.
- **A ranking never invents evidence.** `max` always yields a candidate, so `_best_place_match`
  applies `NAME_EVIDENCE_FLOOR` and returns `None` when the winner shares no containment and too
  little similarity with the query — a name Kakao does not have must fail as `PlaceNotFoundError`,
  not resolve to whatever scored least badly (`마천1치안센터` → `웅동파출소`, 100 km away, zero
  characters in common). The floor applies only where the name *is* the evidence: candidates from a
  nationwide keyword search must clear it, while candidates from an anchored neighbourhood search
  are already spatially constrained, so a brand written the question's way (`S-OIL` for 에쓰오일,
  `OLYMPUS` for 올림푸스한국) still resolves — `require_name_evidence` carries that distinction, and
  enforcing the floor on both paths empties `filter_by_direction` and collapses direction accuracy.
  `match_distance_options` reports the same idea as `fits` / `error_ratio`:
  the nearest option is always *some* option, and a kilometre-scale error means the places were
  resolved wrong, not that the answer is the least-bad number. Do not restore a "closest wins"
  fallback in either place.
- No separate HTTP backend server, web UI, or extra datastore beyond the SQLite cache. Keep
  `src/agent/` and `src/tools/` as the only source subpackages and `main.py` as the only entry
  point.

## Invariants that break silently if violated

- **0-based option indices everywhere** — dataset `answer`, agent `predicted_answer`, logs,
  reports. The final-answer wire format is `^^N^^` (`src/parsing.py::parse_answer`).
- **Gold answer and eval-only metadata never reach an agent.** `BenchmarkItem.agent_input()`
  returns only `(question, options)`; `answer`, `classification`, `region`, `difficulty`, and
  `verified_at` stay in the evaluator.
- **No per-question special-casing.** Heuristics must be generic over a classification, never
  keyed to a specific question id or option string, and answers are never hardcoded.
- Provider failures and agent-reasoning failures are distinct `failure_type`s; `find_provider_failure`
  in `src/agent/base.py` recognizes provider errors by their serialized `ProviderError` class-name
  prefix, so keep those names when adding exception types. Place-not-found, timeout, auth failure,
  rate limit, route-not-found, and unsupported travel mode must each fail as their own explicit
  `ProviderError` subclass rather than degrade silently.
- Counters (`api_calls`, `cache_hits`, `cache_misses`, `tool_calls`) are recorded as deltas around
  each question, read off the shared provider/registry — new tool paths must go through the
  registry or their calls vanish from the metrics.
- Per-question logs carry `question_id`, `classification`, `agent_type`, `predicted_answer`,
  correctness, and every tool name with normalized arguments, status, and API-call counts.
- Do not create persistent dumps of raw Kakao responses; usage rights are not established for them.

## Spatial-Agent / GeoFlow

Pipeline in `src/agent/spatial.py::SpatialAgent.answer`, preserving the upstream
`Route → Plan → Execute → Evaluate → Generate` shape:
`analyze → retrieve_templates → compose (LLM graph) → factorize + validate (→ one LLM repair →
pre-validated template fallback) → topological execute → evaluate → generate`.

The formalism lives in `src/agent/geoflow.py`: `OPERATOR_CONTRACTS` (output core-concept type per
operator), `OPERATOR_INPUT_TYPES`, `TEMPLATES` (Appendix E macro families, all executable and
G1–G5-valid), `factorize_geoflow`, and `normalize_and_validate_graph` (G1 acyclicity, G2 role
ordering over `sub_condition < condition < support < measure` with contextual roles excluded,
G3 type compatibility, G4 executability, G5 both-direction reachability
`EXTENT/TEXTENT → node → MEASURE`).

Reuse upstream operator semantics wherever they are still valid; change the planner, state, or an
operator only where Kakao compatibility forces it.

**Adding or renaming an operator touches four places, and tests enforce the consistency:**

1. an implementation — `ToolRegistry` (`src/tools/registry.py`, external/API-backed) *or*
   `SpatialOperatorRegistry` (`src/tools/spatial.py`, local/deterministic);
2. `OPERATOR_CONTRACTS` + `OPERATOR_INPUT_TYPES` in `src/agent/geoflow.py`;
3. the operator-contract list inside `GRAPH_PROMPT` in `src/agent/spatial.py` — the planner LLM
   only knows what this prompt spells out;
4. `_normalize_arguments` in `src/tools/spatial.py` if planners emit argument aliases for it.

`SpatialAgent.__init__` raises at construction time if any `OPERATOR_CONTRACTS` entry is not
executable, so a missing step 1 or 2 fails fast; a missing step 3 only shows up as degraded
benchmark accuracy.

Executed steps are recorded twice: `results` keyed by step id (operator state) and `concept_state`
materialized through each step's `output_bindings` (concept state). Both go into the trace the
generation stage conditions on. Step failures are isolated into `{"error": ...}` rather than
aborting the run — keep that behavior.

**Execution is lenient about shape, strict about evidence.** Planners routinely reference the object
that *contains* a place instead of the place. Mirroring upstream's concept-reference resolution:

- `_resolve_references` / `_descend_reference` degrade an over-specified `$node.path` to the closest
  resolvable object (and only raise for an unknown node id); `_resolve_output_binding` does the same
  so concept-state materialization can never abort a run.
- Every coordinate operator normalizes its inputs through `_as_place` / `_as_place_list` in
  `src/tools/spatial.py`, which unwrap `{query, place, candidates}`, `{"location": …}`, `nearest`
  results, single-element branches, and `lat`/`lng`/`x`/`y` spellings. A genuinely unresolved place
  raises `PlaceNotFoundError`; never let it surface as a `TypeError` or `KeyError`.
- Do not tighten these back into hard failures. A shape mismatch is a planner artifact; only missing
  evidence is a real failure.

Question literals are bound after drafting and before validation in `_ground_graph_literals`: the
anchor name, the requested direction, the exact radius, the candidate option texts for all three
`match_*` operators, the two compared POI names of a `distance` question
(`_extract_compared_places`, which the template path shares), and the retrieval spec. `_nearby_retrieval_specs` fans a place type out over
every Kakao keyword/category that covers it, and `_retrieval_steps` merges the branches back under
the planner's original node id so downstream references stay valid. The pre-validated template path
already emits one node per spec, so it grounds with `expand_retrieval=False`.

The generation stage asks for `predicted_answer` *and* `predicted_option`; `_select_option`
reconciles them text-first (exact candidate text → declared index → single containment match) and
records which path fired in the trace as `selection_method`.

## Concurrency, cache, and outputs

`Evaluator` runs `BENCHMARK_CONCURRENCY` (default 4) worker threads, each entering its own
`create_agent_session` context in `main.py` with a private `OpenAIChatClient`, `KakaoMapProvider`,
and agent. Never share an agent across workers (`Evaluator` rejects a shared `agent` when
`max_workers > 1`), and never introduce module-level mutable agent state. Result order is restored
by index, not completion. `src/logging.py` builds a fresh `logging.Logger` per question for the
same reason — a shared logger with temporary handlers would cross-write concurrent traces.

The LLM endpoint is treated as a fallible dependency, not a given. `OpenAIChatClient`
(`src/llm.py`) drives its own retries (the SDK's are disabled) with exponential backoff and jitter
on connection/timeout errors and `RETRYABLE_STATUS_CODES`; `CONFIGURATION_STATUS_CODES`
(401/403/404) fail immediately, and 4xx we caused (400/413/422) propagate unchanged because they are
the agent's prompt, not the endpoint. Exhausted retries and configuration errors raise
`LLMUnavailableError`, which both agents record as `failure_type="llm_unavailable"` — never as
`agent_reasoning_failure`, since an outage says nothing about an architecture. `Evaluator` counts
those, trips a circuit breaker after `BENCHMARK_ABORT_AFTER_LLM_FAILURES` consecutive ones
(remaining questions become `run_aborted`), and stamps `statistics.run_validity.valid = false` so a
0% report can never be read as a benchmark result. `main.py` preflights the endpoint with one cheap
call before each agent leg, so `--agent both` cannot score its second agent 0% on an endpoint that
died between legs. Report `metadata` carries `agent_type`, `llm_model`, and `llm_base_url` so a
report is attributable after the fact.

`SQLiteMapCache` (`src/tools/cache.py`) is keyed by operation + canonicalized arguments, stores
normalized `Place`/`Route` payloads only — never raw responses or keys — with a TTL (`0` = never
expire) and a `SCHEMA_VERSION` that must be bumped when the stored payload shape changes.

`logs/`, `reports/`, and `data/*.db` are generated and gitignored: per-question traces at
`logs/<UTC>_id<id>_<slug>.log`, one `reports/test_<UTC>.json` per batch with `metadata` /
`statistics` / `results`. Primary metric is overall MCQ accuracy; per-classification accuracy, tool
calls, API calls, cache hits/misses, latency, and failures are reported alongside it.

## Datasets

JSONL, one `BenchmarkItem` per line, unique ids, 2–4 options, `answer` a 0-based index, and
`classification` from `nearby | poi | routing | trip | type | direction | distance | radius`
(the same eight values are `SUPPORTED_INTENTS` in `src/agent/spatial.py` — extending the set means
touching both, plus the intent heuristics and evaluation rules). Extra fields are allowed
(`verification_status`, etc.). `dataset/sample.jsonl` is an unverified development fixture; treat
answers as unvalidated until someone confirms them against live Kakao data and sets `verified_at`.
Building the full Korean research benchmark is a separate task, not something to expand into
incidentally.

Because this repo runs the prompting-only path (no SFT/DPO, no embedding retrieval, Kakao instead of
the paper's Google/MapEval-Textual snapshot), reports must be labeled prompting-only and must not be
presented as reproducing the paper's headline numbers.
