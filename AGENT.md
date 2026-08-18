# AGENT.md

Canonical working guidance for coding agents in this repository. `CLAUDE.md` imports this file;
edit here, not there.

## What this is

A research MVP that compares a MapEval-style **ReAct** agent against a **Spatial-Agent** (GeoFlow)
port on Korean multiple-choice map questions, both driven by the same tool layer. The research
question is whether Spatial-Agent's reported gains reproduce on Korean geography and POI data. The
independent variable is agent architecture — everything below the agent (provider, tools, cache,
normalized schemas, evaluator) must stay identical for both.

The tool layer has two interchangeable evidence sources, chosen per run and never mixed:

- **context** (`ContextMapProvider`) — the default. One corpus built from every context the dataset
  carries, in MapEval's context format, serving the tools instead of any API. This is the port of
  upstream Spatial-Agent's local context cache.
- **hybrid** — that corpus with `KakaoMapProvider` behind it for what it does not hold. This is
  upstream's own arrangement (cache first, Google Maps on a miss) with Kakao in Google's place.
- **kakao** (`KakaoMapProvider`) — live Kakao Local / Kakao Mobility alone, with the SQLite cache
  and the region prior. Needed for a dataset whose rows carry no context.

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

python main.py --agent react                  # dataset/seoul_mapeval_v1_mcq_100.jsonl, context evidence
python main.py --agent spatial
python main.py --agent both --ids seoul_mapqa_v0_000907 seoul_mapqa_v0_000009
python main.py --agent both --provider hybrid  # corpus first, Kakao for what it lacks
python main.py --agent both --provider kakao   # live Kakao alone
python main.py --agent spatial --concurrency 4
```

Running `main.py` costs real LLM tokens, and `--provider kakao` costs Kakao API quota on top. Every
test in `tests/` fakes both, so verify with `pytest` first and only run the benchmark when the user
asks for live numbers. Keep unit tests mocked (`httpx.MockTransport` for Kakao, a queued fake for
the LLM); any live-API test stays separate and optional.

## Layering (do not shortcut it)

```
main.py → Evaluator → ReactAgent | SpatialAgent → ToolRegistry → MapProvider
                                                    ├→ ContextMapProvider → the row's own context
                                                    └→ KakaoMapProvider → SQLiteMapCache → Kakao
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
- **Option recovery answers the question's category, stays inside its radius, and never returns
  the anchor.** `recover_option_places` takes the `category_code` the retrieval used (bound by
  `_ground_graph_literals` from `_nearby_retrieval_specs`) and skips the nationwide fallback when
  it is set, so an option is satisfied by the kind of place being asked for rather than any
  namesake — `목동` in a station question otherwise matched 교보문고 목동점, the anchor itself. The
  uncategorised fallback that remains is a nationwide search, so `_within_anchor_radius` keeps only
  what is actually within the radius asked about: `꽃담공방` came back from 순천, 129 km away, and
  entered the candidate set as if it were a neighbour.
- **A name matches one place, and a place answers one option.** `_assign_unique_matches`
  (`src/tools/spatial.py`) pairs option texts with retrieved POIs so neither side is used twice —
  scoring each option independently let one 서울공릉초등학교 clear the floor for both
  서울오륜초등학교 and 서울평화초등학교, and the tie-break then handed the answer to whichever came
  first in the list. `distinguishing_similarity` supplies the other half: Korean POI names of one
  kind share long generic affixes, so similarity is capped by how well the *residue* between the
  shared prefix and suffix matches (오륜 vs 공릉, not 서울…초등학교). A residue too short to
  distinguish anything (`CU 가락센트럴점` against Kakao's `CU 가락센타점`) is a spelling variant and
  is left alone. `_names_the_same_place` applies the same test, which is what stops the
  brand-only retry in `_query_variants` — `CU` for `CU 구로소담점` — from resolving to whichever
  branch of the brand sits nearest the region prior's centre.
- **A reference the provider handed out is a reference it must take back.**
  `parse_coordinate_literal` (`src/tools/spatial.py`, shared by every provider) lets a `"lat,lng"`
  string stand where a place is expected, and `ContextMapProvider._dereference` accepts a
  `place_id` it minted as well. An agent that already holds a POI's coordinates or id is asking
  what is near *them*; sending either through the keyword search raised `PlaceNotFoundError`, and
  a ReAct run then burned its remaining steps re-searching a name that was never a name. Adding a
  provider means implementing both.
- **A ranking never invents evidence.** `max` always yields a candidate, so `_best_place_match`
  applies `NAME_EVIDENCE_FLOOR` and returns `None` when the winner shares no containment and too
  little similarity with the query — a name Kakao does not have must fail as `PlaceNotFoundError`,
  not resolve to whatever scored least badly (`마천1치안센터` → `웅동파출소`, 100 km away, zero
  characters in common). Every candidate must clear it, wherever it came from.
  `match_distance_options` reports the same idea as `fits` / `error_ratio`:
  the nearest option is always *some* option, and a kilometre-scale error means the places were
  resolved wrong, not that the answer is the least-bad number. Do not restore a "closest wins"
  fallback in either place.
- **Proximity is not identity, and containment is not always evidence.** Kakao's keyword search
  is tolerant, so asking an anchor's neighbourhood for a name Kakao does not carry answers with
  places of the same *kind*: `신사정육점` came back as `한아름축산`, `쌍문1치안센터` as
  `수유6치안센터`. Both resolved, both were a different POI, and every operator downstream then
  computed correctly over the wrong place. So the anchored path in `_resolve_batch` requires name
  evidence too; being near the anchor buys exactly one licence, `allow_cross_script`, for a brand
  whose Kakao entry is written in the other script (`A TWOSOME PLACE` / 투썸플레이스, `S-OIL` /
  에쓰오일), where characters cannot testify either way — and `strict_names` withdraws even that.
  When the neighbourhood holds nothing by that name, widen to the nationwide search before giving
  up; that recovers more places than the old exemption ever fabricated. `_containment_is_evidence`
  guards the other end: a short name is a substring of a great many long ones, so containment
  counts only when the shorter key leads the longer (올리브영 / 올리브영 거여역점) or makes up half
  of it — `압구정` inside `해피냠냠라면가게한강버스압구정선착장점` resolved a distance question to a
  POI 12 km from the one asked about.
- **A name is a name, not a name plus an address.** A dataset that has to separate two same-named
  options appends the address to the option text (`버거킹 - 서울특별시 용산구 한강로2가 한강대로
  92`). Kakao indexes names only, so `strip_location_qualifier` (`src/tools/spatial.py`) drops that
  tail inside `_search_key`, `_name_key`, and `_query_variants` — without it the tail drags every
  similarity below the floor and the option cannot resolve at all. `_search_key` folds 파출소 into
  치안센터 for the same reason: which of the two names an institution goes by is editorial, so the
  distinguishing part (연남) has to decide, not the institution word.
- **A place is not near itself.** `_excluding_self` (`src/tools/spatial.py`) drops the anchor from
  its own `nearest` ranking and its own `filter_by_direction` sector, by id or by standing on the
  same spot. The anchor is a place of the type being asked about often enough to appear among the
  candidates — a nearest-convenience-store question lists the store it starts from, and a stored
  retrieval heads its own block — and ranked by distance it wins with 0.0 m every time, which the
  generation stage then reports faithfully. It is kept only when it is the sole candidate, because
  an empty ranking answers nothing.
- **Leniency about shape is a property of the tools, not only the operators.**
  `_as_place_argument` / `_as_place_list_argument` (`src/tools/registry.py`) normalize every
  `Place`-typed tool argument the way `_as_place` normalizes an operator's: a one-element list is
  the geocode result the planner forgot to index into, a wrapper carries the place under `place` or
  `location`, an enriched place still is that place (`Place` forbids the `distance_m` /
  `candidate_index` keys an operator staples on), and an anchor written as a name is that place
  named. Without this the artifact `_as_place` shrugs off failed as a `ValidationError` before any
  tool ran, and the cascade emptied the rest of the graph.
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
- **Intent accuracy is scored over the questions an intent was predicted for.** ReAct has no
  classification stage, so every row carries `predicted_intent=None`; the old denominator reported
  0.0% for a classifier the architecture does not have. `statistics.intent_classification_accuracy`
  now carries `classified` alongside `total`, and `accuracy` is `None` when nothing was classified.
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

The LLM endpoint is treated as slow, not as dead. It is a self-hosted deployment behind a reverse
proxy: it answers 502/503 while it reloads, reports 404 for a model name it serves again a minute
later, and takes minutes to answer a ReAct call carrying a long trace. **Do not add code that judges
the endpoint's health and changes control flow on the verdict** — no preflight ping, no circuit
breaker, no run-level "invalid" stamp. Every one of those turns a slow endpoint into lost questions,
and none of them makes an answer arrive sooner. Wait instead.

Waiting happens at two scales. `OpenAIChatClient` (`src/llm.py`) drives its own retries (the SDK's
are disabled) with exponential backoff, jitter, and a `MAX_RETRY_DELAY_SECONDS` ceiling, for every
failure except `REQUEST_STATUS_CODES` (400/413/422) — those describe the request we sent, so
repeating it only repeats the mistake, and they propagate unchanged as the agent's problem.
`LLM_TIMEOUT_SECONDS` is deliberately generous for the same reason. When the attempts really do run
out, `LLMUnavailableError` is raised and both agents record `failure_type="llm_unavailable"` — never
`agent_reasoning_failure`, since an outage says nothing about an architecture.

`Evaluator._run_single` then retries the whole *question*: `BENCHMARK_QUESTION_RETRIES` extra
attempts with their own backoff and jitter (workers must not retry in lockstep), because an endpoint
can stay down for the entire minute one question takes. Only what `is_transient_failure` accepts
qualifies: `llm_unavailable`, plus a `provider_failure` whose message starts with
`ProviderTimeoutError` / `ProviderRateLimitError`. Never retry a wrong answer, an
`agent_reasoning_failure`, an `answer_parse_failure`, or a `PlaceNotFoundError` — those are the
result the architecture earned, and re-rolling them measures luck. Retries are counted, not hidden:
each row carries `attempts`, and `performance` carries `retried_question_count`,
`retry_recovered_ids`, and `llm_unavailable_count`. That last count is a fact for the write-up to
report next to the accuracy, not a verdict the code acts on. Report `metadata` carries `agent_type`,
`llm_model`, and `llm_base_url` so a report is attributable after the fact.

`SQLiteMapCache` (`src/tools/cache.py`) is keyed by operation + canonicalized arguments, stores
normalized `Place`/`Route` payloads only — never raw responses or keys — with a TTL (`0` = never
expire) and a `SCHEMA_VERSION` that must be bumped when the stored payload shape changes.

`logs/`, `reports/`, and `data/*.db` are generated and gitignored: per-question traces at
`logs/<UTC>_id<id>_<slug>.log`, one `reports/test_<UTC>.json` per batch with `metadata` /
`statistics` / `results`. Primary metric is overall MCQ accuracy; per-classification accuracy, tool
calls, API calls, cache hits/misses, latency, and failures are reported alongside it.

## The context provider

`src/tools/context.py` ports upstream Spatial-Agent's local context cache. `docs/REFERENCE_MAPPING.md`
records the file-by-file comparison against `ecerybao/Spatial-Agent@6876bba`; the invariants are:

- **The context reaches the provider, never the agent.** `BenchmarkItem.context` is provider
  evidence, not agent input — `agent_input()` still returns only `(question, options)`. Upstream is
  the same: `test_agent.py` evaluates on MapEval-API, which has no context field, and no agent
  module reads the text outside the cache. Handing it to the agent instead would delete the tool
  layer from the experiment and measure prompt reading, not agent architecture.
- **One corpus, shared by every question.** `main.py` collects `item.context` across the dataset
  and builds it once, the way `data/build_cache.py` builds one `context_cache.db` from the whole
  MapEval-Textual corpus. Do not scope it per question: an earlier revision did, and it made the
  mere existence of a name an answer signal — "which option exists at all" answered 14 of 100
  questions under per-question scoping against 9 under the corpus. A real map holds places that
  are not the answer, and so must this.
- **The corpus is the cache.** No lookup is an API call; a lookup it answers is a cache hit and one
  it cannot answer is a cache miss. With a `fallback` provider the miss goes there, which is
  upstream's Google Maps fallback; without one the miss is the answer and the caller raises the
  `PlaceNotFoundError` / `RouteNotFoundError` a missing POI deserves.
- **The corpus is a place database, not an answer sheet.** A MapEval context stores the *result*
  of the query its question asks: a nearby list already filtered by type and already sorted by
  distance. Replaying that block — which is what upstream's `get_nearby_places` does — hands the
  agent the answer for the price of one tool call, and the benchmark stops being distinguishable
  from MapEval-Textual, which is exactly what happened: ReAct scored 100/100. What a stored block
  legitimately contributes is its *places*; the ranking is computed in `nearby_search` from
  coordinates, over every place the corpus holds, including the ones belonging to other questions.
  Do not restore block replay.
- **A retrieval filters by type, in whichever vocabulary the caller speaks.** `TYPE_SYNONYMS` maps
  the context's own token, the Kakao category code a planner emits, and the Korean noun a question
  asks by onto one place type. It is generic over place types, never over questions — the same
  lexicon a geocoder keeps. A filter that matches nothing is not evidence of absence in a sparse
  corpus, so the unfiltered neighbourhood answers instead: the source tags a butcher, a stationer
  and an electronics dealer alike as `store`, and a question asking for one of those has to be
  answered from what is actually there. Never conflate two types to make a filter hit — lumping
  `store` with `supermarket` put 정육점 above the supermarkets in a supermarket question.
- **A place is not among its own neighbours,** in the provider as well as in `nearest`: the anchor
  stands at zero metres from itself and would head every ranking it appears in.
- **Containment is evidence in one direction only.** A brand may lead the branch that extends it
  (`CU` → `CU 삼청점`), because the registry's own query variants shorten names that way. The
  reverse must not match: a corpus entry for a bare `GS25` recorded whichever GS25 the retrieval
  found, not the `GS25 합정프리미엄점` an option names. Allowing it also lets a place-type question
  answer itself, since `편의점` sits inside `다모아편의점`. Below containment, `NAME_MATCH_FLOOR`
  and `distinguishing_similarity` apply exactly as on the Kakao path.
- **The category is served as the context wrote it** (`convenience_store`, `amenity=bank`), not
  translated into Korean. A cache serves what it stored; inventing a Korean label for a place-type
  question would be supplying part of the answer.

## Datasets

JSONL, one `BenchmarkItem` per line, unique ids, 2–4 options, `answer` a 0-based index, and
`classification` from `nearby | poi | routing | trip | type | direction | distance | radius`
(the same eight values are `SUPPORTED_INTENTS` in `src/agent/spatial.py` — extending the set means
touching both, plus the intent heuristics and evaluation rules). Extra fields are allowed
(`context`, `template_id`, `verification_status`, …). Building the full Korean research benchmark is
a separate task, not something to expand into incidentally.

`dataset/seoul_mapeval_v1_mcq_100.jsonl` is the benchmark: 100 rows sampled from
`dataset/seoul_mapeval_v1.json` (an OSM-derived Seoul pool, 1530 complete records — the file was
transferred truncated mid-record, so a reader must decode the complete objects and stop at the
break). The recipe, seed `20260818`:

- One quota per source template, so no family is spent on one question shape: `nearest_by_type` 20,
  `direction_by_type` 20, `distance_between` 20, `within_radius_by_type` 15, `type` 15,
  `routing_duration_value` 4, `routing_distance_value` 3, `routing_shortest_duration` 3.
- Distinct anchor place per row (100/100), and no repeated question text.
- `classification` comes from the source `template_id`, not its coarse `nearby|poi|routing` label —
  the eight-way vocabulary is what this benchmark reports by, and the template already encodes it.
- **Options are shuffled per row**, seeded by question id. The source generator emits distance
  options in ascending order and never shuffles, so its entire `distance_between` family carries
  gold at index 2; shuffling every row alike removes option position as evidence in every family
  without special-casing one.

All 100 gold answers are derivable from the shipped context by deterministic computation through
the provider — verify that before trusting a run's accuracy, since a wrong answer then means the
agent, not the evidence.

Because this repo runs the prompting-only path (no SFT/DPO, no embedding retrieval, and an
OSM-derived Korean context rather than the paper's MapEval-Textual snapshot), reports must be
labeled prompting-only and must not be presented as reproducing the paper's headline numbers. A
report's `metadata.provider` records which evidence source produced it, and runs from different
sources must never be pooled.
