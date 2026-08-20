# Reference implementation mapping

> **This document describes the `main` branch**, which is a from-scratch port of both
> architectures with its own tool registry, operator library and grounding stage. On the
> `upstream-kakao` branch the agents are upstream's own code with only the map API swapped, and
> its deviations are recorded in `UPSTREAM_MAPPING.md` instead. Paths named below
> (`src/agent/react.py`, `src/tools/registry.py`, `src/agent/geoflow.py`, …) refer to `main`.
>
> It is kept on this branch for one reason: it is the **provenance of the datasets**. The four
> benchmarks in `dataset/` were built and verified by the code documented here, and the
> construction rules, quotas, margins and no-tool floors recorded below are what a write-up has
> to cite about them. `data/_toolkit/` is that code, moved out of the agent path.

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, sleeps, and remote writes. Uses user-requested 0-based `^^N^^` answers. |
| MapEval tools/backend | `src/tools/registry.py`, `src/tools/map.py`, `src/tools/kakao.py` | Provider injection replaces the separate HTTP backend; tools expose normalized JSON. |
| Spatial information theory analysis | `normalize_analysis`, `ConceptGraph` | Preserves all seven core-concept labels and six roles. Missing EXTENT/MEASURE concepts are made explicit and marked synthetic. Dataset classification metadata is not passed in. |
| Concept transformation drafting | `retrieve_templates`, Spatial-Agent `compose` | All ten Appendix E examples are G1–G5-valid and executable. Retrieval remains deterministic intent/keyword scoring rather than embedding cosine similarity. |
| Concept graph `G` | `ConceptGraph`, `ConceptNode` | Stores IDs, types, roles, attributes, and explicit Analysis dependencies. Role adjacency never fabricates edges; executable G' edges come from actual operator references. |
| Operator-concept hypergraph `G'` | `factorize_geoflow`, `OperatorHyperedge` | Factorization is deterministic rather than learned. Each hyperedge records input concepts, supplementary literal factor parameters, and one or more output-path bindings. Radius, direction, category, and similar constants remain factors instead of being fabricated as operator outputs. Derived intermediate concepts are explicitly marked. |
| Five GeoFlow constraints | `normalize_and_validate_graph` | Enforces G1, G3, G4, and both halves of G5 on operator and concept graphs. G2 applies only to SUBCOND→COND→SUPPORT→MEASURE; contextual roles are excluded as required by Appendix B. |
| Core-concept execution types | `OPERATOR_CONTRACTS`, `SpatialOperatorRegistry` | LOCATION, OBJECT, FIELD, EVENT, NETWORK, AMOUNT, and PROPORTION all have executable producers. The current benchmark directly exercises only a subset. |
| Contextual/functional roles | `factorize_geoflow`, `ROLE_PRIORITY` | EXTENT/TEXTENT are scheduled as context but do not participate in procedural precedence. |
| Topological executor | Spatial-Agent `execute` stage | Records operator state and separately materializes concept state through output bindings, including multiple bindings from one operator. |
| Lenient concept-reference resolution (`_resolve_concept_reference`, `_extract_coordinates_from_concept`, `resolve_place_name`) | `_resolve_references` / `_descend_reference` in `src/agent/spatial.py`, `_as_place` / `_as_place_list` in `src/tools/spatial.py` | An over-specified `$node.path` degrades to the closest resolvable object instead of failing the operator, and every coordinate operator unwraps the place-shaped record it was handed. Only a genuinely unresolved place raises, as an explicit `PlaceNotFoundError`. |
| Executor evidence preprocessing (auto option distances) | `match_options`, `match_distance_options`, `match_type_options` | Candidate option texts are bound verbatim from the question at grounding time, so a planner cannot paraphrase or numerically re-type them before the Measure comparison. |
| Evaluator answer selection | `_select_option` in `src/agent/spatial.py` | Keeps upstream's text-first reconciliation (exact candidate text, then declared index, then a single containment match) on top of the repository's 0-based `^^N^^` contract. |
| Appendix C operators | `ToolRegistry`, `SpatialOperatorRegistry` | Adds reverse/batch details, waypoint directions, instruction route filtering, extractors, pairwise extremes, place filtering, travel-time nearest, temporal operators, and step analysis with paper-level semantics. |
| TSP-TW | `SpatialOperatorRegistry.tsp_tw` | Exhaustive optimization up to nine nodes with service times, windows, and budget; infeasible instances return the paper's nearest-unvisited partial feasible fallback. OR-Tools is not bundled. |
| Temporal operators | `timezone`, `open_at_time`, `calculate_finish_time`, `calculate_start_time` | Handles cross-midnight/24-hour periods. Multi-stop finish time queries cached/live route durations and adds stays. Latest-departure calculation is an explicit template helper. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` | The LLM conditions on question, final state, and full trace. Deterministic match evidence no longer overwrites the generated selection. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/reverse-geocode/nearby; Kakao Mobility handles driving and verified multi-waypoint routes with optional guides. |
| Spatial-Agent context cache | `SQLiteMapCache` | Both agents share one normalized Kakao cache. The upstream MapEval-Textual/Google context snapshot is not bundled. |
| MapEval-API benchmark (300 rows, live Google Maps) | `dataset/seoul_kmapeval_v2_mcq_100.jsonl` (100 rows, live Kakao) | Class mix mirrors MapEval-API's answerable half (nearby 30 / trip 24 / routing 23 / poi 23) so the multi-hop families the paper's gains come from are actually present. Gold answers are computed from Kakao Local and Kakao Mobility, the same provider the agents query. `unanswerable` is excluded — see below. |
| SFT and DPO | Not implemented | This repository evaluates the off-the-shelf prompting path and does not claim fine-tuned Qwen results. |

## Remaining non-equivalences

- Template retrieval is keyword based, not embedding based.
- Factorization and concept binding are deterministic, not SFT/DPO learned.
- Evidence comes from one of two interchangeable sources, recorded per run in
  `metadata.provider`: a per-question context shipped with the dataset (the port of upstream's
  MapEval-Textual setting, see below), or live Kakao POIs.
- Kakao Mobility support is driving-only.
- The benchmark router uses the LLM analysis intent; Korean heuristics are only a fallback when the
  returned intent is missing or unsupported.
- The MCQ trip format evaluates option-order comparison, as MapEval's does. `trip_optimal_order`
  offers four orderings whose travel totals differ by at least 180 s, and `trip_feasible_count`
  asks how many stops fit a budget; both are re-derived through `distance_matrix` + `tsp_tw` in
  `data/verify_benchmark.py`. `tsp_tw` is still never silently substituted for option semantics.
- Kakao Local exposes phone and Kakao URL but not rating, price level, or structured opening hours.
  These optional `Place` fields can only be used when an authoritative enriched cache supplies them;
  missing values are never invented.
- Offline coordinate-to-timezone resolution deliberately covers the Korean benchmark extent only.
- Grounded nearby retrieval fans a requested place type out over the Kakao keywords and category
  codes that actually cover it (`_nearby_retrieval_specs`), because Kakao POI names diverge from
  the requested word (경찰서 also appears as 파출소/지구대/치안센터, 도서관 as 문고/도서실). This is
  planner knowledge inside Spatial-Agent, not a tool or evidence source the ReAct baseline cannot
  reach — ReAct issues the same `nearby_places` tool with whatever keyword its own LLM chooses.
- The planner's authored graph is still bounded by `MAX_REASONING_STEPS`; the deterministic
  retrieval fan-out added during grounding gets its own allowance on top of that bound.

These differences mean the project implements the paper's explicit graph formalism and execution
constraints, but it does not claim numerical reproduction of the paper's trained model or original
map-data environment.

Reports from this repository must therefore be labeled **prompting-only**. They are not directly
comparable to the paper's SFT+DPO headline results, because the learned policy, original weights,
embedding example store, and original map evidence snapshot are not included.

## Context evidence (upstream's local context cache)

Verified against `ecerybao/Spatial-Agent` at `6876bba`: `src/tools/context_parser.py`,
`src/tools/local_context_db.py`, `data/build_cache.py`, `src/agent/operators.py`, `README.md`.

What upstream does:

- Evaluation runs on **MapEval-API** — the multiple-choice rows *without* context
  (`test_agent.py`); the cache is built separately from **MapEval-Textual**, whose rows carry the
  `context` field (`data/build_cache.py` → `data/context_cache.db`).
- The cache is **one SQLite database over the whole corpus** — tables `places`, `travel_times`,
  `routes`, `nearby_places` — and every question queries all of it, with a four-level fuzzy
  cascade (exact → LIKE → normalized → Levenshtein).
- Operators query the cache first and **fall back to the Google Maps API on a miss**
  (`query_local_place` returns `None`, the geocode call runs). The README states this plainly:
  "Google Maps API fallback for cache misses."
- **The cache is optional and is not the paper's evaluation setting.** `SpatialAgent.__init__`
  always constructs a `GoogleMapsClient` and only initializes the cache `if db_path.exists()`,
  logging a fall-through to the live API otherwise; `test_agent.py` has no flag for it at all; and
  the README calls it "an optional local context cache" for "faster and more reliable evaluation",
  shipping no prebuilt database. It is a cost and latency shortcut in a reimplementation, not the
  arrangement the paper's MapEval-API numbers come from — which is live API calls.
- **MapEval-API and MapEval-Textual are the same 300 questions**, ids included; API is Textual
  with the `context` field removed. So a cache built from Textual and used on API is, question for
  question, the retrieval that answers it — which is what "the strongest results depend on a local
  SQLite context cache" amounts to.
- `get_nearby_places(reference_place, category)` returns **the stored block for that reference
  place and nothing else**, re-ranked by haversine distance. It never scans the place table by
  radius.
- **The agent never sees the context text.** `test_agent.py` does not mention it, and no agent
  module reads it outside the database.

What this repo does, and why:

- Same shape: the corpus is built from every context in the dataset and shared by all questions,
  loaded *behind* the tool layer so both architectures still choose tools and still read
  normalized `Place` / `Route` objects. `BenchmarkItem.agent_input()` is unchanged.
- The context travels in the benchmark row rather than in a second file. `main.py` collects
  `item.context` across the dataset and builds one corpus, which is upstream's arrangement without
  the extra artifact. Nothing per-question is bound: an earlier revision scoped the corpus to the
  running question, and that made the mere existence of a name an answer signal — "which option
  exists at all" answered 14 of 100 questions under per-question scoping and 9 under the shared
  corpus.
- `--provider hybrid` is upstream's cache-then-live arrangement with Kakao in Google's place.
  `--provider context` (the default) runs the corpus alone, so a run needs no Kakao key and a miss
  stays a miss.
- **Retrievals are computed, not replayed.** This is the one place the port deliberately does not
  follow upstream, and the reason is an evaluation-validity flaw in upstream that this repo
  reproduced and then measured. MapEval-API is MapEval-Textual with the `context` field removed —
  the same 300 questions, the same ids — so a cache built from Textual holds, for every API
  question, the retrieval result that answers it. `get_nearby_places` returns that block already
  filtered by type and already sorted by distance, which makes one tool call sufficient and
  collapses the API setting into the Textual one. Ported faithfully, it produced ReAct 100/100.
  Here the block contributes its *places* to the corpus and `nearby_search` computes the ranking
  from coordinates over the whole corpus, filtered by type through `TYPE_SYNONYMS`. That is also
  what a live map API does: Kakao and Google compute, only a frozen context is pre-computed.
- Counters follow the upstream framing: the corpus *is* the cache, so it costs no API call; an
  answered lookup is a hit, an unanswerable one a miss.

Deliberate deviations from upstream, and the reason for each:

- **Normalized objects, not text blocks.** Upstream's `get_place_by_name` returns the context's
  `information` text. This repo's standing invariant is that raw provider text never reaches an
  agent, so the corpus is parsed into `Place` / `Route`.
- **Stricter name matching.** Upstream's cascade ends in a Levenshtein search over every place
  name in the corpus. Korean POI names of one kind share long generic affixes, so the Kakao path's
  `NAME_MATCH_FLOOR` and `distinguishing_similarity` apply here too, with containment narrowed to
  one direction — undirected, `편의점` inside `다모아편의점` let a place-type option resolve to the
  place the question was asking about.
- **A place is not near itself.** Upstream never excludes the reference place from its own nearby
  block, and never had to: Google's "Nearby Restaurants of St. Lawrence Market" does not contain
  the market. Our OSM-derived questions ask a 편의점 for its nearest 편의점, so the anchor heads
  its own block at 0 m and won every ranking.
- **No intent gate.** Upstream's `ContextManager.should_use_local_db` restricts the cache to
  `{routing, trip, poi, nearby}`, which is all four answerable MapEval classes — the gate excludes
  nothing. Porting it to this repo's eight-way vocabulary would exclude four classes for no reason,
  so it is not ported.
- Place types are served in the context's own vocabulary (`convenience_store`, `amenity=bank`).
  Translating them into the Korean nouns a place-type question offers as options would be supplying
  part of the answer.

## Kakao-specific constraints

- Raw Kakao JSON never reaches an agent.
- Kakao Local has no standalone place-details REST method; `place_details` and
  `batch_place_details` read normalized cached records from earlier retrieval/enrichment.
- `reverse_geocode` uses `/v2/local/geo/coord2address.json`; waypoint routes use the official
  `POST /v1/waypoints/directions` and verify returned waypoint names and coordinates — within
  `WAYPOINT_TOLERANCE_M`, because Kakao echoes a waypoint about a metre from the coordinate it was
  sent and snaps a POI to the nearest road. The check is for a route sent through somewhere else,
  which is hundreds of metres away, not for rounding.
- `geocode` falls back to the place-name index when the address index has no entry. Google's
  geocoder answers a place name; Kakao keeps addresses and place names apart, and a planner writes
  the question's place name into `geocode` as readily as an address — `대림동 우리 골목형상점가`
  has one exact place entry and no address entry. The fallback is the lookup `place_search`
  performs, so it reaches nothing the tool surface does not already carry.
- Unsupported travel modes fail explicitly. Kakao Mobility routes cars only, where Google Directions
  answers a walking query, so every "걸어가기에 가장 가까운" question was costing a planner a failed
  call: `GRAPH_PROMPT` says that phrasing means `haversine_distance` / `nearest(metric="haversine")`
  here. The benchmark's own golds for those families are straight-line distances, so nothing is
  approximated by saying so.
- **A leg from a place to itself is answered, not requested.** Google returns a zero-length route
  for identical endpoints; Kakao Mobility refuses with "출발지와 도착지가 5 m 이내로 설정된 경우
  경로를 탐색할 수 없음". A trip matrix asks for its own diagonal, so one run collected 750 of
  those refusals through `distance_matrix` and 64 more through the baseline's `travel_time`, and
  the generation stage read them as legs that had failed. `_self_route` answers the leg locally in
  `directions`, `travel_time` and `distance_matrix` alike — the same evidence for both
  architectures — while an absent *off-diagonal* leg is still reported as missing.
- SQLite stores normalized `Place` and `Route` payloads, not raw Kakao responses or API keys.
- **Region prior on name lookups (`KAKAO_SEARCH_CENTER` / `KAKAO_SEARCH_RADIUS_M`).** Neither
  upstream implementation needs one: Google Places disambiguates from a session location, and the
  paper's MapEval-Textual snapshot is a closed evidence set. Kakao Local searches nationwide, and
  Korean POI names repeat across cities, so an ambiguous name resolved to whichever city ranked
  first — 18 of the 100 benchmark anchors landed outside 서울, including 제주, 경남 and 경북, and
  every operator downstream then computed correctly over a POI in the wrong province. The prior
  biases the *first* keyword query to the benchmark's region; a name with no match there still
  resolves nationwide, so it can never hide evidence. It is a deployment setting read from `.env`
  and applied inside `KakaoMapProvider`, identically for ReAct and Spatial-Agent: it says where to
  look, never which option is correct, and no `BenchmarkItem` field reaches it. Blank disables it.
  Reports must state whether it was enabled, since it changes which places the tool layer can see.
- **Coordinate literals accepted where a place name is expected.** Google's Places API takes a
  `location` bias directly, so upstream never has to name a point. Kakao Local resolves a place
  reference by keyword search, so an agent passing back coordinates it already retrieved got
  `PlaceNotFoundError`. `_parse_coordinate_literal` resolves a `"lat,lng"` string into a `Place`
  without a network call. It adds no evidence — the coordinates came from an earlier tool result.
- **Name matching is capped by the distinguishing residue.** Upstream compares English POI names,
  where a brand and its branch share little. Korean POI names of one kind share long generic
  affixes (서울…초등학교, …주유소, CU …점), so plain string similarity clears any sane floor for
  two different places. `distinguishing_similarity` compares only what is left between the shared
  prefix and suffix, and option-to-POI matching (`_assign_unique_matches`) is one-to-one so a
  single retrieved POI cannot answer several options at once.
- **Neighbourhood membership is not name evidence.** Google Places answers a keyword query with
  places bearing that name, so upstream can trust a location-biased search. Kakao Local answers a
  name it does not carry with places of the same *kind* near the bias point: `신사정육점` returned
  `한아름축산`, `쌍문1치안센터` returned `수유6치안센터`. Both looked resolved, and both replaced
  the question's POI with a different one. The anchored branch of `_resolve_batch` therefore applies
  the same name-evidence floor as the nationwide branch, and falls back to the nationwide search
  when the neighbourhood has nothing by that name. The one licence proximity buys is
  `allow_cross_script`, for a brand whose Kakao entry is transliterated (`A TWOSOME PLACE` /
  투썸플레이스); `strict_names`, which the distance path sets, withdraws it.
- **Option texts may carry an address, and institutions may carry two names.** The Korean source
  datasets disambiguate namesakes by appending the address to the option text and follow OSM's
  choice between 치안센터 and 파출소, neither of which appears in a Kakao place name.
  `strip_location_qualifier` drops the appended address before any name comparison or keyword
  query, and `_search_key` folds the two institution words together. Upstream needs neither: its
  option texts are bare names over a single naming authority.

## What MapEval's own answer encoding turned out to be

Measured directly on `MapEval-API.jsonl` (300 rows), because it decides what can be ported:

- **`answer` is 1-based.** The distribution over all 300 rows is `{1: 74, 2: 73, 3: 71, 4: 62,
  0: 20}`, and the twenty `0`s are exactly the twenty `unanswerable` rows. `0` is a sentinel
  meaning *no option is right*, not an index. This repository's contract is 0-based throughout
  (`^^N^^`), so any port of a MapEval row must shift the index and drop the sentinel.
- **`unanswerable` is therefore a refusal channel, not a class of question.** It is excluded from
  `dataset/seoul_kmapeval_v2_mcq_100.jsonl`: mapping the sentinel onto a real option index would
  make "always answer the first option" score 20/20 on it, and the MCQ format has nowhere to put
  a refusal. Adding it would mean adding an answer channel to `parse_answer`, the evaluator, and
  both agents — a separate change, not a dataset one.
- **MapEval-API is MapEval-Textual minus `context`.** Same 300 ids, 297 identical question
  strings, and `MapEval-Textual.jsonl`'s key set is API's plus `context`. A cache built from
  Textual and queried by API is therefore question-for-question an answer key, which is why
  `ContextMapProvider` computes retrievals over the whole corpus instead of replaying the stored
  block (see above).
- **The ReAct baseline's budget is 30 tool rounds** (`max_tool_rounds: int = 30` in
  `mapeval-api/mapeval_api_evaluator.py`). `MAX_REASONING_STEPS=30` matches it, which matters for
  the trip family: a four-option ordering question needs twelve route legs, so a smaller budget
  would make a ReAct loss a budget artifact rather than an architectural result.

## Why the first Kakao run inverted the paper's result

The first run of `dataset/seoul_kmapeval_v2_mcq_100.jsonl` against live Kakao gave ReAct 90/100 and
Spatial-Agent 77/100 — the paper's ordering reversed, and worst on `trip` (ReAct 23/24 against
Spatial-Agent 15/24), the family Spatial-Agent exists for. Four causes, three of them defects in
this port rather than findings about the architecture:

1. **`distance_matrix` and `tsp_tw` could not compose** (fixed). The tool returned
   `{"routes": [...]}`; the operator read `distance_matrix["matrix"]`. No planner could bridge
   that, so `tsp_tw` never ran on a real matrix and the trip questions were answered by summing
   `directions` calls instead. See the invariant in `AGENT.md`.
2. **`DistanceMatrixArgs.origins`/`destinations` rejected `batch_geocode` output** (fixed). A
   planner writes `origins: "$places"`; the records are `{query, place, candidates}`, and every
   other place argument in the registry already normalized that shape. Matrices failed with a
   64-error `ValidationError` before a route was requested.
3. **The `Route-Optimize` template taught an unexecutable pattern** (fixed). Its example carried a
   hardcoded literal matrix and never showed where a matrix comes from, and `GRAPH_PROMPT`
   separately told planners to answer trip questions with route pairs and `aggregate_route_groups`.
   Together they steered the planner away from `tsp_tw` even when the right template was retrieved.
   The prompt now describes the matrix-to-`tsp_tw` chain and the two trip question shapes.
4. **Intent classification had no rule separating trip from routing, or poi from anything**
   (fixed): 12 of 24 trip rows were classified `routing`, and `poi` was never predicted at all for
   23 poi rows. After the prompt change, trip intent went from 11/24 to 23/23 on a re-run and trip
   answers from 15/24 to 18/23.

What is *not* a defect, and matters for reading any number from this benchmark: **its questions are
explicit and closed.** They name the anchor exactly as Kakao stores it, name the category, state
the radius, and enumerate the orderings to compare. One `batch_geocode` plus one `distance_matrix`
answers a trip question, so a ReAct loop has one decision to get right where Spatial-Agent has
four (intent, graph, bindings, generation). MapEval-API's questions are implicit and open — "I am
at X and hungry, where can I eat quickly?" — and that is where decomposition earns its overhead.
A benchmark of fully specified questions measures pipeline reliability, not composition, and should
not be reported as evidence for or against the paper's claim on its own.

## Why harder questions widened the gap instead of closing it

`dataset/seoul_kmapeval_v3_mcq_100.jsonl` was built to test the three Appendix E families v2 never
exercised — Time-Window-Reverse, Multi-Segment-Aggregate, Object-Field-Measure — on the assumption
that compositional questions would let the operator graph earn its overhead. The first run went the
other way: ReAct 92/100, Spatial-Agent 67/100, against 90/77 on the shallower v2. Spatial-Agent
also issued **177 tool calls to ReAct's 400**, which is the tell: it was not failing to reason over
the evidence, it was failing to fetch it.

The losses were not spread across the compositional families. Five of seven were fine or close
(`poi_bearing_and_distance` 14/14 both, `poi_brand_share` 14/14 both, `trip_latest_departure` 13/14
against 14/14). Two collapsed, and each for a single mechanical reason:

- **`trip_finish_time` 8/16.** The authored graph was already correct — `batch_geocode` then
  `calculate_finish_time` with `stay_durations_s` of the right length. It failed on
  `ValueError: Invalid isoformat string: '오전 10시 00분'`: the planner copied the time out of the
  Korean question, and the temporal operators accepted only ISO 8601. `parse_clock_text`
  (`src/tools/spatial.py`) now reads 오전/오후/아침/저녁/밤 with the 12-o'clock boundaries, and
  `_parse_datetime` falls back to it, which fixes `calculate_start_time` on the same path.
- **`nearby_from_need` 1/14.** Every wrong answer was the nearer decoy of the wrong kind the
  family plants deliberately, so the category constraint was never applied. `normalize_analysis`
  returned only `{intent, concepts, measure}`, so a place type the Analysis stage inferred from a
  need was discarded before grounding could bind it; the retrieval then ran with no category and
  the ranking answered "nearest of anything". Analysis now carries `target_type` and
  `_ground_graph_literals` binds it when the question text names no type.

A third defect was in the dataset rather than the port: two families were labelled by their
Appendix E family instead of the intent the agent routes on, and `classification` *is*
`SUPPORTED_INTENTS`. A question that searches a stated radius around one anchor is `radius`
whatever macro family it exercises; the family stays in `template_id`. `Object-Field-Measure`
gained `radius`/`nearby` intents so a neighbourhood-scoped share question can still retrieve it.

The general lesson, and the reason these are recorded as invariants in `AGENT.md`: **a
compositional question only gives structure a chance to pay off — any one stage that cannot read
the question throws that chance away entirely.** Three separate extractors had been written
against a single Korean phrasing each (`반경` only, `에서 가장 가까운` only, ISO 8601 only), and
each silently disabled a whole family while every other stage worked.

## Kakao routing is only reproducible by distance

Measured on one pair (은보갤러리 → 롯데슈퍼프레시 광진구의점, 13,990 m apart in a straight line):

| priority | distance | duration |
|---|---|---|
| RECOMMEND | 17,879 m, then 23,041 m | 2,603 s / 2,457 s |
| TIME | 23,041 m | 3,285 s |
| DISTANCE | 16,992 m | 3,243 s, then 4,337 s |

RECOMMEND and TIME both optimize against live speeds, so the route itself moves between calls —
a 29% spread on distance, wider than the gap between two answer options. DISTANCE is a
shortest-path over the road graph and returned the same distance every time. But the *duration*
moved by a third even on that fixed route, because a duration is always an estimate of current
traffic.

The consequence for this benchmark: a distance gold is a fact about the road network and can be
graded exactly; a duration gold is a snapshot and can only be graded coarsely. `Builder.route`
therefore routes with DISTANCE, and the time-window families space their options at least 85
minutes apart so a traffic estimate cannot reach the neighbouring option. The first v3 build did
neither — it took durations from RECOMMEND and spaced options 25 minutes apart — which is why its
`multisegment_total` golds disagreed with the tools by up to 15% on a whole chain.

## The ReAct baseline was given the composition it was supposed to lack

The paper's claim is that a GeoFlow operator graph beats a ReAct agent working over map API
primitives, and it reports that gain on models as weak as GPT-3.5-Turbo. That rules out "our model
is too small" as an explanation for the gain not reproducing here: a 35B MoE is not worse at
tool-use loops than GPT-3.5-Turbo. The difference is on the other side of the comparison.

MapEval's own baseline (`mapeval-api/Tools.py`) exposes six tools: PlaceSearch, PlaceId,
PlaceDetails, NearbyPlaces, TravelTime, Directions. `src/tools/registry.py` exposes twelve, and
the five extra ones are all aggregations over those primitives:

| tool here | what the paper's baseline must do instead |
|---|---|
| `batch_geocode` | PlaceSearch once per name, across as many turns |
| `batch_place_details` | PlaceDetails once per id |
| `distance_matrix` | TravelTime once per ordered pair, all held in context |
| `calculate_finish_time` | routes, stays and clock arithmetic by hand over many turns |
| `recover_option_places` | ground each option separately |

Aggregating primitives into a typed, executable plan is what GeoFlow is *for*. An earlier revision
of this repository put those aggregations in a tool layer both agents shared, on a rule that the
two must differ only in architecture. **That rule was a design error**, and it is retired. A tool
surface *is* part of an architecture: upstream carries `get_distance_matrix` in
`spatial-agent/src/tools/google_maps.py` and `mapeval-api/Evaluator2.py` hands its baseline nothing
of the kind, so making the surfaces identical did not remove a confound — it deleted the very
difference the paper measures, and handed the baseline the work the architecture was meant to do.

The measurements under the shared surface line up with that reading exactly. In the ReAct run
behind `reports/test_20260819T045330Z.json` (92/100), 79 of 100 questions saw ReAct call at least
one aggregation tool; `trip_finish_time` was 16/16 and `trip_latest_departure` 14/14, every one of
them using them. A four-stop itinerary took five tool calls — one `batch_geocode` plus four
`travel_time` — where the paper's baseline needs four PlaceSearch calls, four TravelTime calls, and
the place ids carried across turns. `batch_geocode` also reconciles the four names against each
other and against an anchor, which PlaceSearch has no equivalent of at all.

So ReAct is now constructed with `allowed=ToolRegistry.MAPEVAL_BASELINE_TOOLS` and `--react-tools`
defaults to `mapeval`. `--react-tools full` restores the shared surface as an explicit ablation —
"does the graph add anything on top of strong aggregation tools" rather than the paper's "does the
graph beat primitives" — and `metadata.react_tools` records which question a run asked. Reports
from the two surfaces must never be pooled; every report predating this default is a `full` run.

The same correction applies to the ReAct prompt. MapEval's baseline is a stock langchain
`STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` agent whose prompt is the question, the options and
the answer format, with no strategy. `REACT_SYSTEM_PROMPT` used to name the benchmark's question
taxonomy and say which tool each shape wants ("coordinates for direction and straight-line
distance", "directions only for road-route questions"): planning handed to the baseline in prose,
the same error in another currency. It now carries role, evidence discipline and the wire format
only. Tool contracts stay in the tool descriptions, which is where MapEval keeps them too.

`MAPEVAL_BASELINE_TOOLS` is exactly the five tools `Evaluator2.py` constructs at line 33 of
`mapeval-api@35d481a`, mapped onto their Kakao counterparts:

| upstream (`FormattedTools.py`) | here | note |
| --- | --- | --- |
| `PlaceSearch` | `place_search` | upstream returns a place id; Kakao's keyword search returns the place itself, so there is no id round-trip |
| `PlaceDetails` | `place_details` | |
| `NearbyPlaces` | `nearby_places` | |
| `TravelTime` | `travel_time` | |
| `Directions` | `directions` | |

Two of this registry's primitives are deliberately **not** in that set. `geocode` (address →
coordinates) and `reverse_geocode` (coordinates → address) have no upstream counterpart at all:
upstream reaches every place through a place id and never converts between an address and
coordinates. They are primitives rather than aggregations, so they do not hand ReAct the
composition GeoFlow provides, but they are still capability the paper's baseline was measured
without — and in the ReAct run behind `reports/test_20260819T045330Z.json`, 40 of the 479 tool
calls were `geocode`, most of them on bare place names (`플라이아트센터`, `노량진컵밥거리`), which
is PlaceSearch reached through a second index rather than address handling.

`Tools.py` additionally defines a `PlaceIdTool` that `Evaluator2.py` never instantiates; an
earlier revision of this document and of `AGENT.md` counted it and called the baseline six tools.
It is five. `FormattedTools.PlaceSearchTool` is documented as "Get place ID for a given
location", which is the same primitive.

### The port is finished, and that is a rule rather than a status

`src/agent/react.py` carries the element-by-element comparison in its header. What matters beyond
the table is the direction of maintenance: **the baseline is not edited in response to a benchmark
result.** An accuracy gap ReAct shows *is* the finding, so closing one by rewriting its prompt,
raising its budget, or glossing a parameter it reaches would make the baseline a function of the
test set — the same overfitting the shared-tool-surface correction above was about, one layer
down. Only two surfaces stay live, and both are shared with the other architecture: the provider
below the tools, and the tool contracts both agents read. A change to either has to be argued from
the provider or from upstream, never from a question ReAct got wrong.

Two consequences, applied together:

- `directions` returns its turn-by-turn guidance by default, and `travel_time` does not
  (`TravelTimeArgs`). Upstream's `DirectionsTool` prints every step's instructions on every call
  while `TravelTimeTool` reports one duration and one distance — the two tools differ by what they
  *report*, not by what an agent has to know to ask for. Our port had `include_steps=False` on
  both, which made the guidance a capability the port withheld from the baseline rather than one
  MapEval's design withholds; `steps_analysis` on the Spatial-Agent side had to bind the flag
  during grounding to get a route it could count turns on, and that binding is now unnecessary.
- `DirectionsArgs.priority` is documented as `"RECOMMEND, TIME, or DISTANCE"` — the accepted
  values, the way upstream documents `travelMode` as "(driving, walking, bicycling, transit)". An
  earlier revision glossed each value with what it optimizes, written after watching ReAct read a
  question that says 거리가 가장 짧은 경로로 as RECOMMEND. Kakao requires the parameter, so the
  baseline has it; *which* priority a question asks for is grounding, and grounding is a
  Spatial-Agent stage under measurement (`_extract_route_priority`). A write-up should report that
  asymmetry as architectural, which it is, rather than close it in the baseline's prose.


## Why the v2 scores are high, measured rather than guessed

ReAct 91 and Spatial-Agent 96 on `dataset/seoul_kmapeval_v2_mcq_100.jsonl` sit far above what
MapEval reports for its API split, and a *baseline* at 91 from a 35B-A3B model is the part that
cannot be right. Five candidate explanations were measured on the run behind
`reports/test_20260819T224214Z.json` (ReAct) and `reports/test_20260819T225908Z.json`
(Spatial-Agent). Four are refuted:

| candidate | measurement | verdict |
| --- | --- | --- |
| the questions are answerable from the model's own knowledge | no-tool floor **31/100**, chosen-option histogram 29/26/23/22 against a 25% chance rate | refuted |
| our place-matching stack carries the baseline | for the 247 option names Kakao returns candidates for, `_best_place_match` agrees with `candidates[0]` — upstream's `results[0]` — in **208**; the 7 overrides are all non-place option strings ("좌회전 (아차산로)", "7번") | refuted |
| ReAct's 30-step budget against langchain's 15 | ReAct used at most 10 steps, and only **5/100** questions took more than 15 tool calls | refuted |
| Spatial-Agent's score is our question-literal grounding | `_ground_graph_literals` replaced by a pass-through: **96 → 92**, and the loss is almost all `routing` (22 → 18, the priority binding) while `trip` rose 22 → 24 | refuted |
| **the questions do not require the map to find anything** | **91/100** of Spatial-Agent's drafted graphs contain no retrieval operator at all; ReAct called `nearby_places` on **16/100** questions and `place_search` 480 times | **confirmed** |

The shape of a v2 answer, from the trace of a radius question:

```
batch_geocode(["호스텔온기"])                     # the anchor, verbatim in the question
batch_geocode([option0, option1, option2, option3])  # the four options *are* the candidate set
within_radius(center, candidates, 800)               # the radius, verbatim in the question
identity_measure -> ^^2^^
```

**The MCQ options are the candidate set, so retrieval never happens.** "Which 대형마트 is within
800 m" becomes "geocode four names and test a predicate" — which is also exactly what
`data/verify_benchmark.py` does, and why it re-derives 100/100. The hard part of a map agent,
finding the places, is answered by the question paper.

Three construction rules remove what is left:

- `Builder.resolves_to` admits a place only if its bare name searches back to within 200 m, so
  place resolution — the failure mode MapEval's baseline spends its budget on — cannot fail. That
  is also why the naive `results[0]` and our matcher agree.
- Ties are rejected with wide margins, and the recorded evidence shows how wide: a
  `routing_via_compare` row offers detours of 1639 / 3406 / 3568 / 3868 s, a factor of two; a
  `trip_feasible_count` row has a 4 h budget against best finishes of 1.3 / 3.1 / 5.2 h. Only
  `trip_optimal_order` is close (194 s over ~2900 s). A constant "15 minutes per leg" and **no map
  at all** answers `trip_feasible_count` 9/10.
- No `unanswerable` class, where MapEval-API puts 20 of its 300 rows and where models lose most.

And one thing no construction rule can fix: Kakao Local publishes no rating, no price level and no
opening hours, so the whole attribute-reasoning half of MapEval-API — Place-Attribute-Query, and
every question that turns on whether somewhere is open or well reviewed — has no Korean
counterpart here. What remains is geometry and routing, which are exact.

So v2 measures evidence handling and arithmetic over exact lookups, not map reasoning, and its
numbers must never be presented as reproducing the paper's. `dataset/seoul_kmapeval_v3_mcq_100.jsonl`
already carries the fix for the biggest of these: **six of its seven families answer with a value**
(`오후 3시 07분`, `약 49.8km`, `약 86%`, `3번`, `동쪽, 약 6.9km`), which no amount of geocoding the
options can shortcut. Only `nearby_from_need` still offers POI names, and it is the family whose
plans do retrieve.

Report the floor next to every accuracy (`data/measure_no_tool_floor.py`). Without it, 91 against
96 is uninterpretable.

## How MapEval built its QA set, and what v4 ports

`mapqator-backend/database/schema.sql` shows the pipeline behind `mapeval-api/dataset.json`: an
annotator issues map API calls that are cached in `places` / `nearby` / `distance` / `directions` /
`inside`, the calls are assembled into one `context`, and a person then writes the question and the
options against that evidence (`dataset.username` records who). `human` holds the human baseline
that the paper reports models against. There is no generator — the questions are hand-written over
collected evidence.

Counting `dataset.json` says what the annotators actually wrote, per class:

| class | value options | name options |
| --- | --- | --- |
| nearby | 19 | 64 |
| poi | 36 | 28 |
| routing | 44 | 22 |
| trip | 48 | 19 |
| unanswerable | 5 | 15 |

`['South, 13.45 kilometers', …]`, `['61.224 km', …]`, `['10.13', '10.23', …]`, `['1','2','3','0']`.
Every one of v2's eleven families offers place names, orderings or guidance strings instead, which
is where its retrieval bypass comes from — geocode the four options and the question is over.

`data/build_mapeval_benchmark.py` ports the method rather than the proportions:

- **Values where upstream uses values.** 45 of 100 rows. `poi_straight_distance`,
  `routing_distance_via`, `trip_total_distance`, `trip_arrival_clock`, `routing_turn_count`,
  `poi_direction_distance`.
- **A constraint where upstream uses names.** Upstream asks for an *orthopedic* hospital; Kakao's
  paths carry the same subtypes, so `nearby_clinic_subtype` / `nearby_cuisine_subtype` offer three
  *nearer* siblings and the subtype is what decides. The gold is selected with
  `matches_required_type` — the tools' own test — because choosing it on the category path while
  `filter_places` also reads the name left one question whose true answer was not among its
  options.
- **The unanswerable class, which here is not a choice.** Kakao publishes no rating, price level or
  opening hours, so upstream's attribute half cannot be asked. Those 7 rows ask it anyway and the
  gold is the refusal; `data/verify_mapeval_benchmark.py` verifies them in the negative, by
  checking every candidate carries `None` for the field.
- **No engineered margins.** v2 rejected close calls; MapEval's annotators read whatever the
  evidence said. Margins survive only where Kakao is not reproducible: `trip_arrival_clock`
  out-spaces the traffic, `trip_feasible_count` keeps only budgets whose answer holds at ±30% per
  leg. `trip_feasible_count` additionally rejects any row a constant travel assumption answers —
  the flaw measured on v2, where "15 minutes a leg" and no map scored 9/10.

### The floor is part of the build loop, not a report afterwards

The first cut of v4 measured **52/100** with no tools, which is not a benchmark. The leaks were in
the option sets and the fix came from upstream's own data:

| family | first floor | leak | fix | floor now |
| --- | --- | --- | --- | --- |
| `nearby_clinic_subtype` | 15/16 | the gold was the only option carrying 정형외과 in its name | three options of the requested subtype plus one *nearer* place of another — upstream's "nearest Mosque among four mosques" beside its "orthopedic hospital among several specialties" | 4/16 |
| `nearby_cuisine_subtype` | 8/12 | same, weaker | same | 5/12 |
| `poi_which_is_closer` | 6/7 | two options over a city the model knows | dropped; its rows went to `poi_straight_distance` | — |

**35/100 overall**, against a 25% chance rate, with every measured family at or near chance:
`routing_turn_count` 0/7, `trip_feasible_count` 1/7, `trip_total_distance` 1/7,
`routing_distance_via` 2/8. The 7 `unanswerable_*` rows score 5/7 without tools, and that is
correct rather than a leak — recognising that a question has no answer is a knowledge task, which
is why MapEval includes the class at all.

Building v4 also surfaced a construction bug the other benchmarks may share: a question was asked
about the *pool's* copy of its anchor while an agent reaches the anchor through `place_search`, and
the two sit up to `ROUND_TRIP_TOLERANCE_M` apart. One 피부과 question put its gold at 153 m from
the pool's 신사동가로수길 while the resolved anchor had a different clinic at 93 m — a gold no
agent could reach. `Builder.as_resolved` returns the resolved place, and the same tolerance is why
a same-subtype option must sit at least 60 m farther than the gold.

Two deviations, both because we have no annotators and one because upstream leaks:

- The questions are templated. The *shape* of every family is upstream's; the phrasing is
  generated, so v4 does not carry the colloquial naming ("the hospital", "the lake") that makes
  upstream's place resolution hard.
- `Evaluator2.py` prepends "Option0: Unanswerable" only when a row is unanswerable, which tells the
  model the answer before it reads the question. Here "주어진 지도 정보로는 알 수 없음" is an
  ordinary option: gold on the 7 unanswerable rows and a distractor on 22 answerable ones.

## What v4 measured, including the part that refutes the diagnosis

First paired run, `reports/test_20260820T011219Z.json` (ReAct) and `test_20260820T014038Z.json`
(Spatial-Agent), prompting-only, `--react-tools mapeval`, provider kakao:

```
no-tool floor  35/100
ReAct          89/100
Spatial-Agent  91/100
```

**The option-set bypass was not what made v2 easy.** Closing it — 45 of 100 rows now answer with a
value, and the constrained families punish both name-reading and proximity — moved the floor from
31 to 35 and the accuracies from 91/96 to 89/91. The families with the lowest floors are the ones
the agents score highest on: `routing_turn_count` floor 0/7 against 7/7 and 7/7,
`trip_total_distance` floor 1/7 against 7/7 and 7/7. So the questions do require the map, the
agents use it, and they are simply good at it. That correction stands against the hypothesis in the
section above, which predicted a substantial drop.

Two things it did establish:

- **The architectures are indistinguishable here.** 83 questions both answer, 3 neither, 6 ReAct
  only, 8 Spatial-Agent only — exact McNemar p = 0.79.
- **A graph that must end in a Measure cannot refuse.** All three of Spatial-Agent's losses to
  ReAct outside the noise are `unanswerable_*` rows: ReAct reads the options, finds the map silent
  and picks the refusal; the GeoFlow pipeline composes a graph, executes it and reports a number.
  That is an architectural property worth a line in a write-up, and it is the opposite of the
  paper's direction.

What is left to explain the gap with MapEval's reported numbers, in the order the evidence supports:

1. **The evidence layer never fails at plumbing.** Upstream's agent threads `place_id` *strings*
   parsed out of langchain's text protocol through a REST backend: `PlaceSearch` returns an id,
   every other tool consumes one, and dropping or garbling it is a whole error class. Ours passes
   normalized `Place` objects, accepts names and coordinates interchangeably, and normalizes every
   place-typed argument. Measured on this run, only 5.9% of ReAct's tool calls and 1.1% of
   Spatial-Agent's returned an error at all. That difference sits *below* both agents, which is
   exactly why both land near 90 — and porting it faithfully (ids only, text returns) is the one
   remaining change that would move these numbers.
2. **The attribute half of MapEval-API cannot be ported.** "Can I visit at 5 PM Saturday?", "a
   restaurant rated 4.8+", "does it serve dinner" — Kakao Local publishes no rating, price level or
   opening hours. Those are where MapEval's models actually fail, and here they exist only as the
   7 unanswerable rows.
3. **The questions are templated.** No colloquial referring expressions ("the hospital", "the
   lake"), no typos, and every place named as Kakao stores it, so the question text never makes
   resolution hard.
4. **The model is current.** `nvidia/Qwen3.6-35B-A3B-NVFP4` against the paper's 2024 frontier
   models, on a benchmark whose measure is exact arithmetic over exact lookups.

## Porting the id-threading, and what it cost

The section above named the plumbing as the last untested explanation for the gap with MapEval's
reported numbers, so it was ported: a place argument on the five baseline tools is now a reference
the provider issued, and a name is refused with an error saying to call `place_search` first
(`ToolRegistry._reference`, `MapProvider.dereference`, and the same refusal inside both providers).
The aggregations keep resolving names, because resolving the names a plan holds is what makes them
aggregations over PlaceSearch.

Same benchmark, same model, before and after:

| | tool calls | error rate | "not a reference" errors | accuracy |
| --- | --- | --- | --- | --- |
| ReAct, names accepted | 579 | 5.9% | — | 89/100 |
| ReAct, references only | 784 | **25.5%** | 159 | 87/100 |
| Spatial-Agent, names accepted | 1080 | 1.1% | — | 91/100 |
| Spatial-Agent, references only | 1082 | 0.7% | 0 | 91/100 |

The burden is real and it is asymmetric exactly as upstream's design implies: a quarter of ReAct's
tool calls now fail, 159 of them on the id-threading MapEval's baseline does by hand, while
Spatial-Agent never hits it — its planner resolves names in one `batch_geocode` and carries
references through the graph, which is the architectural claim the paper makes.

**It moved the accuracy by two points.** ReAct 89 → 87, Spatial-Agent 91 → 91, exact McNemar
p = 0.50 — still indistinguishable. The reason is visible in the same table: ReAct *recovers*. It
reads the error, calls `place_search`, and retries, spending 784 calls and 550 reasoning steps over
100 questions — an average of 5.5 against a budget of 30. The plumbing costs it turns, not answers,
because the budget absorbs them and the error message says what to do. Upstream's 15-iteration
default and its silent "Incorrect place name" would absorb fewer.

So the ledger on this benchmark now reads: floor 35, ReAct 87, Spatial-Agent 91, and the
architectures separated by 4 points that a paired test does not distinguish. What remains
unported is not plumbing but evidence — Kakao publishes no rating, price level or opening hours,
so the half of MapEval-API that turns on reading an attribute table cannot be asked here at all,
and the questions are templated rather than written by a person. A write-up should say that the
comparison reproduces MapEval's *method* and not its *difficulty*.

One family keeps pointing the other way, and it is the one worth reporting: Spatial-Agent scores
3/7 on the `unanswerable_*` rows against ReAct's 7/7. A graph that must terminate in a Measure
composes, executes and reports a number; ReAct reads the options, finds the map silent and picks
the refusal.
