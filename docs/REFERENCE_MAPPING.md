# Reference implementation mapping

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
  `POST /v1/waypoints/directions` and verify returned waypoint names and coordinates.
- Unsupported travel modes fail explicitly.
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
