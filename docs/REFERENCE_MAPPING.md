# Reference implementation mapping

This file describes the current port and its deliberate deviations. Current operational rules
are in [AGENTS.md](../AGENTS.md). Past measurements, superseded designs, and benchmark summaries
are in [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md); their historical advice is not a current rule.

| Upstream concept | K-MapEval implementation | Deliberate deviation |
|---|---|---|
| MapEval `Evaluator2.py` structured ReAct loop | `src/agent/react.py` | Removes the localhost backend, sleeps, and remote writes. Uses user-requested 0-based `^^N^^` answers. |
| MapEval tools/backend | `src/tools/registry.py`, `src/tools/map.py`, `src/tools/kakao.py` | Provider injection replaces the separate HTTP backend; tools expose normalized JSON. |
| Spatial information theory analysis | `normalize_analysis`, `src/agent/concepts.py` | Preserves all seven core-concept labels and six roles. Runtime intent classification and dataset labels are absent; missing concepts are completed from typed facts and marked implicit/synthetic. |
| Concept transformation drafting | `src/agent/templates.py`, `src/agent/composition.py`, Spatial-Agent `compose` | Appendix E's ten reusable macro fragments expose typed I/O ports. Retrieval reads typed concepts and factors, and templates are priors rather than hard constraints. The planner authors transformation edges and refers to the Analysis stage's concepts by id; it declares only concepts that stage does not carry. |
| Concept graph `G` | `GeoFlowGraph`, `ConceptNode`, `TransformationEdge` | Vertices are concepts with `core_concept`, `functional_role`, and `attributes`; directed hyperedges are transformations. The executable operator graph is a later representation. |
| Factors and operator-concept hypergraph `G'` | `FactorNode`, `attach_grounding_factors`, `factorize_semantic_graph` | Radius, ordinal, direction, time budget, stays, route objective, fixed order, and return-to-start are explicit factor vertices connected to transformation/operator hyperedges. Tool selection remains deterministic from types, factors, and contracts. |
| Five GeoFlow constraints | `src/agent/canonicalization.py`, `src/agent/validation.py`, `normalize_and_validate_graph` | Deterministic canonicalization completes the drafted graph's references before G1–G5 read it: the Analysis stage's concepts and their derived factors are in scope, an undeclared output is typed by the transformation that produces it, a redefined concept is renamed and later readers follow it, and a produced concept's type comes from the vocabulary rather than the planner. Every completion is additive and last-resort, so a draft that already passes canonicalizes to itself. G1–G5 then refuse strictly on both the draft and the repair. This port's own output-type, role-ordering and argument-value checks are skipped (`strict_types=False`) on a last attempt over the repaired graph and then the original; a graph still invalid there becomes `graph_validation_failure`. |
| Core-concept execution types | `OPERATOR_CONTRACTS`, `SpatialOperatorRegistry` | LOCATION, OBJECT, FIELD, EVENT, NETWORK, AMOUNT, and PROPORTION all have executable producers. The current benchmark directly exercises only a subset. |
| Contextual/functional roles | `factorize_geoflow`, `ROLE_PRIORITY` | EXTENT/TEXTENT are scheduled as context but do not participate in procedural precedence. |
| Topological executor | `src/agent/execution.py`, Spatial-Agent `execute` stage | Executes the validated order through injected registries, records operator state, and separately materializes concept state through output bindings. |
| Lenient concept-reference resolution (`_resolve_concept_reference`, `_extract_coordinates_from_concept`, `resolve_place_name`) | `_resolve_references` / `_descend_reference` in `src/agent/spatial.py`, `_as_place` / `_as_place_list` in `src/tools/spatial.py` | An over-specified `$node.path` degrades to the closest resolvable object instead of failing the operator, and every coordinate operator unwraps the place-shaped record it was handed. Only a genuinely unresolved place raises, as an explicit `PlaceNotFoundError`. |
| Grounded answer generation | `src/agent/answering.py` | The spatial core produces a `GroundedAnswer` from execution evidence without seeing MCQ options. |
| Evaluator answer selection | `MCQAdapter` in `src/mcq_adapter.py` | MCQ reconciliation is outside GeoFlow (exact grounded text, then unambiguous containment/value match) and preserves the 0-based `^^N^^` contract. An index emitted by the grounded-answer stage is ignored; `MATCH_OPTIONS` is not a core transformation. |
| Appendix C operators | `ToolRegistry`, `SpatialOperatorRegistry` | Adds reverse/batch details, waypoint directions, instruction route filtering, extractors, pairwise extremes, place filtering, travel-time nearest, temporal operators, and step analysis with paper-level semantics. |
| TSP-TW | `SpatialOperatorRegistry.tsp_tw` | Exhaustive optimization up to nine nodes with service times, windows, and budget; infeasible instances return the paper's nearest-unvisited partial feasible fallback. OR-Tools is not bundled. |
| Temporal operators | `timezone`, `open_at_time`, `calculate_finish_time`, `calculate_start_time` | Handles cross-midnight/24-hour periods. Multi-stop finish time queries cached/live route durations and adds stays. Latest-departure calculation is an explicit template helper. |
| Spatial-Agent evaluator/generator | Spatial-Agent `evaluate` and `generate` | The LLM conditions on question, final state, and full trace. Deterministic match evidence no longer overwrites the generated selection. |
| Google Maps client | `KakaoMapProvider` | Kakao Local handles search/geocode/reverse-geocode/nearby; Kakao Mobility handles driving and verified multi-waypoint routes with optional guides. |
| Spatial-Agent context cache | Not implemented | Legacy dataset context fields are ignored. `SQLiteMapCache` caches normalized Kakao responses only; it is not a MapEval context corpus. |
| MapEval-API benchmark | `data/build_kmapeval_dataset.py` | Live Kakao evidence; current generation uses the upstream class proportions plus the explicit unanswerable extension. Dataset/revision-specific measurements are in EXPERIMENT_HISTORY.md. |
| SFT and DPO | Not implemented | This repository evaluates the off-the-shelf prompting path and does not claim fine-tuned Qwen results. |

## Remaining non-equivalences

- Macro-template retrieval is typed and deterministic. Question–validated-graph demonstration
  retrieval uses embedding cosine similarity when an embedding backend is configured; exact test
  questions and explicitly excluded example IDs cannot be returned.
- Factorization and concept binding are deterministic, not SFT/DPO learned.
- Evidence always comes from Kakao and is recorded as `kakao` in `metadata.provider`.
- Kakao Mobility support is driving-only.
- Spatial-Agent has no benchmark-label or intent router. `classification`, `mapeval_class`, and
  `template_id` remain evaluator/reporting metadata only.
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
- The planner's authored graph is bounded by `SPATIAL_MAX_STEPS` (falling back to
  `MAX_REASONING_STEPS` when unset); the deterministic
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

Current K-MapEval behavior:

- The context-cache port has been removed. Every run constructs `KakaoMapProvider`; there is no
  `--provider` selector, corpus parsing, or context-to-Kakao fallback.
- Legacy MapEval-Textual `context` fields remain in their dataset rows for provenance, but they are
  metadata only: runtime code never collects them, parses them, or derives provider configuration
  from them.
- `SQLiteMapCache` remains because it caches normalized Kakao requests and responses for both
  architectures. It is independent of the removed MapEval context corpus.

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
  경로를 탐색할 수 없음". A trip matrix includes its own diagonal. `_self_route` answers the leg locally in
  `directions`, `travel_time` and `distance_matrix` alike — the same evidence for both
  architectures — while an absent *off-diagonal* leg is still reported as missing.
- SQLite stores normalized `Place` and `Route` payloads, not raw Kakao responses or API keys.
- **Region prior on name lookups (`KAKAO_SEARCH_CENTER` / `KAKAO_SEARCH_RADIUS_M`).** Neither
  upstream implementation needs one: Google Places disambiguates from a session location, and the
  paper's MapEval-Textual snapshot is a closed evidence set. Kakao Local searches nationwide, and
  Korean POI names repeat across cities, so an ambiguous name resolved to whichever city ranked
  first. The prior biases the *first* keyword query to the benchmark's region; a name with no match there still
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

## Reference evaluation conditions

The reference ReAct surface preserves both upstream's argument contracts and loop behavior:
`reference` executes one action per iteration, stops without an answer at its limit, and uses
15 iterations. `native` (legacy alias `mapeval`) and `full` are separate ablations.

Spatial-Agent counts authored transformation edges, not loop iterations. The project's recorded
comparison condition is ReAct 15 / Spatial-Agent 30, concurrency 32. Each architecture's optional
step setting falls back to MAX_REASONING_STEPS; record actual settings for every run.

The upstream report's measured accuracy and evidence conditions are preserved in
[the experiment history](EXPERIMENT_HISTORY.md#what-upstreams-own-number-was-measured-with).
A comparison with that report must state its context-assisted evidence setting; the current
Kakao-only runtime has no curated context corpus. Do not treat those settings as equivalent.
