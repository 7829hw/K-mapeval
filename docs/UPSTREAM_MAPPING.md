# Upstream mapping

Every deviation from the two upstream implementations on this branch, and why it exists.
Add to this file when you make another one.

The branch's premise is that **both agents are upstream's code, and the only thing swapped is
the map API.** That premise is only worth anything if the swap is auditable, which is what
this document is for. If a deviation is not listed here, it is a bug.

| what | upstream | here |
| --- | --- | --- |
| Spatial-Agent | [`ecerybao/Spatial-Agent`](https://github.com/ecerybao/Spatial-Agent) @ `6876bba` | `src/spatial_agent/` |
| ReAct baseline | [`MapEval/MapEval-API`](https://github.com/MapEval/MapEval-API) @ `35d481a` | `src/mapeval_api/` + `src/agent/react.py` |
| map API | Google Maps (Places, Directions, Distance Matrix, Geocoding, Time Zone) | Kakao Local + Kakao Mobility, via `src/kakao_maps.py` |

Verify the vendored trees have not drifted:

```bash
diff -r ~/spatial-agent/src/agent src/spatial_agent/agent   # only the renames in §1
diff ~/mapeval-api/Evaluator2.py src/mapeval_api/Evaluator2.py   # identical
```

---

## 1. Spatial-Agent: what was changed

`src/spatial_agent/` is upstream's `src/` tree. Every `.py` file under it is byte-identical to
upstream apart from one mechanical rename and one replaced module.

**The rename**, applied with `sed` across the tree, touching no logic:

| upstream | here |
| --- | --- |
| `from src.tools.…` / `from src.agent.…` | `from src.spatial_agent.tools.…` / `from src.spatial_agent.agent.…` |
| `google_maps` (module) | `kakao_maps` |
| `GoogleMapsClient` | `KakaoMapsClient` |
| `google_client` (attribute, parameter) | `kakao_client` |
| `google_api_key` (parameter) | `kakao_api_key` |
| `GOOGLE_MAPS_API_KEY` (env var) | `KAKAO_REST_API_KEY` |
| `from_google_fallback` (dict key) | `from_kakao_fallback` |
| "Google Maps" / "Google API" in comments, log strings and **prompts** | "Kakao Map" / "Kakao API" |

That last row is the only one that changes behaviour rather than names. `planner.py`'s
`GRAPH_PROMPT` tells the planner which operators are API-backed, in prose that named Google;
leaving it would tell the planner it is querying a provider it is not.

**The replaced module.** `src/tools/google_maps.py` became `src/spatial_agent/tools/kakao_maps.py`,
which is a three-line re-export of `src/kakao_maps.py`. The client lives one level up because
the ReAct baseline reads the same one — see §3.

**Nothing else.** `agent/operators.py` (2,842 lines), `agent/executors.py` (1,912),
`agent/spatial_agent.py` (1,010), `agent/nodes.py` (946), `agent/planner.py` (748),
`agent/state.py`, `utils/optimization.py`, `utils/logging_*.py`, `tools/context_parser.py` and
`tools/local_context_db.py` are untouched, including upstream's own bugs. (One example, so it
is not "fixed" by accident: `operators.py:1110` reads `client.geocode(wp)` as
`{'results': [{'geometry': …}]}`, which is not the shape `geocode` returns on either provider.
That branch was already dead upstream. Leave it.)

`tools/local_context_db.py` is vendored and inert. It looks for `data/context_cache.db`, which
this branch does not ship — upstream's corpus is English MapEval-Textual and has nothing to say
about Seoul — so `ContextManager` disables itself at startup and every lookup goes to Kakao.
That is the behaviour we want, reached through upstream's own code path rather than by deleting it.

---

## 2. MapEval-API: what was changed

`Evaluator2.py`, `LLM.py` and `BenchmarkDataset.py` are vendored **unmodified**, as the
reference the adapter is checked against. They do not run: `src/evaluator.py` is the harness.

`FormattedTools.py` is the port. Kept verbatim: the five tool classes, their `name`,
`description` and `args_schema`, the place-id threading between them, and the four
`*_to_context` formatters that decide what an observation says. Replaced: every `_run` body,
which was an HTTP GET to `http://localhost:5000/api` (the MapQaTor backend proxying Google).

Deleted, with reasons:

- `Tools.py` — the unformatted tool variants, plus a `PlaceIdTool` `Evaluator2.py` never
  constructs. Nothing imports it, and every method in it called the deleted backend.
- `types.json` — Google's place-type list, used to decide `type=` vs `keyword=`.
  `TYPE_VOCABULARY` in `src/kakao_maps.py` answers the same question in Kakao's vocabulary.
- the nine `GPT4.py` / `Claude.py` / … model wrappers and `main.py` — one `ChatOpenAI` built
  from `LLM_*` settings replaces them, so both architectures decode identically.

Changed behaviour in `FormattedTools.py`, all three forced by Kakao:

| | upstream | here | why |
| --- | --- | --- | --- |
| `time.sleep(30)` after every successful call | 30 s | `MAPEVAL_TOOL_SLEEP_SECONDS`, default 0 | The sleep was for MapQaTor's hosted rate limit. Kakao served directly with a local cache has none, and 30 s a call adds hours to a run without changing an answer. |
| `travelMode` | driving / walking / bicycling / transit | driving only; anything else is refused in words | Kakao Mobility has no public walking, cycling or transit directions API. Answering a walking question with a driving route answers a different question. |
| a failed call | every exception swallowed into "Incorrect place name / place id" | same, except `KakaoAuthError` and `KakaoRateLimitError`, which propagate | Against a rejected key, swallowing turns a whole run into confidently unanswerable questions instead of one loud failure. A quota exhaustion will not fix itself inside the question either. |

One behaviour was **added** to `NearbyPlacesTool`: the anchor is dropped from its own
neighbour list. It stands at zero metres from itself and would head every ranking it appears
in, and a nearest-cafe question asked from a cafe lists that cafe. Google's Nearby Search
excludes the centre; Kakao's category search does not.

---

## 3. The Kakao client

`src/kakao_maps.py` is the only hand-written module of consequence. Every public method keeps
its Google counterpart's name, argument names, argument types and return shape.

Both architectures read this one client. Two architectures over one evidence source is the
whole premise: a difference between them cannot be a difference in what they were shown.

### Method by method

| method | Google | Kakao | note |
| --- | --- | --- | --- |
| `geocode` | Geocoding API | `/search/address.json`, then `/search/keyword.json` | Kakao splits addresses and POI names across two indexes. A Korean question names a POI far more often than an address, so a geocode that stopped at the address index could not resolve most of the benchmark. Upstream's `location_bias` selection and its 100 km / 200 km rejection thresholds are kept unchanged. |
| `reverse_geocode` | Geocoding API | `/geo/coord2address.json` | Road address preferred, jibun as fallback. |
| `nearby_search` | Places Nearby | `/search/category.json` or `/search/keyword.json` | See "the type vocabulary" below. |
| `text_search` | Places Text Search | `/search/keyword.json` | |
| `get_place_details` | Place Details | the client's own place table | Kakao has **no** details endpoint — a search response already carries everything Kakao knows. Every search writes its places to a SQLite table, and details are served from it. An id this client never issued is unknown, which is the answer Google gave for a malformed one. |
| `get_directions` | Directions | Kakao Mobility `/v1/directions`, or `/v1/waypoints/directions` | Driving only. `optimize_waypoints` is ignored: Kakao visits waypoints in the order given. |
| `get_distance_matrix` | Distance Matrix | one `/v1/directions` call per off-diagonal pair | Kakao has no matrix endpoint of that shape. See "the diagonal" below. |
| `get_timezone` | Time Zone API | answered locally | Kakao's coverage is Korea, which is `Asia/Seoul` at UTC+9 and has observed no DST since 1988. Zero API calls; a fact about the coverage area, not a value invented to fill a gap. |

### Fields Kakao does not publish

Kakao Local carries no **rating**, no **review count**, no **price level** and no **opening
hours**. They are not absent from the return shape — the keys stay, so the vendored formatters
keep working — but they carry `None` (or `0` in `nearby_search`, which is what upstream's own
`place.get('rating', 0)` produced for a place without one, and which keeps its sort behaving).

The consequence that matters: **`min_rating` and `open_now` are accepted and ignored.**
Upstream's `nearby_search` drops any candidate with `rating < min_rating`; against an all-zero
rating that deletes every candidate, and an empty list is what a generation stage guesses over.
A source that publishes no ratings is not a source in which every place is unrated.

This also means **MapEval's whole attribute half has no counterpart here**. Questions about a
rating, a price level, whether a place is open on a Monday, or whether it serves lunch are
unanswerable by nature on Kakao — not hard, unanswerable. Do not read a low score on such a
family as an architecture result.

### The ranking

Upstream sorted nearby results by `(rating, user_ratings_total)` descending. Both are constant
under Kakao, and Python's sort is stable, so the line is a no-op that preserves Kakao's own
distance ordering — which is the ranking a nearby search should return. The line is kept rather
than deleted so the ported code still reads against its original.

### The type vocabulary

Google took `type` and `keyword` side by side on one endpoint. Kakao takes a category group
code on one endpoint and a query on another, so `TYPE_VOCABULARY` maps each requested type onto
whichever of the two can express it:

- a Google place type, because the upstream prompts were written against Google and a vendored
  planner still emits `convenience_store`;
- the Korean noun a Korean question asks by, so a planner that copies the question's own word
  reaches the same retrieval;
- a bare Kakao code (`CS2`), because a planner that has seen one will write it.

A type Kakao has no code for is asked for by name. When a narrowing keyword inside a code finds
nothing, the coarser category retrieval answers instead: an empty result from a vocabulary gap
is not evidence that the neighbourhood holds no such place.

### The diagonal

`get_distance_matrix` fills the diagonal with zero **locally**, without an API call. Kakao
refuses a leg whose endpoints are within 5 m of each other
("출발지와 도착지가 5 m 이내로 설정된 경우 경로를 탐색할 수 없음"), and a trip matrix asks for its own
diagonal on every run. It is the only leg that may be filled: a missing off-diagonal leg is
reported as `ZERO_RESULTS`, never as a free hop, because an absent leg is missing evidence.

### Route text

`format_distance` and `format_duration` reproduce Google's phrasing ("5.3 km", "15 mins",
"1 hour 5 mins") because the vendored formatters print those strings verbatim and
`tools/context_parser.py` matches them with regexes.

`overview_polyline` is `None`: Kakao returns route vertices, not an encoded polyline. Nothing
in either agent reads it.

### The region prior

`KAKAO_SEARCH_CENTER` / `KAKAO_SEARCH_RADIUS_M` bias the **first** keyword query toward the
benchmark's region. Korean POI names repeat across cities, so a nationwide keyword search
resolves a bare brand name to whichever branch has the shortest name, anywhere in the country.
A name with no match in the region still falls back to the unbiased nationwide search, so the
prior can never hide a place.

It is deployment configuration, not evidence: it applies to whichever agent queries, it reads
nothing from the dataset, and it says where to look first, never which option is right. Blank
disables it, and a report must say which it was. It has no upstream counterpart — Google's
`region` parameter is a ccTLD bias, which Kakao has no use for since it serves one country.

### The cache

`_ResponseCache` is a SQLite cache of Kakao responses plus the place table `get_place_details`
is served from, keyed by method + URL + arguments. It exists for the same reason upstream's
`data/context_cache.db` does: a run re-asks the same lookups and Kakao quota is finite. It has
its own table names (`kakao_responses`, `kakao_places`) so a `data/kakao_cache.db` written by
another implementation cannot be read as this one's.

---

## 4. The harness

`src/evaluator.py`, `src/dataset.py`, `src/logging.py`, `src/metrics.py`, `src/config.py` and
`src/parsing.py` are this repository's, not either upstream's. They replace upstream's two
outer loops:

- `Evaluator2.py`'s loop, which POSTs each verdict to `http://localhost:5000/api/evaluation/`
  and sleeps 60 s between questions.
- `spatial-agent/test_agent.py`.

What that buys: four concurrent workers, question-level retry for failures that are the
endpoint's rather than the agent's, one report format for both architectures, and a per-question
log. What it costs: the runs are not byte-comparable with either upstream's own output.

### Indices

**Upstream MapEval numbers its options from 1** (`Option1: …`, gold stored as `correct + 1`).
**Upstream Spatial-Agent is 0-based** (`spatial_agent.py` line 35: "predicted_option is
zero-based"). This repository is 0-based everywhere.

So `src/agent/react.py` builds upstream's one-based prompt verbatim — changing it would change
the task — and converts once, on the way out. `src/agent/spatial.py` converts nothing. Both are
pinned by tests.

### `Option0: Unanswerable`

Upstream prepends it on rows whose classification is None, which announces the answer before
the question is read. It is not prepended here; the benchmarks carry their refusal as an
ordinary option. `^^0^^` therefore has no index and is recorded as `answer_parse_failure`,
which is upstream's own `verdict: invalid`.

### Intent

Upstream Spatial-Agent routes to four intents (`nearby`, `routing`, `trip`, `poi`); this
repository's benchmarks classify eight ways. An intent outside upstream's four is recorded as
`None` — a vocabulary the architecture does not have, not a miss. The ReAct baseline has no
classification stage at all and always records `None`; the intent metric counts only questions
an intent was predicted for.

### Failure classification, and where it is weaker than on `main`

`process_question` catches every exception and reports `str(e)`, which drops the class name. So
`src/agent/spatial.py::classify_error` recovers the marker from the text and re-prefixes the
message, because `is_transient_failure` decides retryability from the name a message starts with.

**Known limitation:** because the vendored agent swallows exceptions, an LLM outage on the
Spatial-Agent side cannot be told apart from a reasoning failure at that boundary. It is
absorbed one level down, by the retry budget on the `ChatOpenAI` it is built with, and a run's
`llm_unavailable_count` will under-report on that side. The ReAct baseline has no such gap:
`src/agent/react.py` raises `LLMUnavailableError` out of `answer`.

### Environment bridging

The vendored code reads `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` and
`KAKAO_REST_API_KEY` from the environment at construction time.
`main.export_upstream_environment` exports this repository's `LLM_*` / `KAKAO_*` settings under
those names. That is how the vendored files stay unmodified.

`SpatialAgent.__init__` then replaces the client and the model the vendored agent built for
itself, so each worker's counters are its own and both architectures decode identically. Note
that `operators.py` constructs a bare `KakaoMapsClient()` in two fallback paths (lines 310 and
620); those calls work — they read the key from the environment — but their API calls are not
counted in that question's totals.

---

## 5. Offline dataset tooling

`data/_toolkit/` is the `main` branch's `src/tools/` and `src/models.py`, moved so the dataset
builders and verifiers keep working. **Nothing in `src/` imports it**, and no benchmark run may
depend on it having been executed.

It is a *second* Kakao implementation, and that is only acceptable because it never touches an
agent: it is what built and verified the datasets in `dataset/`, and those artefacts are what
this branch consumes. A gold answer this branch's agents disagree with is not automatically the
agent's fault — the two Kakao paths normalize differently — so re-verify against `main` before
treating a verifier disagreement as an agent result.

---

## 6. What this branch cannot answer

- **Anything about ratings, price levels or opening hours.** Kakao publishes none. MapEval's
  attribute half has no counterpart here.
- **Anything about walking, cycling or transit routes.** Kakao Mobility routes cars only.
- **Anything outside Korea.** Kakao Local's coverage is Korea, and `get_timezone` assumes it.
- **A byte-comparable reproduction of either upstream's reported numbers.** Different map,
  different country, different questions, different harness, and the prompting-only path — no
  SFT, no DPO, no embedding retrieval. Report it as prompting-only, on Kakao, and never present
  it as reproducing a headline number.
