# Concept Transformation stage (items 3+4+5): one architectural ablation

Measured 2026-08-26 on `dataset/seoul_kmapeval_v7_mcq_300.jsonl` (283 rows), Spatial-Agent only,
`temperature=0`, `max_reasoning_steps=15`, `concurrency=32`, model
`google/gemma-4-E4B-it-qat-w4a16-ct`, Kakao evidence, same cache database. ReAct is untouched by
this change and was not run.

**Baseline is `af51e93`: 82.1% pooled over three passes** (81.6 / 81.3 / 83.4), from
`intent_removal_a0_a3.md`.

## The intervention

`GRAPH_PROMPT` stopped naming operators. The planner now emits one of eighteen semantic
transformations per node -- RESOLVE_PLACES, PLACE_SEARCH, DISTANCE_MEASURE, ROUTE_OPTIMIZE, SORT,
ORDINAL_SELECT, AGGREGATE, MATCH_OPTIONS … -- and `src/agent/semantics.py` maps each onto an
executable operator deterministically, from the concept types, the transformation and the facts
the analysis extracted. It never sees the question. No extra LLM call: the split is between
semantics and implementation, not between two planners.

`Retrieve-Rank-Ordinal` is deleted. The precondition was that the factorizer reproduce it
compositionally, and it does: lifting all eleven macro-template examples into transformations and
factorizing them back returns the operators they started from, the ordinal template included, as
`RESOLVE_PLACES -> PLACE_SEARCH -> SORT -> ORDINAL_SELECT -> MATCH_OPTIONS` with `ordinal=2`.
`ordinal=1` is the superlative -- the same graph, one factor apart. The catalogue is Appendix E's
ten and nothing else.

## By stage

| | `af51e93` | run 1 `41c29bf` | run 2 `1506362` | run 3 `8f67f01` |
|---|---:|---:|---:|---:|
| planner nodes naming an operator | n/a | 0 / 1,384 | 0 / 1,563 | **0 / 1,548** |
| questions composed wholly in semantics | n/a | 235 / 283 | 273 / 283 | 269 / 283 |
| graph-generation failures | 8 (3 passes) | **47** | 10 | 14 |
| execution errors (steps / questions) | 240 / — | — | 446 / 165 | 389 / 144 |
| **accuracy** | **82.1** (3 passes) | 40.3 | 48.4 | **50.9** (1 pass) |

Runs 1 and 2 were stopped after one pass rather than confirming a broken configuration over
three. Run 3 is one pass, so its spread is unmeasured; the gap is far larger than any spread this
endpoint shows.

## By family, run 3 against A3

| family | A3 | run 3 | delta |
|---|---:|---:|---:|
| `routing_turn_count_via` | 84.1 | 23.8 | **-60.3** |
| `nearby_within_radius_count` | 61.1 | 8.3 | **-52.8** |
| `trip_total_distance` | 95.2 | 42.9 | **-52.3** |
| `nearby_cuisine_subtype` | 83.3 | 44.4 | -38.9 |
| `trip_feasible_count_five` | 73.0 | 38.1 | -34.9 |
| `poi_distance_difference` | 94.9 | 60.6 | -34.3 |
| `nearby_kth_nearest` | 75.0 | 41.7 | -33.3 |
| `routing_nth_turn` | 90.5 | 76.2 | -14.3 |
| `trip_optimal_order` | 56.9 | 54.2 | -2.7 |

Nothing improved. The damage is broad rather than concentrated, which is what rules out a single
remaining defect as the explanation.

## What the ablation establishes

**The semantic split itself is not the problem, and it works.** The planner answered in
transformations on the very first pass and never once reached for an operator -- 0 of 1,548 nodes
across the final run. Operator selection moved into deterministic code cleanly: after six wiring
fixes, 311 of 319 recorded graphs validate (97.5%), against 47 outright graph-generation failures
before them.

**Six defects were found offline, without a benchmark pass.** The semantic graphs a run records
are enough to replay factorization, grounding and validation, so each was diagnosed and fixed for
free. The largest: 122 of 642 `RESOLVE_PLACES` nodes named a concept the analysis did not have,
and 190 graphs measured an anchor against a candidate *set* that was wired as a pair -- measuring
the first candidate and discarding the rest. Every one has a test naming the count that found it.

**The regression is the 163 deleted prompt lines, not the architecture.** Those lines were not
operator documentation. They were guidance about *what shape of graph a question needs*, and it
was worth a great deal. The clearest case is measured: the Analysis stage lists the four candidate
texts as concepts, so `RESOLVE_PLACES` over them looks like the obvious first node, and the graph
then answers "which of these four is closest" rather than "what is the k-th closest bank". The old
prompt spent five emphatic lines preventing exactly that; restoring one paragraph of it in
semantic form moved `nearby_kth_nearest` 25.0 -> 41.7 and `nearby_cuisine_subtype` 16.7 -> 44.4 in
a single pass. That is the mechanism, confirmed -- and roughly 31 points of it remain unrecovered.

So the finding is a decomposition of what that prompt was doing. Operator selection is mechanical,
belongs in deterministic code, and provably survives the move. Graph *shape* selection is
semantic, has no home in the factorizer, and must be rebuilt in the prompt rather than assumed
redundant. Rebuilding it is the remaining work, and each attempt costs a benchmark run.

## Not to be read as

Not a held-out number: v7-283 has been tuned against throughout this project. Not a three-pass
result: run 3 is one pass. Not a statement about ReAct, which this change does not touch.


---

# Recovery iteration: skeletons, then Concept Analysis

Same dataset, same conditions. All runs one pass, so no spread is claimed; the movements below
are larger than any spread this endpoint has shown on 283 rows.

## The required ablation

**A** = the semantic planner as it stood (`8f67f01`). **B** = A plus retrieved semantic
macro-template skeletons (`ecb9400`). **C** = B plus the Concept Analysis completion
(`689734d`), which the acceptance criteria redirected to when B did not recover.

| metric | A `8f67f01` | B `ecb9400` | C `689734d` | `af51e93` |
|---|---:|---:|---:|---:|
| concrete operator leakage | 0 / 1,548 | 0 / 1,640 | **0 / 1,678** | n/a |
| semantic-vocabulary ratio | 100% | 100% | **100%** | n/a |
| questions composed wholly in semantics | 269 / 283 | 275 / 283 | **277 / 283** | n/a |
| graph-generation success | 95.1% | 97.5% | **97.9%** | 99.1% |
| G1-G5 validation success | 95.1% | 97.5% | **97.9%** | 99.1% |
| execution success (no errored step) | 49.1% | 47.7% | **54.8%** | — |
| **final accuracy** | 50.9% | 50.5% | **60.4%** | **82.1%** (3 passes) |

Graph-generation and validation success are one number here: a question that fails validation
after the draft and the repair is exactly the one that produces no graph, and it is reported as
`graph_validation_failure`.

### Acceptance criteria

* 0% concrete operator leakage -- **met**, 0 of 1,678 nodes, and no compound operator name or
  category code appears in the prompt, the patterns or the skeletons.
* approximately >=95% graph validation -- **met**, 97.9%.
* clear accuracy recovery from 50.9% -- **met**, 60.4%.

## Per family, against `af51e93`

| family | `af51e93` | C | delta |
|---|---:|---:|---:|
| `nearby_subtype_kth` | — | 100.0 | — |
| `nearby_kth_nearest` | 75.0 | 75.0 | **+0.0** |
| `trip_optimal_order` | 56.9 | 62.5 | **+5.6** |
| `trip_feasible_count_five` | 73.0 | 66.7 | -6.3 |
| `routing_nth_turn` | 90.5 | 71.4 | -19.1 |
| `poi_distance_difference` | 94.9 | 60.6 | -34.3 |
| `trip_total_distance` | 95.2 | 52.4 | -42.8 |
| `nearby_within_radius_count` | 61.1 | 16.7 | -44.4 |
| `nearby_cuisine_subtype` | 83.3 | 33.3 | -50.0 |
| `routing_turn_count_via` | 84.1 | 14.3 | -69.8 |

`nearby_kth_nearest` reaching parity is the one that matters most: it is the family the whole
diagnosis was built on, it was at 41.7% two iterations ago, and the ordinal is now a factor on a
composed shape rather than a template of its own.

## What B established, and why it did not show in the accuracy

B is not a failure. It fixed the thing it was aimed at -- validation rose to 97.5%, and
`nearby_kth_nearest` stopped ranking the four answer texts -- but the accuracy did not move,
because a second constraint was binding underneath it.

## What C established

Concept Analysis quality, measured over the same benchmark rows at both revisions:

| | `af51e93` (82.1%) | `ecb9400` (50.5%) |
|---|---:|---:|
| no usable concepts at all | 24% | 19% |
| names a kind of place, `target_type` null | 43% | 44% |
| candidate options present as concepts | 8% | 9% |

Identical. The Analysis stage was not damaged by any of this work; it has always been this weak.
What changed is that the old architecture did not depend on it -- the planner copied place names
straight out of the question, so the concept graph was decoration for the role checks. The
semantic architecture routes place identity *through* the concept graph, so a fifth of questions
went from "thin analysis" to "no place to resolve", and the fallback handed `RESOLVE_PLACES` the
entire question text as one concept, typed as a place, to geocode.

Completing that fallback from the facts the deterministic extractors had already read off the
same question -- the anchor, the kind, the stops, the compared pair -- is worth **+9.9 points**,
and 312 of 312 recorded fallback analyses now geocode places rather than a sentence.

## The remaining 22 points, diagnosed

Three families carry most of it, and none of them is a factorizer question:

* `routing_turn_count_via` (-69.8, 30 errored steps). A via-route has three places and
  `ROUTE_MEASURE` wires only an origin and a destination, so the waypoint is dropped and the turn
  count is counted on a different route. The vocabulary has no way to say "through here".
* `trip_total_distance` (-42.8, **2** errored steps). The graph runs and computes the wrong
  number: `AGGREGATE(scope=groups)` over a route matrix has no groups wired, so it totals the
  matrix rather than the consecutive legs the itinerary visits.
* `nearby_cuisine_subtype` (-50.0) and `nearby_within_radius_count` (-44.4). Both retrieve and
  then narrow, and both still error on a third of their steps.

The first two are missing semantic distinctions -- a via point, and which legs an aggregate
covers -- which is where the acceptance criteria point next. Neither is fixed by changing how
operators are chosen.
