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
