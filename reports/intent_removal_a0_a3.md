# Intent-removal measurement (A0-A3)

Measured on 2026-08-26 with Spatial-Agent, `temperature=0`, `max_reasoning_steps=15`,
`concurrency=32`, model `google/gemma-4-E4B-it-qat-w4a16-ct`, Kakao evidence, and the same
configured cache database. The first sandboxed A3 attempt ended before any LLM call and is not a
measurement. All rows below come from successful API runs.

## Main-set sequence

The fixed dataset is `dataset/seoul_kmapeval_v7_mcq_300.jsonl` (283 rows). Accuracy is pooled over
three passes; the pass column retains endpoint variation. A repair occurrence is a question with
four LLM calls instead of the normal analysis/planner/evaluator three. Every such question in
these runs had exactly one extra call.

| stage | revision | pass accuracy (%) | pooled accuracy (%) | repair occurrence | terminal failures |
|---|---|---:|---:|---:|---:|
| A0: baseline | `643fe24` | 82.0 / 84.5 / 85.5 | **84.0** (713/849) | 111/849 (13.1%) | 3 `agent_reasoning_failure` |
| A1: operator-implied gates removed | `b2bac77` | replayed A0 graphs | **identical by replay** | identical by replay | identical by replay |
| A2: fact/structure grounding | `c07b998` | 82.3 / 84.5 / 86.6 | **84.5** (717/849) | 122/849 (14.4%) | 3 `agent_reasoning_failure` |
| A3: planner/evaluator routing removed | `af51e93` | 81.6 / 81.3 / 83.4 | **82.1** (697/849) | 141/849 (16.6%) | 8 `agent_reasoning_failure` |

A1 is not an independent stochastic API rerun. Replay over all 2,577 recorded A0 planner graphs
found zero grounded-graph differences, so the downstream result is exactly inherited for those
graphs. A2 changed 293/849 grounded graphs in its live passes. A3 changes template selection
before planning and therefore required the live run shown above.

### MapEval task category

`unanswerable` remains a separate port-added row and is not folded into the paper's four-category
mean.

| `mapeval_class` | A0 (%) | A2 (%) | A3 (%) | A3 - A2 (points) |
|---|---:|---:|---:|---:|
| nearby | 80.6 | 80.2 | 78.6 | -1.6 |
| poi | 89.9 | 89.1 | 94.2 | +5.1 |
| routing | 90.9 | 89.4 | 88.9 | -0.5 |
| trip | 81.8 | 85.9 | 74.2 | -11.6 |
| unanswerable | 69.8 | 71.4 | 73.0 | +1.6 |

### Measurement type

| `classification` | A0 (%) | A2 (%) | A3 (%) | A3 - A2 (points) |
|---|---:|---:|---:|---:|
| distance | 89.9 | 89.1 | 94.2 | +5.1 |
| nearby | 78.9 | 79.2 | 79.6 | +0.4 |
| radius | 75.0 | 72.2 | 61.1 | -11.1 |
| routing | 90.9 | 89.4 | 88.9 | -0.5 |
| trip | 81.8 | 85.9 | 74.2 | -11.6 |

### Generator-template footprint

The largest A3-A2 movements on the main set were:

| `template_id` | A2 (%) | A3 (%) | delta (points) |
|---|---:|---:|---:|
| `nearby_cuisine_subtype` | 55.6 | 83.3 | +27.8 |
| `trip_optimal_order` | 80.6 | 56.9 | -23.6 |
| `nearby_kth_nearest` | 88.9 | 75.0 | -13.9 |
| `trip_feasible_count_five` | 85.7 | 73.0 | -12.7 |
| `nearby_within_radius_count` | 72.2 | 61.1 | -11.1 |
| `routing_nth_turn` | 98.4 | 90.5 | -7.9 |
| `poi_distance_difference` | 88.9 | 94.9 | +6.1 |
| `routing_turn_count_via` | 79.4 | 84.1 | +4.8 |
| `trip_total_distance` | 92.1 | 95.2 | +3.2 |

Small `unanswerable_*` families are omitted from this delta table: a single row is 11-50 points
depending on the family. Full template splits remain in every JSON report.

The A3-A2 paired question-cluster difference is -2.36 points with an approximate 95% interval of
[-5.82, +1.11]. Thus these three passes do not show a statistically distinguishable overall
change, but they also do not establish equivalence under a predeclared narrow margin. The main
diagnostic is the family split: fixed-order trip totals held at 95.2%, while optimal-order and
feasible-count families declined. Since the endpoint is known to move families sharply between
draws, this is evidence to investigate on a fresh draw, not a license to tune against these spent
rows.

## A3 on the three previously named holdouts

These are A3 robustness levels, not fresh held-out claims: all three datasets had already been run
and used in project history before this change. There are no same-revision A0/A2 runs for these
three sets, so they must not be presented as controlled A3 deltas.

| dataset | pass accuracy (%) | pooled accuracy (%) | spread (points) | repair occurrence | terminal failures |
|---|---:|---:|---:|---:|---:|
| v7h | 84.0 / 87.0 / 80.0 | **83.7** | 7.0 | 50/300 (16.7%) | 3 `agent_reasoning_failure` |
| v7h2 | 75.0 / 86.0 / 85.0 | **82.0** | 11.0 | 44/300 (14.7%) | 3 `agent_reasoning_failure` |
| v7h3 | 86.0 / 83.0 / 83.0 | **84.0** | 3.0 | 47/300 (15.7%) | 2 `agent_reasoning_failure` |

## Source reports

- A0: `test_20260826T044359Z.json`, `test_20260826T045342Z.json`,
  `test_20260826T050332Z.json`
- A2: `test_20260826T065105Z.json`, `test_20260826T070054Z.json`,
  `test_20260826T071102Z.json`
- A3 v7: `test_20260826T090918Z.json`, `test_20260826T091849Z.json`,
  `test_20260826T092857Z.json`
- A3 v7h: `test_20260826T093256Z.json`, `test_20260826T093634Z.json`,
  `test_20260826T094030Z.json`
- A3 v7h2: `test_20260826T094434Z.json`, `test_20260826T094820Z.json`,
  `test_20260826T095158Z.json`
- A3 v7h3: `test_20260826T095606Z.json`, `test_20260826T100028Z.json`,
  `test_20260826T100410Z.json`
