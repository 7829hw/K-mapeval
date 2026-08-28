# Step-budget ablation at `354e3dd`

`MAX_REASONING_STEPS=30` against the default 15. Requested during the
concept-geoflow regression work, run as an ablation and recorded here rather
than shipped: `AGENTS.md` fixes the default at 15 because that is langchain's
own default and therefore the reference baseline's, and the budget is *one*
budget shared by both architectures, so a run at 30 is not a controlled
comparison of either.

Nothing under `src/` changed for this. The default in `src/config.py` and `.env`
is untouched; the ablation ran under an environment override and its reports
carry `metadata.max_reasoning_steps: 30`.

## Conditions

Spatial-Agent, `dataset/seoul_kmapeval_v7a_mcq_100.jsonl`, 100 questions, three
passes, concurrency 32, `google/gemma-4-E4B-it-qat-w4a16-ct` at temperature 0,
provider `kakao`.

## Result

| | budget 15 | budget 30 |
|---|---|---|
| pass 1 / 2 / 3 | 67 / 75 / 66 | 68 / 68 / 64 |
| **mean** | **69.3** | **66.7** |
| spread | 9 | 4 |
| `graph_validation_failure` | 14 | 8 |
| `answer_parse_failure` | 14 | 19 |

**Overall accuracy does not move.** The 2.6-point difference sits inside both
runs' spreads.

By family:

| family | budget 15 | budget 30 |
|---|---|---|
| `trip_feasible_count_five` | 28.6% | **66.7%** |
| `unanswerable_price_level` | 66.7% | 100.0% |
| `nearby_kth_nearest` | 83.3% | 87.5% |
| `poi_farthest_of_three` | 90.0% | 93.3% |
| `nearby_within_radius_count` | 50.0% | 25.0% |
| `trip_optimal_order` | 45.8% | 29.2% |
| `trip_total_distance` | 71.4% | 57.1% |
| `unanswerable_opening_hours` | 100.0% | 66.7% |

## Reading

The budget binds exactly one family. `trip_feasible_count_five` is a five-stop
itinerary whose graph needs more than fifteen transformations, and eleven of the
budget-15 run's fourteen `graph_validation_failure`s were that family exceeding
it; at 30 they stop and the family gains 38 points. Every other movement is a
12-to-24-row family swinging the way those families swing here — the same set
read `nearby_within_radius_count` at 8.3, 58.3, 16.7 and 50.0 across four
consecutive measurements at a fixed budget.

So raising the budget is not a repair for what remains of the regression. It
buys one family, costs the comparability of every number it touches, and leaves
the overall accuracy where it was. The default stays 15, and this file is the
report `AGENTS.md` requires for having run it at all.

This also reproduces, on a different set and a different revision, what
`AGENTS.md` already records about budget 30: it binds one of the trip families
and not the other, and it moves both architectures at once.
