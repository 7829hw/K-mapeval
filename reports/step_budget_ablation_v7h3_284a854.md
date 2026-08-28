# Step-budget ablation, replicated on the held-out set

Second run of the budget question, at `284a854`, on a different set from the
first. `reports/step_budget_ablation_354e3dd.md` is the first.

Nothing under `src/` changed for this. `src/config.py` and `.env` keep 15; both
arms ran under an environment override and their reports carry
`metadata.max_reasoning_steps`.

## Conditions

Spatial-Agent, `dataset/seoul_kmapeval_v7h3_holdout_100.jsonl`, 100 questions,
three passes per arm, concurrency 32, `google/gemma-4-E4B-it-qat-w4a16-ct` at
temperature 0, provider `kakao`.

## Result

| | budget 15 | budget 30 |
|---|---|---|
| pass 1 / 2 / 3 | 81 / 73 / 73 | 81 / 68 / 69 |
| **mean** | **75.7** | **72.7** |
| spread | 8 | 13 |
| over-budget refusals | 9 | **0** |
| `graph_validation_failure` | 19 | 12 |
| `answer_parse_failure` | 37 | **49** |

By family, 3 passes a side:

| family | rows | budget 15 | budget 30 |
|---|---|---|---|
| `trip_feasible_count_five` | 21 | 47.6% | **81.0%** |
| `unanswerable_price_level` | 3 | 66.7% | 100.0% |
| `poi_farthest_of_three` | 30 | 86.7% | 100.0% |
| `nearby_kth_nearest` | 24 | 87.5% | 95.8% |
| `trip_optimal_order` | 24 | 62.5% | 66.7% |
| `nearby_subtype_kth` | 30 | 100.0% | 83.3% |
| `routing_nth_turn` | 21 | 81.0% | 61.9% |
| `nearby_within_radius_count` | 12 | 58.3% | 33.3% |
| `routing_turn_count_via` | 21 | 66.7% | 38.1% |
| `trip_total_distance` | 21 | 76.2% | 61.9% |
| `routing_detour_cost` | 24 | 62.5% | 50.0% |

## Reading

The budget binds one family and costs the rest. Raising it removes every
over-budget refusal -- 9 to 0, all of them `trip_feasible_count_five` -- and
lifts that family 33.4 points. Overall accuracy falls 3.0.

The mechanism is visible in the failure counts: `graph_validation_failure` drops
19 to 12 while `answer_parse_failure` rises 37 to 49. Given room, the planner
writes larger graphs; larger graphs reach execution and then go wrong in more
ways than a refused one can. Four families lose 17 to 29 points that way, and
none of them is over budget at 15.

This replicates `reports/step_budget_ablation_354e3dd.md` on a different set and
a later revision: there the family moved 28.6% -> 66.7% and the overall fell 2.6
(69.3 -> 66.7); here 47.6% -> 81.0% and 3.0 (75.7 -> 72.7). Two independent
draws, same shape.

So `trip_feasible_count_five` *is* budget-bound, and the budget is still not the
repair. It is one budget shared by both architectures -- `AGENTS.md` records
budget 30 moving ReAct 39.0 -> 47.3 and Spatial-Agent 69.0 -> 74.0 in the same
pass -- so a number measured at 30 is not a controlled comparison of either, and
15 is langchain's default and therefore the reference baseline's. The default
stays 15.

What the family needs instead is the compact graph the vocabulary already has:
one ROUTE_MATRIX over the stops and `tsp_tw` with service times and a budget,
which is 4 edges rather than 17 to 25. Retrieval now offers `ROUTE-OPTIMIZE` for
these questions and the prompt states the convention; this model still unrolls
the trip leg by leg. That is the planner's ceiling at this budget, and it is
what the 9 refusals are.
