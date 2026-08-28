# 최종 검증 기록 — `876c772` (evidence only)

작성일 2026-08-28. 이 문서는 검증 결과 기록이며, `src/` 아래 어떤 동작도 이 문서를 위해
바뀌지 않았다. 새 벤치마크 실행은 하지 않았다 — 이미 기록된 실행이 치명적 회귀를 보여
주었고, 같은 조건의 재실행에 LLM/Kakao 할당량을 쓰기 전에 회귀를 먼저 확정하는 쪽을
택했다.

## 1. 검증 대상

| 항목 | 값 |
|---|---|
| 커밋 | `876c772f1c9f` "refactor spatial agent around concept geoflow" |
| 작업 트리 | 깨끗함. 검증 직전 `git reset --hard 876c772`로 미커밋 변경 22파일 / +1783줄(신규 `src/agent/canonicalization.py` 포함)을 폐기 |
| 추적되지 않는 잔여 파일 | `dataset/seoul_kmapeval_v7a_mcq_100.jsonl` (벤치마크 입력, 의도적으로 보존) |

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| Python | 3.12.3 (`.venv`) |
| OS | Linux 6.8.0-137-generic x86_64, glibc 2.39 |
| LLM 모델 | `google/gemma-4-E4B-it-qat-w4a16-ct` |
| LLM 엔드포인트 | `https://hpclabllmapi3.duckdns.org/v1` (OpenAI 호환 vLLM) |
| `llm_temperature` | 0.0 |
| `max_reasoning_steps` | 15 |
| `react_parallel_tool_calls` / `react_forces_final_answer` | false / false |
| provider | `kakao` |
| concurrency | 32 |
| `max_tokens` | 미전송 (배포 측 상한) |

## 3. 정적 검증 — 전부 통과

| 검사 | 명령 | 결과 |
|---|---|---|
| 단위/회귀 테스트 | `pytest` | **712 passed**, 0 failed (4.30s) |
| 린트 | `ruff check .` | **All checks passed** |
| 데이터셋 감사 | `python data/audit_dataset.py dataset/seoul_kmapeval_v7a_mcq_100.jsonl` | **clean** (exit 0) |

`nearby_kth_nearest` k-균형을 포함해 감사 규칙 전부를 통과하는 100문항 세트다.

## 4. 벤치마크 — 기록된 실행의 재귀속

신규 실행 없음. 이 커밋에 귀속되는 실행은 `reports/test_20260827T145235Z.json` 하나다.

귀속 근거: 리포트의 `code_revision`은 `.git/HEAD`만 읽으므로 작업 트리의 청결 여부를
기록하지 않는다. 대신 트레이스 어휘로 확인했다 — 해당 실행의 로그 100건은
`ANALYZE / RETRIEVE_TEMPLATES / COMPOSE / REPAIR / VALIDATE / …` 로, `876c772`의
`trace.append` 집합과 정확히 일치한다. 이후 폐기된 작업 트리의 실행 로그는
`DETERMINISTIC_REPAIR`를 가지고 `REPAIR`가 없어 명확히 구분된다.

### Spatial-Agent, `seoul_kmapeval_v7a_mcq_100.jsonl`, 100문항, 1패스

| 지표 | 값 |
|---|---|
| **전체 정확도** | **14/100 = 14.0%** |
| 평균 소요 | 51.75 s/문항 |
| 평균 LLM 호출 | 3.06 |
| 총 토큰 | 1,219,851 |

`mapeval_class`별:

| class | correct/total | accuracy |
|---|---|---|
| nearby | 14/28 | 50.0% |
| poi | 0/21 | **0.0%** |
| routing | 0/22 | **0.0%** |
| trip | 0/22 | **0.0%** |
| unanswerable | 0/7 | **0.0%** |

정답이 나온 template은 `nearby_kth_nearest`(5), `nearby_cuisine_subtype`(5),
`nearby_subtype_kth`(4) 셋뿐이다.

실패 유형:

| failure_type | 건수 |
|---|---|
| `graph_validation_failure` | **74** |
| 정답 | 14 |
| `answer_parse_failure` | 8 |
| `provider_failure` | 2 |
| `llm_output_truncated` | 1 |
| 오답(실패 아님) | 1 |

### 직전 리비전과의 대비

| 리비전 | 데이터셋 | 행 수 | Spatial-Agent |
|---|---|---|---|
| `58c0aad` | `v7_mcq_300` 부분집합 | 99 | 75.8 / 71.7 / 75.8 (평균 **74.4**) |
| `58c0aad` | `v7a_mcq_300` | 282 | **76.95** (1패스) |
| **`876c772`** | `v7a_mcq_100` | 100 | **14.0** (1패스) |

서로 다른 draw이므로 family 단위 비교는 성립하지 않는다. 그러나 100행 기준 재추첨
편차는 약 ±8점이고 관측된 차이는 60점을 넘으므로, 회귀는 draw로 설명되지 않는다.

참고로 같은 시기 ReAct는 `v7a_mcq_300`에서 44.3 / 45.4였다. `876c772`에서의 ReAct 실행
기록은 없다 — 이 커밋은 Spatial-Agent만 건드렸지만, 대조군 수치가 없다는 점은 한계로
남는다.

## 5. 회귀 진단

기록된 100건의 `[VALIDATE]` 트레이스를 집계한 결과:

| 결과 | 건수 |
|---|---|
| 최초 draft에서 valid | **18** |
| draft invalid → repair 후 valid | 7 |
| draft invalid → repair 후에도 invalid (`graph_validation_failure`) | **74** |
| draft invalid, repair 단계가 검증 전에 중단 | 1 |

즉 planner가 만든 그래프 중 **18%만이 첫 시도에 G1–G5를 통과**하고, 최종적으로 실행에
도달한 그래프는 25/100이다. `58c0aad`에서 `graph_validation_failure`는 282행 중 9건
(3.2%)이었다.

최초 draft 실패의 제약별 분포는 G4 52건, G3 19건, G5 4건, G2 1건이다. 실제 메시지를
정규화해 묶으면 지배적인 형태는 두 가지다.

1. **planner가 존재하지 않는 개념/인자를 지칭한다** — `transformation <id> outputs unknown
   concept <id>`(9), `transformation <id> has no available contextual data`(9),
   `reads unknown concept/factor …`(4+). 새 IR에서는 `transformation_edges`가 참조하는 모든
   concept id가 Analysis 단계 출력에 실재해야 하는데, 로컬 소형 모델이 이를 자주 지키지
   못한다.
2. **선언한 출력 타입이 변환의 타입과 맞지 않는다** — `ROUTE_MEASURE produces field, not
   amount, field`(4), `ROUTE_OPTIMIZE produces network, not field`(2) 등.

여기에 검증 시도 횟수가 줄어든 것이 겹친다. `58c0aad`의 `SpatialAgent.answer`는 네 번
시도했다 — strict draft → strict repair → **lenient repair** → **lenient original**.
`876c772`는 두 번(strict draft → strict repair)만 시도하고 포기한다.
`_factorize_validate_plan`에는 `strict_types` 파라미터 자체가 없고 `True`가 하드코딩되어
있다. 이는 `AGENTS.md`가 명시한 불변식과 충돌한다 —

> "Output-type compatibility, role ordering and the statically knowable argument values are
> this port's own rules … so they inform the repair round and are skipped
> (`strict_types=False`) on the last attempt before a question is given up on."

다만 lenient 패스만으로 74건이 회복되지는 않는다. 실패의 대부분은 G3/G4/G5, 즉 논문
자체의 형식 제약이고 이들은 lenient 패스에서도 거절된다(`strict_types=False`가 완화하는
것은 G2 역할 순서와 이 포트 고유의 값 검사뿐이다). 회귀의 1차 원인은 **새 concept/edge
IR에 대한 planner의 준수율 붕괴**이며, lenient 패스 제거는 이를 흡수하던 완충을 함께
없앤 2차 원인이다.

## 6. 알려진 실패 사례 및 한계

1. **`poi` / `routing` / `trip` / `unanswerable` 4개 클래스에서 정확도 0%.** 72/100 문항이
   한 문제도 맞히지 못한다. 동작하는 것은 `nearby` 계열 3개 template뿐이다.
2. **compose/repair 트레이스가 비어 있다.** 100/100 로그에서
   `[COMPOSE] {"stage": "compose", "graph": null}`이다. `trace.append`가
   `plan.get("graph") or plan.get("steps")`를 기록하는데, planner는 이제
   `transformation_edges` IR로 답하기 때문이다. `[REPAIR]`도 동일하게 null이다.
   결과적으로 **planner 그래프가 기록에 남지 않으며**, `AGENTS.md`가 값싼 진단 수단으로
   지정한 `data/replay_grounding.py`가 이 리비전의 로그에 대해 무의미하게 동작한다 —
   실행하면 "replayed 100 graphs"라고 보고하지만 실제로는 100건 모두 빈 그래프를 재생한다.
   변경 전후 비교에서 "0 changed"와 구분되지 않으므로, 이 결함은 진단 도구를 조용히
   무력화한다.
3. **G3 오류 메시지의 어순이 뒤집혀 있다.** `src/agent/validation.py:120` 부근에서
   `produces {transform.output_type}, not {output_types}`로 출력하는데, 앞의 값이 변환의
   선언 타입이고 뒤가 그래프가 선언한 출력 타입이다. 그래서 `produces field, not amount,
   field`처럼 허용 집합 안의 타입이 거절된 것처럼 읽힌다. 검사 로직
   (`output_types <= accepted_outputs`)은 옳다.
4. **테스트 712건이 이 회귀를 전혀 잡지 못한다.** 전부 통과하는 상태에서 벤치마크는
   14%다. 단위 테스트는 손으로 쓴 유효 그래프를 검증하므로, 실제 planner가 유효 그래프를
   *생산할 수 있는가*는 측정하지 않는다.
5. **`876c772`의 ReAct 대조군 수치 없음** — 아키텍처 간 비교는 이 리비전에서 성립하지
   않는다.
6. 이 실행은 1패스다. `AGENTS.md`가 요구하는 3패스 측정이 아니지만, 74%가 검증 단계에서
   죽는 상태에서 패스 수는 판정을 바꾸지 않는다.

## 7. 판정

**`876c772`는 v1.0 후보가 아니다.**

완료 기준이 요구한 "Spatial-Agent가 task-family별 solver가 아니라 일반화된 GeoFlow
reasoning architecture로 구현되어 있음"은 구조적으로는 이 커밋에서 오히려 강화되었다 —
operator 선택이 planner에서 완전히 제거되어 concept 타입/factor/contract로부터 결정론적으로
정해지고, MCQ 정합은 `MCQAdapter`로 GeoFlow 밖으로 분리되었으며, family 라벨은 어떤
런타임 분기에도 도달하지 않는다. 그러나 그 일반화의 대가로 planner가 준수해야 하는 IR이
현재 배포 모델이 지킬 수 없을 만큼 엄격해졌고, 4개 클래스에서 정확도가 0이다.

릴리스 후보로 삼으려면 최소한 (a) planner 그래프가 다시 트레이스에 기록되어
`replay_grounding` 진단이 살아나야 하고, (b) 첫 draft 준수율 18%가 회복되어야 하며,
(c) 3패스 측정과 ReAct 대조군이 같은 리비전에서 확보되어야 한다.
