# K-MapEval — upstream 코드 그대로, API만 Kakao로

MapEval 방식 **ReAct** 베이스라인과 **Spatial-Agent(GeoFlow)** 를 한국어 지도 객관식 문항에서
비교하는 연구용 MVP입니다.

이 브랜치(`upstream-kakao`)의 전제는 하나입니다. **두 에이전트는 원본 구현을 그대로 쓰고,
바뀌는 것은 지도 API뿐입니다.**

| | 원본 | 이 저장소 |
| --- | --- | --- |
| Spatial-Agent | [`ecerybao/Spatial-Agent`](https://github.com/ecerybao/Spatial-Agent) @ `6876bba` | `src/spatial_agent/` — 기계적 rename 외 byte-identical |
| ReAct 베이스라인 | [`MapEval/MapEval-API`](https://github.com/MapEval/MapEval-API) @ `35d481a` | `src/mapeval_api/` — `Evaluator2.py`는 무수정 vendoring, `FormattedTools.py`가 이식분 |
| 지도 API | Google Maps (Places / Directions / Distance Matrix / Geocoding / Time Zone) | Kakao Local + Kakao Mobility (`src/kakao_maps.py`) |

따라서 이 브랜치가 내는 수치는 **원본 아키텍처에 귀속됩니다**. 재구현의 성능이 아닙니다.

> `main` 브랜치는 다른 실험입니다. 자체 tool registry·operator·grounding을 갖춘 처음부터의
> 포팅이고, 두 브랜치의 수치는 서로 다른 질문에 답하므로 **합산하면 안 됩니다.**

원본과의 **모든** 차이는 `docs/UPSTREAM_MAPPING.md`에 기록되어 있습니다. 거기에 없는 차이는
버그입니다.

## 구조

```text
k-mapeval/
├── main.py                  # 단일 실행 진입점
├── src/
│   ├── kakao_maps.py        # Google Maps 클라이언트 형태를 유지한 Kakao 클라이언트 (교체의 전부)
│   ├── spatial_agent/       # ecerybao/Spatial-Agent @ 6876bba (vendoring)
│   │   ├── agent/           #   operators / executors / planner / nodes / state — 무수정
│   │   ├── tools/kakao_maps.py  #   google_maps.py 자리, src/kakao_maps.py 재수출
│   │   └── utils/           #   optimization(TSP-TW) / logging — 무수정
│   ├── mapeval_api/         # MapEval/MapEval-API @ 35d481a
│   │   ├── Evaluator2.py    #   무수정 vendoring (참조용, 실행되지 않음)
│   │   └── FormattedTools.py#   5개 도구 — 포맷터는 그대로, 호출부만 Kakao로
│   ├── agent/               # 위 둘을 Evaluator에 물리는 얇은 어댑터
│   ├── evaluator.py         # 4-worker 동시 실행, 문항 단위 재시도, 보고서
│   ├── config.py / dataset.py / logging.py / metrics.py / parsing.py
├── data/_toolkit/           # 데이터셋 빌더/검증기 전용. src/에서 import 금지
├── dataset/                 # 평가 데이터셋 (main에서 생성·검증됨)
├── tests/                   # Kakao·LLM 모두 stub
├── logs/  reports/          # 자동 생성, gitignored
```

두 에이전트는 **worker마다 하나씩의 동일한 `KakaoMapsClient`** 를 읽습니다. 두 구조 간 정확도
차이가 "본 증거가 달라서" 생기는 일이 없도록 하는 것이 이 배치의 목적입니다.

정답·classification·region·difficulty·verified_at은 에이전트 입력에 포함되지 않습니다.
`BenchmarkItem.agent_input()`은 `(question, options)`만 반환합니다.

## 설치

Python 3.11 이상.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp example.env .env
```

`.env`에 LLM 설정과 `KAKAO_REST_API_KEY`를 넣습니다. Kakao REST API 키 하나를 Local API와
Mobility API에 함께 씁니다. `LLM_BASE_URL`을 비우면 OpenAI 기본 endpoint, 채우면 OpenAI 호환
Chat Completions endpoint를 씁니다.

원본 두 스택이 같이 설치됩니다. MapEval-API는 langchain 0.3의 고전 agent API
(`initialize_agent` + `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`)를, Spatial-Agent는
`langgraph` StateGraph와 `ortools`를 씁니다.

## 실행

```bash
python main.py --agent both                   # dataset/seoul_kmapeval_v4_mcq_100.jsonl
python main.py --agent react
python main.py --agent spatial --concurrency 4
python main.py --agent both --ids seoul_kmapeval_v4_000 seoul_kmapeval_v4_024
python main.py --agent react --verbose-agent  # 원본 Evaluator2와 같은 chain-of-thought 출력
```

실행은 실제 LLM 토큰과 Kakao 쿼터를 씁니다. 문항별 trace는
`logs/<UTC>_id<문항ID>_<질문-slug>.log`, 배치 보고서는 `reports/test_<UTC>.json`
(`metadata` / `statistics` / `results`)에 남습니다. `correct_answer`/`predicted_option`은
모두 0-based이고, 로그에는 API 키가 들어가지 않습니다.

보고서 `metadata`에는 `agent_type`, `llm_model`, `llm_base_url`, `code_revision`과 두 원본
리비전이 기록됩니다.

## 캐시

`src/kakao_maps.py`가 Kakao 응답과, `get_place_details`가 읽는 place 테이블을 SQLite에
저장합니다. Kakao Local에는 **place details endpoint가 없어서** 검색 응답이 곧 그 장소에 대한
Kakao의 전부이고, 그래서 검색이 돌려준 장소를 직접 보관합니다.

```ini
KAKAO_CACHE_DB_PATH=data/kakao_cache.db   # 비우면 캐시 비활성화
KAKAO_CACHE_TTL_SECONDS=86400             # 0이면 만료 없음
```

테이블 이름은 `kakao_responses` / `kakao_places`로, `main` 브랜치가 만든 같은 경로의 DB를
잘못 읽지 않도록 분리되어 있습니다. 평가 결과에는 `cache_hits`, `cache_misses`, 실제
`api_calls`가 따로 기록됩니다.

## 이 브랜치가 측정할 수 없는 것

수치를 아키텍처 결과로 읽기 전에 확인해야 합니다.

- **평점 / 가격대 / 영업시간.** Kakao Local이 제공하지 않습니다. MapEval의 attribute 계열
  문항은 여기서는 어려운 게 아니라 **답이 없습니다**. 그래서 `min_rating`·`open_now`는 받되
  무시합니다 — 적용하면 후보가 전부 삭제되고, 빈 목록은 생성 단계가 추측하는 대상이 됩니다.
- **도보 / 자전거 / 대중교통 경로.** Kakao Mobility는 자동차만 제공하므로, 해당 모드는 자동차
  경로로 답하지 않고 말로 거절합니다.
- **한국 밖.**
- **Spatial-Agent 쪽 LLM 장애.** 원본 `process_question`이 모든 예외를 삼키므로 그쪽
  `llm_unavailable_count`는 과소 집계됩니다. ReAct 쪽에는 이 공백이 없습니다.

SFT+DPO와 임베딩 검색은 구현하지 않은 prompting-only 경로이며, Kakao 데이터로 평가하므로
어느 논문의 headline 수치 재현도 주장하지 않습니다.

## 테스트

Kakao(`httpx.MockTransport`)와 LLM(`FakeListChatModel`)을 모두 stub하므로 키도 네트워크도
필요 없습니다.

```bash
pytest
ruff check .

# vendoring된 트리가 원본에서 벗어나지 않았는지 확인
diff -r ~/spatial-agent/src/agent src/spatial_agent/agent   # rename만 나와야 함
diff ~/mapeval-api/Evaluator2.py src/mapeval_api/Evaluator2.py
```

작업 지침은 `AGENT.md`, 원본과의 대응·차이는 `docs/UPSTREAM_MAPPING.md`,
데이터셋의 출처와 구축 규칙은 `docs/REFERENCE_MAPPING.md`에 있습니다.
