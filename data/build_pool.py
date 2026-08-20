"""Collect a Seoul place pool from Kakao Local, the source the benchmark is graded against.

Upstream Spatial-Agent builds its evidence with `data/build_cache.py`; this is the same idea for
a benchmark that must be gradable by the provider the agents query. Every place here comes from
Kakao, so a gold answer can never disagree with what an agent retrieves for source reasons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data._toolkit.kakao import KakaoMapProvider  # noqa: E402
from src.config import Settings  # noqa: E402

# Seoul's 25 district offices, used only as search centers so the pool covers the whole city
# rather than clustering on one downtown query.
DISTRICT_OFFICES = [
    "서울 종로구청", "서울 중구청", "서울 용산구청", "서울 성동구청", "서울 광진구청",
    "서울 동대문구청", "서울 중랑구청", "서울 성북구청", "서울 강북구청", "서울 도봉구청",
    "서울 노원구청", "서울 은평구청", "서울 서대문구청", "서울 마포구청", "서울 양천구청",
    "서울 강서구청", "서울 구로구청", "서울 금천구청", "서울 영등포구청", "서울 동작구청",
    "서울 관악구청", "서울 서초구청", "서울 강남구청", "서울 송파구청", "서울 강동구청",
]

CATEGORIES = [
    "AT4",  # tourist attraction
    "CT1",  # culture
    "AD5",  # accommodation
    "SW8",  # subway station
    "CE7",  # cafe
    "FD6",  # restaurant
    "MT1",  # large market
    "CS2",  # convenience store
    "BK9",  # bank
    "PO3",  # public office
    "SC4",  # school
    "HP8",  # hospital
    "PK6",  # parking
]

RADIUS_M = 3000
LIMIT = 45


def main() -> None:
    settings = Settings()
    provider = KakaoMapProvider(
        settings.kakao_rest_api_key,
        timeout=settings.kakao_timeout_seconds,
        cache_path=settings.kakao_cache_db_path,
        cache_ttl_seconds=settings.kakao_cache_ttl_seconds,
        search_center=settings.search_center(),
        search_radius_m=settings.kakao_search_radius_m,
    )
    pool: dict[str, dict] = {}
    for office in DISTRICT_OFFICES:
        found = provider.search_place(office, limit=1)
        if not found:
            print(f"  ! no center for {office}", flush=True)
            continue
        center = found[0]
        for code in CATEGORIES:
            try:
                places = provider.nearby_search(
                    center, category_code=code, radius_m=RADIUS_M, limit=LIMIT
                )
            except Exception as error:  # noqa: BLE001 - a partial pool is still usable
                print(f"  ! {office}/{code}: {type(error).__name__}: {error}", flush=True)
                continue
            for place in places:
                pool[place.place_id] = {
                    **place.model_dump(),
                    "category_code": code,
                    "district": office.replace("서울 ", "").replace("청", ""),
                }
        print(f"{office}: pool={len(pool)}", flush=True)

    out = Path(__file__).with_name("seoul_kakao_pool.json")
    out.write_text(json.dumps(list(pool.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out} places={len(pool)} api_calls={provider.api_call_count}")
    provider.close()


if __name__ == "__main__":
    main()
