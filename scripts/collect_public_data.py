"""공식 공개데이터를 내려받고 강남구 도로망에 연결한다."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_OUTPUT = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "reference"
    / "icing_segments"
    / "habitual_icing_segments_20251107.csv"
)
CLEANED_OUTPUT = (
    PROJECT_DIR
    / "outputs"
    / "data"
    / "cleaned"
    / "gangnam_habitual_icing_segments.geojson"
)
MATCH_OUTPUT = PROJECT_DIR / "outputs" / "reports" / "habitual_icing_road_matches.csv"
REPORT_OUTPUT = PROJECT_DIR / "outputs" / "reports" / "public_data_collection.json"
ROADS_OUTPUT = PROJECT_DIR / "outputs" / "data" / "cleaned" / "gangnam_roads_clean.geojson"
KMA_RAW_DIR = PROJECT_DIR / "data" / "raw" / "weather" / "road_observations"
KMA_DAILY_CACHE_DIR = KMA_RAW_DIR / "daily"
KMA_HOURLY_OUTPUT = PROJECT_DIR / "outputs" / "data" / "cleaned" / "kma_road_weather_hourly.csv"

DEFAULT_KMA_START = date(2024, 11, 1)
DEFAULT_KMA_END = date(2026, 2, 1)
GANGNAM_CENTER = (37.4979, 127.0276)
WINTER_MONTHS = {11, 12, 1, 2}

KMA_COLUMNS = [
    "observed_at",
    "station_id",
    "station_name",
    "road_name",
    "station_type",
    "lat",
    "lon",
    "height_m",
    "road_surface_state",
    "road_surface_temp",
    "visibility_m",
    "humidity",
    "daily_precip_mm",
    "snow_depth_cm",
    "air_temp",
    "wind_direction_1min",
    "wind_speed_1min",
    "gust_direction",
    "gust_speed",
    "sea_level_pressure_hpa",
    "channel_1",
    "channel_2",
    "channel_3",
    "cloud_amount",
]
KMA_NUMERIC_COLUMNS = KMA_COLUMNS[5:]

HABITUAL_ICING_SOURCE_PAGE = "https://www.data.go.kr/data/15067396/fileData.do"
HABITUAL_ICING_DOWNLOAD_URL = (
    "https://www.data.go.kr/cmm/cmm/fileDownload.do"
    "?atchFileId=FILE_000000003556460&fileDetailSn=1&insertDataPrcus=N"
)
KMA_ROAD_WEATHER_PAGE = (
    "https://apihub.kma.go.kr/apiList.do"
    "?seqApi=971&seqApiSub=293&apiMov=3.%20도로기상관측정보"
)
KMA_ROAD_OBSERVATION_ENDPOINT = (
    "https://apihub.kma.go.kr/api/typ01/url/road_stn_obs.php"
)

REQUIRED_ICING_COLUMNS = {
    "구간번호",
    "관리청",
    "도로분류",
    "대표지역",
    "도로(노선)명",
    "기점 위도(WGS84(4326))",
    "기점 경도(WGS84(4326))",
    "종점 위도(WGS84(4326))",
    "종점 경도(WGS84(4326))",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, output: Path) -> None:
    """중간 파일을 거쳐 원자적으로 공개데이터를 저장한다."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with requests.get(url, timeout=90, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if temporary.stat().st_size < 1000:
        raise ValueError("다운로드 파일이 비정상적으로 작습니다.")
    temporary.replace(output)


def request_kma_text(params: dict[str, object], api_key: str, retries: int = 2) -> bytes:
    """인증키가 예외 메시지에 노출되지 않도록 기상청 API를 호출한다."""
    request_params = dict(params)
    request_params["authKey"] = api_key
    last_error = "unknown"
    for attempt in range(retries):
        try:
            response = requests.get(
                KMA_ROAD_OBSERVATION_ENDPOINT,
                params=request_params,
                timeout=(10, 20),
            )
            if response.status_code == 200:
                return response.content
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        if attempt + 1 < retries:
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"기상청 API 호출 실패({last_error})")


def parse_kma_road_observations(payload: bytes) -> pd.DataFrame:
    """EUC-KR CSV형 도로기상 응답을 정규화하고 -999 결측값을 제거한다."""
    text = payload.decode("euc-kr", errors="replace")
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parsed = next(csv.reader([line], skipinitialspace=True))
        if len(parsed) == len(KMA_COLUMNS):
            rows.append([value.strip() for value in parsed])

    frame = pd.DataFrame(rows, columns=KMA_COLUMNS)
    if frame.empty:
        return frame
    frame["observed_at"] = pd.to_datetime(
        frame["observed_at"], format="%Y%m%d%H%M", errors="coerce"
    )
    for column in KMA_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column] <= -900, column] = np.nan
    frame["station_id"] = frame["station_id"].astype(str)
    frame["road_surface_state"] = frame["road_surface_state"].round().astype("Int64")
    return frame.dropna(subset=["observed_at", "station_id"]).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def select_nearest_icing_station(
    stations: pd.DataFrame,
    center: tuple[float, float] = GANGNAM_CENTER,
) -> dict[str, object]:
    """목표(결빙) 관측소 중 강남 중심과 가장 가까운 유효 관측소를 고른다."""
    valid = stations.dropna(subset=["lat", "lon"]).copy()
    if valid.empty:
        raise ValueError("유효 좌표를 가진 결빙 목표 관측소가 없습니다.")
    valid["distance_to_gangnam_km"] = [
        haversine_km(center[0], center[1], lat, lon)
        for lat, lon in valid[["lat", "lon"]].itertuples(index=False, name=None)
    ]
    row = valid.sort_values("distance_to_gangnam_km").iloc[0]
    return {
        "station_id": str(row["station_id"]),
        "station_name": str(row["station_name"]),
        "road_name": str(row["road_name"]),
        "station_type": str(row["station_type"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "distance_to_gangnam_km": float(row["distance_to_gangnam_km"]),
    }


def discover_nearest_kma_station(api_key: str, reference_time: datetime) -> dict[str, object]:
    payload = request_kma_text(
        {
            "mode": 3,
            "tm1": reference_time.strftime("%Y%m%d%H%M"),
            "var": 4,
            "help": 0,
            "disp": 1,
        },
        api_key,
    )
    return select_nearest_icing_station(parse_kma_road_observations(payload))


def winter_days(start: date, end: date) -> list[date]:
    """종료일 미포함 범위에서 11~2월 날짜만 반환한다."""
    if end <= start:
        raise ValueError("KMA 종료일은 시작일보다 뒤여야 합니다.")
    days = []
    current = start
    while current < end:
        if current.month in WINTER_MONTHS:
            days.append(current)
        current += timedelta(days=1)
    return days


def fetch_kma_station_day(api_key: str, station_id: str, day: date) -> pd.DataFrame:
    payload = request_kma_text(
        {
            "mode": 1,
            "tm1": f"{day:%Y%m%d}0000",
            "tm2": f"{day:%Y%m%d}2359",
            "var": station_id,
            "help": 0,
            "disp": 1,
        },
        api_key,
    )
    return parse_kma_road_observations(payload)


def fetch_kma_station_day_cached(
    api_key: str,
    station_id: str,
    day: date,
    refresh: bool = False,
) -> pd.DataFrame:
    """날짜별 응답을 즉시 저장해 중단 후에도 이어받을 수 있게 한다."""
    cache_path = KMA_DAILY_CACHE_DIR / station_id / f"{day:%Y%m%d}.csv.gz"
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, compression="gzip", parse_dates=["observed_at"])
        cached["station_id"] = cached["station_id"].astype(str)
        cached["road_surface_state"] = cached["road_surface_state"].astype("Int64")
        return cached

    frame = fetch_kma_station_day(api_key, station_id, day)
    if not frame.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".part")
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            compression="gzip",
        )
        temporary.replace(cache_path)
    return frame


def aggregate_kma_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """분 단위 관측을 모델 결합용 시간 단위 자료로 집계한다."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    # 00:00~00:59 평균은 01:00 시점에 확정된 과거 한 시간 관측으로 취급한다.
    work["hour"] = work["observed_at"].dt.floor("h") + pd.Timedelta(hours=1)
    work["freezing_observation"] = work["road_surface_temp"].le(0).where(
        work["road_surface_temp"].notna()
    )

    def state_mode(series: pd.Series) -> object:
        mode = series.dropna().mode()
        return mode.iloc[0] if not mode.empty else pd.NA

    hourly = work.groupby(["station_id", "hour"], as_index=False).agg(
        station_name=("station_name", "first"),
        road_name=("road_name", "first"),
        station_type=("station_type", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        height_m=("height_m", "first"),
        road_surface_state=("road_surface_state", state_mode),
        road_surface_temp=("road_surface_temp", "mean"),
        road_surface_temp_min=("road_surface_temp", "min"),
        road_surface_temp_max=("road_surface_temp", "max"),
        freezing_ratio=("freezing_observation", "mean"),
        observation_count=("observed_at", "count"),
    )
    hourly["road_surface_state"] = hourly["road_surface_state"].astype("Int64")
    return hourly.sort_values(["station_id", "hour"]).reset_index(drop=True)


def collect_kma_road_weather(
    api_key: str,
    start: date = DEFAULT_KMA_START,
    end: date = DEFAULT_KMA_END,
    station_id: str | None = None,
    refresh: bool = False,
) -> dict[str, object]:
    """강남 인접 결빙 관측소의 겨울철 이력을 수집하고 시간 단위로 정제한다."""
    days = winter_days(start, end)
    if not days:
        raise ValueError("선택 범위에 겨울철 날짜가 없습니다.")

    reference_time = datetime.combine(days[-1], datetime.min.time()).replace(hour=12)
    discovered = discover_nearest_kma_station(api_key, reference_time)
    selected_id = str(station_id or discovered["station_id"])
    if station_id and selected_id != discovered["station_id"]:
        discovered = {
            "station_id": selected_id,
            "station_name": "사용자 지정",
            "road_name": "unknown",
            "station_type": "unknown",
            "lat": None,
            "lon": None,
            "distance_to_gangnam_km": None,
        }

    raw_path = KMA_RAW_DIR / (
        f"road_weather_{selected_id}_{start:%Y%m%d}_{end:%Y%m%d}.csv.gz"
    )
    failures: list[str] = []
    if raw_path.exists() and not refresh:
        minute = pd.read_csv(raw_path, compression="gzip", parse_dates=["observed_at"])
        minute["station_id"] = minute["station_id"].astype(str)
        minute["road_surface_state"] = minute["road_surface_state"].astype("Int64")
    else:
        frames: list[pd.DataFrame] = []
        print(
            f"기상청 도로기상 수집: 관측소 {selected_id}, "
            f"겨울철 {len(days)}일",
            flush=True,
        )
        # API 서버의 동시 요청 제한을 고려해 낮은 병렬도로 수집한다.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    fetch_kma_station_day_cached,
                    api_key,
                    selected_id,
                    day,
                    refresh,
                ): day
                for day in days
            }
            completed = 0
            for future in as_completed(futures):
                day = futures[future]
                try:
                    daily = future.result()
                    if daily.empty:
                        failures.append(f"{day:%Y-%m-%d}:empty")
                    else:
                        frames.append(daily)
                except Exception as exc:
                    failures.append(f"{day:%Y-%m-%d}:{type(exc).__name__}")
                completed += 1
                if completed % 20 == 0 or completed == len(days):
                    print(
                        f"  {completed}/{len(days)}일 완료, 실패 {len(failures)}일",
                        flush=True,
                    )

        if not frames:
            raise RuntimeError("기상청 도로기상 관측을 한 건도 수집하지 못했습니다.")
        if len(failures) / len(days) > 0.1:
            raise RuntimeError(
                f"기상청 도로기상 수집 실패율이 10%를 초과했습니다({len(failures)}/{len(days)})."
            )
        minute = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["station_id", "observed_at"])
            .sort_values("observed_at")
            .reset_index(drop=True)
        )
        KMA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        minute.to_csv(raw_path, index=False, encoding="utf-8", compression="gzip")

    hourly = aggregate_kma_hourly(minute)
    KMA_HOURLY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(KMA_HOURLY_OUTPUT, index=False, encoding="utf-8-sig")
    valid_temp = int(minute["road_surface_temp"].notna().sum())
    unique_minutes = int(minute[["station_id", "observed_at"]].drop_duplicates().shape[0])
    expected_minutes = int(len(days) * 24 * 60)
    road_temperature = minute["road_surface_temp"].dropna()
    return {
        "status": "collected",
        "station": discovered,
        "requested_station_id": selected_id,
        "start_date": start.isoformat(),
        "end_date_exclusive": end.isoformat(),
        "winter_days_requested": len(days),
        "failed_days": failures,
        "minute_rows": int(len(minute)),
        "expected_minute_rows": expected_minutes,
        "missing_minute_rows": max(expected_minutes - unique_minutes, 0),
        "minute_coverage_ratio": float(unique_minutes / expected_minutes),
        "valid_road_surface_temperature_rows": valid_temp,
        "road_surface_temperature_min_c": float(road_temperature.min()),
        "road_surface_temperature_max_c": float(road_temperature.max()),
        "road_surface_temperature_freezing_ratio": float(
            road_temperature.le(0).mean()
        ),
        "hourly_rows": int(len(hourly)),
        "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
        "raw_sha256": sha256_file(raw_path),
        "hourly_path": KMA_HOURLY_OUTPUT.relative_to(PROJECT_DIR).as_posix(),
        "hour_timestamp_semantics": "preceding-hour aggregate labeled at hour end",
        "missing_value_rule": "API -999 계열 값을 결측치로 변환",
        "scope_warning": "강남 직접 관측이 아닌 인접 고속도로 관측소 자료",
    }


def load_habitual_icing_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    missing = sorted(REQUIRED_ICING_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"상습결빙구간 필수 컬럼 누락: {missing}")
    return frame


def gangnam_icing_segments(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    """전국 자료에서 강남구와 유효 좌표를 가진 구간만 추출한다."""
    selected = frame[
        frame["대표지역"].astype(str).str.contains("서울특별시 강남구", na=False)
    ].copy()
    coordinate_columns = [
        "기점 위도(WGS84(4326))",
        "기점 경도(WGS84(4326))",
        "종점 위도(WGS84(4326))",
        "종점 경도(WGS84(4326))",
    ]
    for column in coordinate_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=coordinate_columns).copy()
    selected = selected[
        selected["기점 위도(WGS84(4326))"].between(33, 39)
        & selected["종점 위도(WGS84(4326))"].between(33, 39)
        & selected["기점 경도(WGS84(4326))"].between(124, 132)
        & selected["종점 경도(WGS84(4326))"].between(124, 132)
    ].copy()
    selected["geometry"] = [
        LineString([(start_lon, start_lat), (end_lon, end_lat)])
        for start_lat, start_lon, end_lat, end_lon in selected[
            coordinate_columns
        ].itertuples(index=False, name=None)
    ]
    return gpd.GeoDataFrame(selected, geometry="geometry", crs="EPSG:4326")


def match_segments_to_roads(
    segments: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    buffer_m: float = 20.0,
) -> pd.DataFrame:
    """상습결빙 선분과 20m 이내 표준 링크를 모두 연결한다."""
    segments_metric = segments.to_crs(epsg=5186)
    roads_metric = roads.to_crs(epsg=5186)
    rows: list[dict[str, object]] = []
    for _, segment in segments_metric.iterrows():
        distances = roads_metric.geometry.distance(segment.geometry)
        local = distances[distances <= buffer_m].index.tolist()
        if not local:
            local = [distances.idxmin()]
        rows.append(
            {
                "segment_id": str(segment["구간번호"]),
                "region": str(segment["대표지역"]),
                "road_name": str(segment["도로(노선)명"]),
                "source_length_km": float(
                    pd.to_numeric(segment.get("총길이(km)"), errors="coerce")
                ),
                "matched_link_count": int(len(local)),
                "matched_link_ids": "|".join(
                    roads_metric.loc[local, "LINK_ID"].astype(str).tolist()
                ),
                "minimum_distance_m": float(distances.loc[local].min()),
                "buffer_m": float(buffer_m),
            }
        )
    return pd.DataFrame(rows)


def collect(
    refresh: bool = False,
    collect_kma: bool = True,
    kma_start: date = DEFAULT_KMA_START,
    kma_end: date = DEFAULT_KMA_END,
    kma_station: str | None = None,
) -> dict[str, object]:
    if refresh or not RAW_OUTPUT.exists():
        download_file(HABITUAL_ICING_DOWNLOAD_URL, RAW_OUTPUT)

    national = load_habitual_icing_csv(RAW_OUTPUT)
    gangnam = gangnam_icing_segments(national)
    if gangnam.empty:
        raise ValueError("공개자료에서 강남구 상습결빙구간을 찾지 못했습니다.")
    if not ROADS_OUTPUT.exists():
        raise FileNotFoundError("scripts/preprocess_data.py를 먼저 실행하세요.")
    roads = gpd.read_file(ROADS_OUTPUT)

    CLEANED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gangnam.to_file(CLEANED_OUTPUT, driver="GeoJSON")
    matches = match_segments_to_roads(gangnam, roads)
    MATCH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCH_OUTPUT, index=False, encoding="utf-8-sig")

    kma_key = os.environ.get("KMA_API_KEY", "").strip()
    if collect_kma and kma_key:
        kma_result = collect_kma_road_weather(
            kma_key,
            start=kma_start,
            end=kma_end,
            station_id=kma_station,
            refresh=refresh,
        )
    else:
        kma_result = {
            "status": "skipped" if not collect_kma else "missing_KMA_API_KEY",
            "required_environment_variable": "KMA_API_KEY",
        }
    report = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "habitual_icing_segments": {
            "provider": "행정안전부",
            "source_page": HABITUAL_ICING_SOURCE_PAGE,
            "download_url": HABITUAL_ICING_DOWNLOAD_URL,
            "license": "이용허락범위 제한 없음",
            "raw_path": RAW_OUTPUT.relative_to(PROJECT_DIR).as_posix(),
            "sha256": sha256_file(RAW_OUTPUT),
            "national_rows": int(len(national)),
            "seoul_rows": int(
                national["대표지역"].astype(str).str.contains("서울특별시", na=False).sum()
            ),
            "gangnam_rows": int(len(gangnam)),
            "matched_link_rows": int(matches["matched_link_count"].sum()),
            "cleaned_path": CLEANED_OUTPUT.relative_to(PROJECT_DIR).as_posix(),
            "matches_path": MATCH_OUTPUT.relative_to(PROJECT_DIR).as_posix(),
        },
        "kma_road_weather_observations": {
            "provider": "기상청",
            "source_page": KMA_ROAD_WEATHER_PAGE,
            "endpoint": KMA_ROAD_OBSERVATION_ENDPOINT,
            "available_from": "2024-01-01",
            "variables": [
                "road_surface_state",
                "road_surface_temperature",
                "visibility",
                "humidity",
                "precipitation",
                "snow_depth",
                "air_temperature",
                "wind",
            ],
            "scope": "12개 고속도로 366개 관측소; 강남구 직접 관측 아님",
            **kma_result,
        },
    }
    REPORT_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="캐시된 원본이 있어도 공식 페이지에서 다시 다운로드",
    )
    parser.add_argument(
        "--skip-kma",
        action="store_true",
        help="KMA_API_KEY가 있어도 도로기상 API 수집을 건너뜀",
    )
    parser.add_argument(
        "--kma-start",
        type=date.fromisoformat,
        default=DEFAULT_KMA_START,
        help="도로기상 수집 시작일(YYYY-MM-DD, 기본 2024-11-01)",
    )
    parser.add_argument(
        "--kma-end",
        type=date.fromisoformat,
        default=DEFAULT_KMA_END,
        help="도로기상 수집 종료일(미포함, 기본 2026-02-01)",
    )
    parser.add_argument(
        "--kma-station",
        help="관측소 번호. 생략하면 강남에서 가장 가까운 결빙 목표 관측소 선택",
    )
    args = parser.parse_args()
    collect(
        refresh=args.refresh,
        collect_kma=not args.skip_kma,
        kma_start=args.kma_start,
        kma_end=args.kma_end,
        kma_station=args.kma_station,
    )


if __name__ == "__main__":
    main()
