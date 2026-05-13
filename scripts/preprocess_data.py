"""원본 데이터를 점검하고 정제본을 outputs/data/cleaned 아래에 저장한다."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "data" / "cleaned"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

GN_LON_MIN, GN_LON_MAX = 127.01, 127.09
GN_LAT_MIN, GN_LAT_MAX = 37.47, 37.53
GANGNAM_CODE = "11680"


def file_head_is_lfs(path: Path) -> bool:
    """Git LFS pointer 파일인지 확인한다."""
    return path.read_bytes()[:128].startswith(b"version https://git-lfs.github.com/spec")


def issue(level: str, dataset: str, message: str, count: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"level": level, "dataset": dataset, "message": message}
    if count is not None:
        item["count"] = int(count)
    return item


def safe_to_csv(df: pd.DataFrame, path: Path, report: list[dict[str, Any]], dataset: str) -> Path:
    """Windows 파일 잠금이 있어도 전처리 결과를 잃지 않도록 대체 파일명으로 저장한다."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        report.append(issue("warning", dataset, f"{path.name} 파일이 잠겨 대체 파일로 저장: {fallback.name}"))
        return fallback


def fix_geometry(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """invalid geometry를 가능한 범위에서 복구한다."""
    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return gdf, 0

    gdf = gdf.copy()
    try:
        from shapely.validation import make_valid

        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
    except Exception:
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
    return gdf, invalid_count


def clean_roads(report: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    road_path = RAW_DIR / "roads" / "[2024-03-25]NODELINKDATA" / "MOCT_LINK.shp"
    if file_head_is_lfs(road_path):
        raise FileNotFoundError(f"도로 shp가 Git LFS pointer입니다: {road_path}")

    roads = gpd.read_file(road_path).to_crs(epsg=4326)
    raw_count = len(roads)
    candidate = roads.cx[GN_LON_MIN:GN_LON_MAX, GN_LAT_MIN:GN_LAT_MAX].copy()
    metric_centroids = candidate.to_crs(epsg=5186).geometry.centroid
    lonlat_centroids = gpd.GeoSeries(metric_centroids, crs="EPSG:5186").to_crs(epsg=4326)
    candidate["cent_lon"] = lonlat_centroids.x.values
    candidate["cent_lat"] = lonlat_centroids.y.values
    cleaned = candidate[
        candidate["cent_lon"].between(GN_LON_MIN, GN_LON_MAX)
        & candidate["cent_lat"].between(GN_LAT_MIN, GN_LAT_MAX)
    ].copy()

    keep_cols = ["LINK_ID", "F_NODE", "T_NODE", "LANES", "ROAD_RANK", "MAX_SPD", "LENGTH", "cent_lon", "cent_lat", "geometry"]
    cleaned = cleaned[keep_cols]
    before_numeric = len(cleaned)
    cleaned["LANES"] = pd.to_numeric(cleaned["LANES"], errors="coerce").fillna(2).astype(int)
    cleaned["MAX_SPD"] = pd.to_numeric(cleaned["MAX_SPD"], errors="coerce")
    cleaned["LENGTH"] = pd.to_numeric(cleaned["LENGTH"], errors="coerce")
    cleaned["ROAD_RANK"] = cleaned["ROAD_RANK"].astype(str)
    cleaned = cleaned.dropna(subset=["LENGTH", "geometry"])
    cleaned = cleaned[cleaned["LENGTH"] > 0]
    duplicate_count = int(cleaned["LINK_ID"].duplicated().sum())
    cleaned = cleaned.drop_duplicates("LINK_ID").reset_index(drop=True)

    fixed, invalid_count = fix_geometry(cleaned)
    cleaned = fixed[~fixed.geometry.is_empty & fixed.geometry.notna()].reset_index(drop=True)
    cleaned.to_file(OUTPUT_DIR / "gangnam_roads_clean.geojson", driver="GeoJSON")

    report.append(issue("info", "roads", f"원본 도로 {raw_count:,}개 중 강남 중심점 기준 {len(cleaned):,}개 사용"))
    removed_by_centroid = len(candidate) - before_numeric
    if removed_by_centroid:
        report.append(issue("warning", "roads", "bbox와 걸쳤지만 중심점이 강남 범위 밖인 링크 제거", removed_by_centroid))
    if duplicate_count:
        report.append(issue("warning", "roads", "중복 LINK_ID 제거", duplicate_count))
    if invalid_count:
        report.append(issue("warning", "roads", "invalid geometry 복구", invalid_count))
    return cleaned


def clean_buildings(report: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    building_path = RAW_DIR / "buildings" / "AL_D162_11_20260115" / "AL_D162_11_20260115.shp"
    if file_head_is_lfs(building_path):
        raise FileNotFoundError(f"건물 shp가 Git LFS pointer입니다: {building_path}")

    buildings = gpd.read_file(building_path)
    gangnam = buildings[buildings["A39"].astype(str).str.startswith(GANGNAM_CODE)].copy()
    height_raw = pd.to_numeric(gangnam["A31"], errors="coerce")
    floors = pd.to_numeric(gangnam["A18"], errors="coerce")
    height_clean = height_raw.fillna(0)
    fallback_mask = height_clean <= 0
    height_clean.loc[fallback_mask] = floors.loc[fallback_mask].fillna(1).clip(lower=1) * 3
    gangnam["height_m"] = height_clean.clip(lower=0, upper=250)
    gangnam = gangnam[gangnam.geometry.notna() & ~gangnam.geometry.is_empty].copy()
    gangnam, invalid_count = fix_geometry(gangnam)
    gangnam = gangnam[gangnam.geometry.notna() & ~gangnam.geometry.is_empty].reset_index(drop=True)

    cols = ["A0", "A18", "A31", "A39", "height_m", "geometry"]
    available = [col for col in cols if col in gangnam.columns]
    gangnam[available].to_file(OUTPUT_DIR / "gangnam_buildings_clean.geojson", driver="GeoJSON")

    report.append(issue("info", "buildings", f"강남구 건물 {len(gangnam):,}개 정제"))
    report.append(issue("warning", "buildings", "건물 높이 결측/0 값을 층수 기반 height_m으로 보정", int(fallback_mask.sum())))
    if invalid_count:
        report.append(issue("warning", "buildings", "invalid geometry 복구", invalid_count))
    return gangnam


def clean_weather(report: list[dict[str, Any]]) -> pd.DataFrame:
    weather_dir = RAW_DIR / "weather"
    main_cols = ["일시", "기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)", "적설(cm)", "지면온도(°C)"]
    frames = []
    skipped = []
    for path in sorted(weather_dir.glob("OBS_ASOS_TIM_*.csv")):
        df = pd.read_csv(path, encoding="cp949")
        if "기온(°C)" not in df.columns:
            skipped.append(path.name)
            continue
        available = [col for col in main_cols if col in df.columns]
        df = df[available].copy()
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("기온 컬럼이 있는 OBS_ASOS_TIM CSV가 없습니다.")

    weather = pd.concat(frames, ignore_index=True)
    weather["일시"] = pd.to_datetime(weather["일시"], errors="coerce")
    null_time = int(weather["일시"].isna().sum())
    weather = weather.dropna(subset=["일시"]).sort_values("일시")
    duplicate_count = int(weather["일시"].duplicated().sum())
    weather = weather.drop_duplicates("일시", keep="last")

    for col in ["강수량(mm)", "적설(cm)"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce").fillna(0)
    for col in ["기온(°C)", "습도(%)", "풍속(m/s)", "지면온도(°C)"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce").ffill().bfill()

    weather["month"] = weather["일시"].dt.month
    winter = weather[weather["month"].isin([11, 12, 1, 2])].copy()
    winter = winter.rename(
        columns={
            "기온(°C)": "temp",
            "강수량(mm)": "precip",
            "풍속(m/s)": "wind",
            "습도(%)": "humidity",
            "적설(cm)": "snow",
            "지면온도(°C)": "ground_temp",
        }
    )
    safe_to_csv(winter, OUTPUT_DIR / "weather_winter_clean.csv", report, "weather")

    if skipped:
        report.append(issue("warning", "weather", "기온 컬럼 없는 OBS 보조 파일 제외: " + ", ".join(skipped)))
    if null_time:
        report.append(issue("warning", "weather", "일시 파싱 실패 행 제거", null_time))
    if duplicate_count:
        report.append(issue("warning", "weather", "중복 일시 행 제거", duplicate_count))
    report.append(issue("info", "weather", f"겨울철 기상 {len(winter):,}행 정제"))
    return winter


def grid_to_lonlat(grid_id: Any, transformer: Any) -> tuple[float | None, float | None]:
    gid = str(grid_id).replace("다사", "")
    if len(gid) < 8 or not gid[:8].isdigit():
        return None, None
    x_5179 = 900000 + int(gid[:4]) * 10
    y_5179 = 1900000 + int(gid[4:8]) * 10
    return transformer.transform(x_5179, y_5179)


def clean_population(report: list[dict[str, Any]]) -> pd.DataFrame:
    import pyproj

    pop_dirs = [
        RAW_DIR / "population" / "250_LOCAL_RESD_202501",
        RAW_DIR / "population" / "250_LOCAL_RESD_202512",
    ]
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    total_rows = 0
    gangnam_rows = 0

    for folder in pop_dirs:
        for path in sorted(folder.glob("*.csv")):
            df = pd.read_csv(path, encoding="cp949")
            df.columns = [col.strip().strip('"') for col in df.columns]
            required = ["행정동코드", "250M격자", "생활인구합계"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                report.append(issue("error", "population", f"{path.name} 필수 컬럼 누락: {missing}"))
                continue
            total_rows += len(df)
            df["행정동코드"] = df["행정동코드"].astype(str)
            df = df[df["행정동코드"].str.startswith(GANGNAM_CODE)].copy()
            gangnam_rows += len(df)
            pop = pd.to_numeric(df["생활인구합계"].astype(str).str.replace("*", "0", regex=False), errors="coerce").fillna(0)
            grouped = pop.groupby(df["250M격자"].astype(str)).agg(["sum", "count"])
            for grid_id, row in grouped.iterrows():
                sums[grid_id] += float(row["sum"])
                counts[grid_id] += int(row["count"])

    grid_avg = pd.DataFrame(
        {
            "grid_id": list(sums.keys()),
            "avg_pop": [sums[key] / max(counts[key], 1) for key in sums.keys()],
            "sample_count": [counts[key] for key in sums.keys()],
        }
    )
    transformer = pyproj.Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    coords = [grid_to_lonlat(grid_id, transformer) for grid_id in grid_avg["grid_id"]]
    grid_avg["grid_lon"] = [lon for lon, _ in coords]
    grid_avg["grid_lat"] = [lat for _, lat in coords]
    bad_coord = int(grid_avg["grid_lon"].isna().sum())
    grid_avg = grid_avg.dropna(subset=["grid_lon", "grid_lat"])
    grid_avg = grid_avg[
        grid_avg["grid_lon"].between(GN_LON_MIN, GN_LON_MAX)
        & grid_avg["grid_lat"].between(GN_LAT_MIN, GN_LAT_MAX)
    ].copy()
    pop_max = max(float(grid_avg["avg_pop"].max()), 1.0)
    grid_avg["pop_weight"] = grid_avg["avg_pop"] / pop_max
    safe_to_csv(grid_avg, OUTPUT_DIR / "population_grid_clean.csv", report, "population")

    report.append(issue("info", "population", f"생활인구 원본 {total_rows:,}행 중 강남 {gangnam_rows:,}행 집계"))
    report.append(issue("info", "population", f"강남 250m 격자 {len(grid_avg):,}개 정제"))
    if bad_coord:
        report.append(issue("warning", "population", "좌표 변환 실패 격자 제거", bad_coord))
    return grid_avg


def inspect_accident_references(report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accident_dir = RAW_DIR / "reference" / "accidents"
    references = []
    for path in sorted(accident_dir.glob("*.xlsx")):
        try:
            xls = pd.ExcelFile(path)
            references.append({"file": path.name, "sheets": xls.sheet_names})
        except ImportError:
            report.append(issue("warning", "accidents", f"{path.name} 확인에는 openpyxl이 필요합니다. 현재 학습 파이프라인에서는 참고자료로만 둡니다."))
        except Exception as exc:
            report.append(issue("warning", "accidents", f"{path.name} 읽기 실패: {exc}"))
    return references


def write_report(report: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"issues": report, "extra": extra}
    json_path = REPORT_DIR / "data_quality_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Data Quality Report", ""]
    for item in report:
        count = f" ({item['count']})" if "count" in item else ""
        lines.append(f"- [{item['level']}] {item['dataset']}: {item['message']}{count}")
    lines.append("")
    lines.append("## Cleaned Outputs")
    for path in sorted(OUTPUT_DIR.glob("*")):
        lines.append(f"- `{path.relative_to(PROJECT_DIR)}`")
    (REPORT_DIR / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    roads = clean_roads(report)
    buildings = clean_buildings(report)
    weather = clean_weather(report)
    population = clean_population(report)
    accident_refs = inspect_accident_references(report)
    write_report(
        report,
        {
            "roads_clean_rows": len(roads),
            "buildings_clean_rows": len(buildings),
            "weather_winter_rows": len(weather),
            "population_grid_rows": len(population),
            "accident_references": accident_refs,
        },
    )
    print(json.dumps({"status": "ok", "issues": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
