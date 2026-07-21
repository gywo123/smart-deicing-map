"""실제 결빙사고 다발지역을 이용한 독립 공간 proxy 검증."""

from __future__ import annotations

import json
import os
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ACCIDENT_PATH = PROJECT_DIR / "data" / "raw" / "weather" / "13_24_freezing.csv"
OUTPUT_DATA_DIR = PROJECT_DIR / "outputs" / "data"
REPORTS_DIR = PROJECT_DIR / "outputs" / "reports"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
MAPS_DIR = PROJECT_DIR / "outputs" / "maps"
HABITUAL_ICING_MATCH_PATH = REPORTS_DIR / "habitual_icing_road_matches.csv"


def configure_korean_font() -> None:
    for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/NanumGothic.ttf"]:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_korean_font()


def weather_bucket(row: pd.Series) -> str:
    """기존 시각화 테스트와 문서 호환을 위한 기상 구분 함수."""
    if float(row.get("snow", 0) or 0) > 0:
        return "눈"
    if float(row.get("precip", 0) or 0) > 0:
        return "비"
    if float(row.get("humidity", 0) or 0) >= 80:
        return "흐림"
    return "맑음"


def scenario_severity(weather: pd.DataFrame) -> np.ndarray:
    """지도 설명용 기상 강도이며 모델 성능 검증 정답으로 사용하지 않는다."""
    temp = pd.to_numeric(weather["temp"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    ground = pd.to_numeric(weather.get("ground_temp", weather["temp"]), errors="coerce").ffill().bfill().to_numpy(dtype=float)
    snow = pd.to_numeric(weather.get("snow", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
    precip = pd.to_numeric(weather.get("precip", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
    return np.clip(
        0.45 * np.clip((2 - ground) / 10, 0, 1)
        + 0.30 * np.clip((2 - temp) / 14, 0, 1)
        + 0.15 * np.clip(snow / 5, 0, 1)
        + 0.10 * np.clip(precip / 5, 0, 1),
        0,
        1,
    )


def load_roads() -> gpd.GeoDataFrame:
    path = OUTPUT_DATA_DIR / "gangnam_roads_result.geojson"
    if not path.exists():
        raise FileNotFoundError("MLP 파이프라인의 gangnam_roads_result.geojson이 필요합니다.")
    roads = gpd.read_file(path).to_crs(epsg=4326)
    required = {"LINK_ID", "risk", "geometry"}
    missing = sorted(required - set(roads.columns))
    if missing:
        raise ValueError(f"도로 결과 컬럼 누락: {missing}")
    return roads


def load_gangnam_accident_hotspots() -> gpd.GeoDataFrame:
    """결빙사고 다발지역 원자료에서 강남구 관측만 읽는다."""
    if not ACCIDENT_PATH.exists():
        raise FileNotFoundError(f"결빙사고 다발지역 자료가 없습니다: {ACCIDENT_PATH}")
    accidents = pd.read_csv(ACCIDENT_PATH, encoding="cp949")
    required = {"사고다발지id", "시도시군구명", "지점명", "사고건수", "사상자수", "경도", "위도"}
    missing = sorted(required - set(accidents.columns))
    if missing:
        raise ValueError(f"결빙사고 자료 컬럼 누락: {missing}")
    accidents = accidents[
        accidents["시도시군구명"].astype(str).str.contains("강남구", na=False)
    ].copy()
    if accidents.empty:
        raise ValueError("결빙사고 자료에 강남구 행이 없습니다.")
    return cluster_hotspot_records(accidents)


def cluster_hotspot_records(accidents: pd.DataFrame, distance_m: float = 100.0) -> gpd.GeoDataFrame:
    """여러 연도에 반복된 동일 사고 다발지역을 하나의 실제 장소로 합친다."""
    rows: list[dict[str, object]] = []
    for _, record in accidents.iterrows():
        lon = pd.to_numeric(record.get("경도"), errors="coerce")
        lat = pd.to_numeric(record.get("위도"), errors="coerce")
        if pd.isna(lon) or pd.isna(lat):
            continue
        geometry = None
        raw_polygon = record.get("다발지역폴리곤")
        if isinstance(raw_polygon, str) and raw_polygon.strip():
            try:
                geometry = shape(json.loads(raw_polygon))
            except (ValueError, TypeError, json.JSONDecodeError):
                geometry = None
        if geometry is None or geometry.is_empty:
            geometry = Point(float(lon), float(lat))
        hotspot_id = str(record.get("사고다발지id", ""))
        year = int(hotspot_id[:4]) if hotspot_id[:4].isdigit() else None
        rows.append(
            {
                "location": str(record.get("지점명", "")),
                "year": year,
                "accidents": float(pd.to_numeric(record.get("사고건수"), errors="coerce") or 0),
                "casualties": float(pd.to_numeric(record.get("사상자수"), errors="coerce") or 0),
                "geometry": geometry,
            }
        )
    points = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(epsg=5186)
    parents = list(range(len(points)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    geometries = list(points.geometry)
    centroids = [geometry.centroid for geometry in geometries]
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if geometries[left].intersects(geometries[right]) or centroids[left].distance(centroids[right]) <= distance_m:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    clusters = []
    for cluster_id, indices in enumerate(groups.values(), start=1):
        subset = points.iloc[indices]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "locations": sorted(set(subset["location"].astype(str))),
                "years": sorted(int(year) for year in subset["year"].dropna().unique()),
                "source_records": int(len(subset)),
                "accidents": float(subset["accidents"].sum()),
                "casualties": float(subset["casualties"].sum()),
                "geometry": unary_union(list(subset.geometry)),
            }
        )
    return gpd.GeoDataFrame(clusters, geometry="geometry", crs="EPSG:5186")


def hotspot_road_indices(
    roads_metric: gpd.GeoDataFrame,
    geometry: object,
) -> tuple[list[object], float]:
    """사고 다발지역과 겹치는 모든 도로를 반환하고, 없으면 최단거리 도로를 쓴다."""
    distances = roads_metric.geometry.distance(geometry)
    local = distances[distances <= 1.0].index.tolist()
    if not local:
        local = [distances.idxmin()]
    return local, float(distances.loc[local].min())


def evaluate_hotspot_risk_ranking(
    roads_metric: gpd.GeoDataFrame,
    hotspots: gpd.GeoDataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """지역별 최고값 선택 없이 겹치는 도로 전체의 평균 순위를 평가한다."""
    risk_percentile = roads_metric["risk"].rank(method="average", pct=True)
    matches = []
    for _, hotspot in hotspots.iterrows():
        local_indices, distance_m = hotspot_road_indices(roads_metric, hotspot.geometry)
        local_risk = roads_metric.loc[local_indices, "risk"].astype(float)
        local_percentile = risk_percentile.loc[local_indices].astype(float)
        matches.append(
            {
                "cluster_id": int(hotspot["cluster_id"]),
                "locations": " | ".join(hotspot["locations"]),
                "years": ",".join(str(year) for year in hotspot["years"]),
                "source_records": int(hotspot["source_records"]),
                "accidents": float(hotspot["accidents"]),
                "casualties": float(hotspot["casualties"]),
                "local_link_count": int(len(local_indices)),
                "matched_link_ids": "|".join(
                    roads_metric.loc[local_indices, "LINK_ID"].astype(str).tolist()
                ),
                "distance_m": distance_m,
                "risk_mean": float(local_risk.mean()),
                "risk_percentile": float(local_percentile.mean()),
                "risk_percentile_median": float(local_percentile.median()),
                "risk_percentile_max_descriptive_only": float(local_percentile.max()),
            }
        )
    table = pd.DataFrame(matches)
    weights = table["accidents"].to_numpy(dtype=float)
    observed_weighted = float(np.average(table["risk_percentile"], weights=weights))
    rng = np.random.default_rng(42)
    road_percentiles = risk_percentile.to_numpy(dtype=float)
    local_counts = table["local_link_count"].to_numpy(dtype=int)
    random_scores = np.asarray(
        [
            np.average(
                [
                    rng.choice(road_percentiles, size=count, replace=False).mean()
                    for count in local_counts
                ],
                weights=weights,
            )
            for _ in range(10000)
        ],
        dtype=float,
    )
    metrics = {
        "hotspot_clusters": int(len(table)),
        "source_records": int(hotspots["source_records"].sum()),
        "year_range": [
            int(min(year for years in hotspots["years"] for year in years)),
            int(max(year for years in hotspots["years"] for year in years)),
        ],
        "mean_risk_percentile": float(table["risk_percentile"].mean()),
        "accident_weighted_mean_risk_percentile": observed_weighted,
        "top_30pct_hotspot_count": int((table["risk_percentile"] >= 0.70).sum()),
        "top_30pct_hotspot_rate": float((table["risk_percentile"] >= 0.70).mean()),
        "matching_rule": "mean percentile of every intersecting road; nearest road only when none intersects",
        "random_road_weighted_mean": float(random_scores.mean()),
        "random_road_weighted_p95": float(np.quantile(random_scores, 0.95)),
        "one_sided_permutation_p_value": float(
            (1 + np.sum(random_scores >= observed_weighted)) / (len(random_scores) + 1)
        ),
        "permutations": 10000,
        "seed": 42,
    }
    return metrics, table


def evaluate_selected_coverage(
    roads_metric: gpd.GeoDataFrame,
    hotspots: gpd.GeoDataFrame,
) -> dict[str, object]:
    if "selected" not in roads_metric.columns:
        return {"available": False, "reason": "selected 컬럼 없음"}
    selected_mask = roads_metric["selected"].astype(bool)
    local_indices = [
        hotspot_road_indices(roads_metric, geometry)[0]
        for geometry in hotspots.geometry
    ]
    local_selected_share = np.asarray(
        [float(selected_mask.loc[indices].mean()) for indices in local_indices],
        dtype=float,
    )
    hit_arr = local_selected_share > 0
    accident_weights = hotspots["accidents"].to_numpy(dtype=float)
    observed = float(np.average(local_selected_share, weights=accident_weights))

    rng = np.random.default_rng(42)
    random_scores = []
    select_count = min(int(selected_mask.sum()), len(roads_metric))
    positions = {
        index: position for position, index in enumerate(roads_metric.index)
    }
    local_positions = [
        np.asarray([positions[index] for index in indices], dtype=int)
        for indices in local_indices
    ]
    for _ in range(10000):
        sample = rng.choice(len(roads_metric), size=select_count, replace=False)
        random_selected = np.zeros(len(roads_metric), dtype=bool)
        random_selected[sample] = True
        random_local_share = np.asarray(
            [random_selected[indices].mean() for indices in local_positions],
            dtype=float,
        )
        random_scores.append(float(np.average(random_local_share, weights=accident_weights)))
    random_arr = np.asarray(random_scores)
    overall_selected_share = float(selected_mask.mean())
    return {
        "available": True,
        "selected_roads": int(selected_mask.sum()),
        "overall_selected_road_share": overall_selected_share,
        "hotspot_hit_count": int(hit_arr.sum()),
        "hotspot_hit_rate": float(hit_arr.mean()),
        "accident_weighted_local_selected_share": observed,
        "selection_lift_vs_citywide": float(observed / max(overall_selected_share, 1e-9)),
        "random_same_count_mean": float(random_arr.mean()),
        "random_same_count_p95": float(np.quantile(random_arr, 0.95)),
        "one_sided_permutation_p_value": float(
            (1 + np.sum(random_arr >= observed)) / (len(random_arr) + 1)
        ),
        "permutations": 10000,
        "seed": 42,
        "matching_rule": "accident-weighted share of selected roads within each hotspot",
    }


def evaluate_habitual_icing_ranking(
    roads_metric: gpd.GeoDataFrame,
    source_matches: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """행정안전부 상습결빙구간에 연결된 모든 링크의 평균 위험 순위를 평가한다."""
    risk_percentile = roads_metric["risk"].rank(method="average", pct=True)
    link_to_index = {
        str(link_id): index
        for index, link_id in roads_metric["LINK_ID"].items()
    }
    rows: list[dict[str, object]] = []
    for record in source_matches.itertuples(index=False):
        requested = str(record.matched_link_ids).split("|")
        indices = [link_to_index[link_id] for link_id in requested if link_id in link_to_index]
        if not indices:
            continue
        local_percentiles = risk_percentile.loc[indices].astype(float)
        rows.append(
            {
                "segment_id": str(record.segment_id),
                "road_name": str(record.road_name),
                "source_length_km": float(record.source_length_km),
                "matched_link_count": int(len(indices)),
                "risk_percentile": float(local_percentiles.mean()),
                "risk_percentile_median": float(local_percentiles.median()),
                "risk_percentile_max_descriptive_only": float(local_percentiles.max()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return {"available": False, "reason": "매칭된 상습결빙 링크 없음"}, table

    weights = table["source_length_km"].clip(lower=0.01).to_numpy(dtype=float)
    observed = float(table["risk_percentile"].mean())
    observed_weighted = float(np.average(table["risk_percentile"], weights=weights))
    counts = table["matched_link_count"].to_numpy(dtype=int)
    road_percentiles = risk_percentile.to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    random_means: list[float] = []
    random_weighted: list[float] = []
    for _ in range(10000):
        local_scores = np.asarray(
            [
                rng.choice(road_percentiles, size=count, replace=False).mean()
                for count in counts
            ],
            dtype=float,
        )
        random_means.append(float(local_scores.mean()))
        random_weighted.append(float(np.average(local_scores, weights=weights)))
    random_values = np.asarray(random_means)
    random_weighted_values = np.asarray(random_weighted)
    return {
        "available": True,
        "source_segments": int(len(table)),
        "matched_road_links": int(counts.sum()),
        "mean_risk_percentile": observed,
        "length_weighted_mean_risk_percentile": observed_weighted,
        "top_30pct_segment_count": int((table["risk_percentile"] >= 0.70).sum()),
        "top_30pct_segment_rate": float((table["risk_percentile"] >= 0.70).mean()),
        "random_mean": float(random_values.mean()),
        "random_p95": float(np.quantile(random_values, 0.95)),
        "one_sided_permutation_p_value": float(
            (1 + np.sum(random_values >= observed)) / (len(random_values) + 1)
        ),
        "length_weighted_random_mean": float(random_weighted_values.mean()),
        "length_weighted_one_sided_p_value": float(
            (1 + np.sum(random_weighted_values >= observed_weighted))
            / (len(random_weighted_values) + 1)
        ),
        "permutations": 10000,
        "seed": 42,
        "matching_rule": "mean percentile of every road link matched within 20m",
    }, table


def save_figure(matches: pd.DataFrame, ranking: dict[str, object], coverage: dict[str, object]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(matches["cluster_id"].astype(str), matches["risk_percentile"], color="#1565c0")
    axes[0].axhline(0.70, color="#ef6c00", linestyle="--", label="상위 30% 기준")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("사고 다발지역 군집")
    axes[0].set_ylabel("매칭 도로 위험도 백분위")
    axes[0].set_title("실제 결빙사고 다발지역의 위험 순위")
    axes[0].legend()

    axes[1].scatter(matches["accidents"], matches["risk_percentile"], s=80, color="#2e7d32")
    for row in matches.itertuples(index=False):
        axes[1].annotate(str(row.cluster_id), (row.accidents, row.risk_percentile))
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("누적 사고건수")
    axes[1].set_ylabel("위험도 백분위")
    axes[1].set_title("사고건수와 독립 위험 순위")

    if coverage.get("available"):
        labels = ["제안 경로", "동일 도로 수\n무작위 평균", "무작위 95%"]
        values = [
            coverage["accident_weighted_local_selected_share"],
            coverage["random_same_count_mean"],
            coverage["random_same_count_p95"],
        ]
        axes[2].bar(labels, values, color=["#d32f2f", "#90a4ae", "#546e7a"])
        axes[2].set_ylim(0, 1)
        axes[2].set_title(f"사고건수 가중 포함률 (p={coverage['one_sided_permutation_p_value']:.3f})")
    else:
        axes[2].text(0.5, 0.5, "선정 도로 검증 불가", ha="center", va="center")
        axes[2].set_axis_off()
    plt.suptitle("독립 공간 proxy 검증 - 도로 결빙 정확도가 아님", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_simulation.png", dpi=160)
    plt.close()


def save_map(roads: gpd.GeoDataFrame, hotspots: gpd.GeoDataFrame) -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = roads.total_bounds
    center = [float((min_lat + max_lat) / 2), float((min_lon + max_lon) / 2)]
    map_obj = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
    for _, road in roads.iterrows():
        score = float(road["risk"])
        color = "#d32f2f" if score >= roads["risk"].quantile(0.7) else "#90a4ae"
        folium.GeoJson(
            road.geometry.__geo_interface__,
            style_function=lambda _, c=color: {"color": c, "weight": 2, "opacity": 0.75},
            tooltip=f"LINK {road['LINK_ID']} | 상대 위험지수 {score:.3f}",
        ).add_to(map_obj)
    for _, hotspot in hotspots.to_crs(epsg=4326).iterrows():
        point = hotspot.geometry.centroid
        folium.Marker(
            [point.y, point.x],
            tooltip=f"실제 결빙사고 다발지역 군집 {hotspot['cluster_id']}",
            popup=(
                f"관측 연도: {hotspot['years']}<br>"
                f"원자료: {hotspot['source_records']}건<br>"
                f"사고: {hotspot['accidents']:.0f}건, 사상자: {hotspot['casualties']:.0f}명"
            ),
            icon=folium.Icon(color="orange", icon="info-sign"),
        ).add_to(map_obj)
    map_obj.save(MAPS_DIR / "map_validation_simulation.html")


def main() -> dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    roads = load_roads()
    roads_metric = roads.to_crs(epsg=5186)
    hotspots = load_gangnam_accident_hotspots()
    ranking, matches = evaluate_hotspot_risk_ranking(roads_metric, hotspots)
    coverage = evaluate_selected_coverage(roads_metric, hotspots)
    if HABITUAL_ICING_MATCH_PATH.exists():
        habitual_source = pd.read_csv(HABITUAL_ICING_MATCH_PATH)
        habitual_ranking, habitual_table = evaluate_habitual_icing_ranking(
            roads_metric, habitual_source
        )
        habitual_table.to_csv(
            REPORTS_DIR / "habitual_icing_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        habitual_ranking = {
            "available": False,
            "reason": "python scripts/collect_public_data.py를 먼저 실행하세요.",
        }

    matches.to_csv(REPORTS_DIR / "validation_hotspot_matches.csv", index=False, encoding="utf-8-sig")
    road_summary = roads.drop(columns="geometry").copy()
    road_summary.to_csv(REPORTS_DIR / "validation_road_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(matches, ranking, coverage)
    save_map(roads, hotspots)

    payload = {
        "schema_version": 2,
        "mode": "independent_spatial_proxy_validation",
        "source": ACCIDENT_PATH.relative_to(PROJECT_DIR).as_posix(),
        "used_for_training_or_tuning": False,
        "actual_road_icing_labels_available": False,
        "ranking_agreement": ranking,
        "habitual_icing_segment_validation": habitual_ranking,
        "selected_road_coverage": coverage,
        "independent_findings": {
            "accident_hotspots": (
                "상대 위험 순위가 무작위보다 높아 양의 공간 일치도가 확인됨"
            ),
            "official_habitual_icing_segments": (
                "상대 위험 순위가 무작위보다 높지 않아 공간 일반화가 확인되지 않음"
            ),
            "selected_routes": (
                "사고 다발지역 내부 선택률이 무작위 동일 규모 선택보다 높지 않음"
            ),
        },
        "allowed_claim": (
            "두 독립 공간자료와 비교했으며, 사고 다발지역에서는 양의 일치도가 "
            "나왔지만 공식 상습결빙구간에서는 일반화되지 않았다."
        ),
        "prohibited_claims": [
            "도로 결빙 정확도",
            "사고 발생 확률 정확도",
            "사고 감소율 실증",
        ],
        "note": (
            "사고 다발지역은 개별 시각의 도로 결빙 정답이 아니다. "
            "지역 안의 최고 위험 도로를 고르지 않고 겹치는 모든 도로의 평균을 사용했다. "
            "공식 상습결빙구간도 학습이나 계수 조정에 사용하지 않았다. "
            "따라서 본 결과는 독립 공간 proxy 검증이며 모델의 분류 성능으로 해석하지 않는다."
        ),
        "outputs": {
            "matches": (REPORTS_DIR / "validation_hotspot_matches.csv").relative_to(PROJECT_DIR).as_posix(),
            "figure": (FIGURES_DIR / "validation_simulation.png").relative_to(PROJECT_DIR).as_posix(),
            "map": (MAPS_DIR / "map_validation_simulation.html").relative_to(PROJECT_DIR).as_posix(),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (REPORTS_DIR / "validation_simulation.json").write_text(text, encoding="utf-8")
    (REPORTS_DIR / "spatial_proxy_validation.json").write_text(text, encoding="utf-8")
    print(text)
    return payload


if __name__ == "__main__":
    main()
