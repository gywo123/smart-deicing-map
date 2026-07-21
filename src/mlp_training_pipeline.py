"""실측 노면온도 MLP와 제설 최적화 파이프라인 연결 모듈."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .ground_temperature_mlp import GroundTemperatureMLPConfig
from .road_surface_temperature_mlp import (
    ROAD_FEATURES,
    apply_mlp_road_risk,
    train_road_surface_temperature_mlp,
)


def load_enriched_weather(project_dir: Path) -> pd.DataFrame:
    """전처리된 MLP용 기상 피처를 읽고 스키마를 검증한다."""
    path = project_dir / "outputs" / "data" / "cleaned" / "weather_winter_clean.csv"
    if not path.exists():
        raise FileNotFoundError("scripts/preprocess_data.py를 먼저 실행해야 합니다.")
    weather = pd.read_csv(path)
    weather["일시"] = pd.to_datetime(weather["일시"], errors="coerce")
    required = {
        "일시",
        "ground_temp",
        *[
            feature
            for feature in ROAD_FEATURES
            if feature != "ground_temp_now" and not feature.startswith("road_surface_temp_")
        ],
    }
    missing = sorted(required - set(weather.columns))
    if missing:
        raise ValueError(
            "기상 정제본이 이전 형식입니다. scripts/preprocess_data.py를 다시 실행하세요. "
            f"누락 컬럼: {missing}"
        )
    return weather.dropna(subset=["일시"]).sort_values("일시").reset_index(drop=True)


def load_road_weather(project_dir: Path) -> pd.DataFrame:
    """기상청 분 단위 원지 관측의 시간 집계 정제본을 읽는다."""
    path = project_dir / "outputs" / "data" / "cleaned" / "kma_road_weather_hourly.csv"
    if not path.exists():
        raise FileNotFoundError(
            "실측 노면온도 자료가 없습니다. KMA_API_KEY를 설정하고 "
            "scripts/collect_public_data.py를 먼저 실행하세요."
        )
    frame = pd.read_csv(path)
    required = {"hour", "road_surface_temp", "station_id", "station_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"노면기상 정제본 필수 컬럼 누락: {missing}")
    frame["hour"] = pd.to_datetime(frame["hour"], errors="coerce")
    frame["road_surface_temp"] = pd.to_numeric(
        frame["road_surface_temp"], errors="coerce"
    )
    return frame.dropna(subset=["hour", "road_surface_temp"]).sort_values("hour")


def train_mlp_pipeline(
    pipeline: Any,
    config: GroundTemperatureMLPConfig | None = None,
) -> dict[str, object]:
    """실제 도로관측 노면온도 MLP를 학습하고 지도·경로 산출물을 갱신한다."""
    project_dir = Path(pipeline.BASE_DIR)
    weather = load_enriched_weather(project_dir)
    road_weather = load_road_weather(project_dir)

    roads = pipeline.load_roads()
    roads = pipeline.load_buildings_and_shadow(roads)
    roads = pipeline.load_population_and_map(roads)

    model_path = Path(pipeline.MODELS_DIR) / "road_surface_temperature_mlp.pt"
    bundle, metrics = train_road_surface_temperature_mlp(
        weather,
        road_weather,
        model_path=model_path,
        reports_dir=Path(pipeline.REPORTS_DIR),
        figures_dir=Path(pipeline.FIGURES_DIR),
        config=config,
    )
    roads = apply_mlp_road_risk(roads, weather, road_weather, bundle)
    roads = pipeline.calculate_priority(roads)
    roads, selected = pipeline.hybrid_bmc_knapsack_optimize(roads, budget_ratio=0.4)
    vehicle_route_roads, vehicle_route_coords, vehicle_meta = pipeline.vrp_route(roads, selected)

    route_coords_flat: list[tuple[float, float]] = []
    for coords in vehicle_route_coords:
        route_coords_flat.extend(coords)

    results = pipeline.run_simulation(roads)
    pipeline.create_maps(
        roads,
        route_coords_flat,
        vehicle_route_roads,
        vehicle_route_coords,
        vehicle_meta,
        results,
    )
    pipeline.generate_navigation_report(
        roads,
        vehicle_route_roads,
        vehicle_route_coords,
        vehicle_meta,
        results,
    )

    export = roads.drop(columns=["risk_bin"], errors="ignore").copy()
    export.to_file(Path(pipeline.OUTPUT_DATA_DIR) / "gangnam_roads_result.geojson", driver="GeoJSON")

    summary = {
        "mode": "observed_road_surface_temperature_mlp_with_relative_road_risk",
        "model_path": model_path.relative_to(project_dir).as_posix(),
        "model_version": metrics["model_version"],
        "target": metrics["target"],
        "pseudo_label_used": False,
        "features": metrics["features"],
        "test_period": metrics["test_period"],
        "test_rows": metrics["test_rows"],
        "regression_metrics": metrics["regression"],
        "freezing_metrics": metrics["freezing"],
        "actual_road_surface_temperature_validated": True,
        "road_icing_state_validated": False,
        "selected_roads": len(selected),
        "road_risk_weather_basis": str(roads["risk_weather_basis"].iloc[0]),
        "road_risk_weather_rows": int(roads["risk_weather_rows"].iloc[0]),
        "road_risk_event_rows": int(roads["risk_event_rows"].iloc[0]),
        "interpretation": (
            "MLP 성능은 강남 인접 원지 관측소의 실측 노면온도 예측 성능이다. "
            "도로 risk는 예측 노면온도와 수분·건물 그림자를 결합한 상대 제설 "
            "우선순위이며 강남 전체 도로의 절대 결빙 확률이 아니다."
        ),
    }
    summary_path = Path(pipeline.REPORTS_DIR) / "mlp_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


__all__ = [
    "ROAD_FEATURES",
    "GroundTemperatureMLPConfig",
    "load_enriched_weather",
    "load_road_weather",
    "train_mlp_pipeline",
]
