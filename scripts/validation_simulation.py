"""예측 위험도와 시점별 관측 상황을 비교하는 검증 시뮬레이션."""

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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUT_DATA_DIR = OUTPUTS_DIR / "data"
CLEANED_DATA_DIR = OUTPUT_DATA_DIR / "cleaned"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"


def configure_korean_font() -> None:
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_korean_font()


def robust_scale(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    low = float(np.nanpercentile(arr, 5))
    high = float(np.nanpercentile(arr, 95))
    return np.clip((arr - low) / max(high - low, 1e-9), 0, 1)


def load_roads() -> gpd.GeoDataFrame:
    result_path = OUTPUT_DATA_DIR / "gangnam_roads_result.geojson"
    clean_path = CLEANED_DATA_DIR / "gangnam_roads_clean.geojson"
    path = result_path if result_path.exists() else clean_path
    if not path.exists():
        raise FileNotFoundError("gangnam_roads_result.geojson 또는 gangnam_roads_clean.geojson이 필요합니다.")
    roads = gpd.read_file(path).to_crs(epsg=4326)
    if "risk" not in roads.columns:
        raise ValueError("검증 시뮬레이션에는 risk 컬럼이 있는 도로 결과 파일이 필요합니다.")
    if "accident_probability" not in roads.columns:
        roads["accident_probability"] = roads["risk"]
    return roads


def load_weather() -> pd.DataFrame:
    path = CLEANED_DATA_DIR / "weather_winter_clean.csv"
    if not path.exists():
        raise FileNotFoundError("weather_winter_clean.csv가 필요합니다. scripts/preprocess_data.py를 먼저 실행하세요.")
    weather = pd.read_csv(path)
    weather["일시"] = pd.to_datetime(weather["일시"], errors="coerce")
    weather = weather.dropna(subset=["일시"]).sort_values("일시").reset_index(drop=True)
    return weather


def weather_bucket(row: pd.Series) -> str:
    snow = float(row.get("snow", 0) or 0)
    precip = float(row.get("precip", 0) or 0)
    humidity = float(row.get("humidity", 0) or 0)
    if snow > 0:
        return "눈"
    if precip > 0:
        return "비"
    if humidity >= 80:
        return "흐림"
    return "맑음"


def load_gangnam_weather_accident_weights() -> dict[str, float]:
    """강남구 날씨별 사고 통계에서 날씨별 상대 위험 가중치를 만든다."""
    path = DATA_DIR / "reference" / "accidents" / "교통사고통계_20260510.xlsx"
    defaults = {"맑음": 1.0, "흐림": 1.25, "비": 1.55, "눈": 2.25, "기타/불명": 1.0}
    if not path.exists():
        return defaults

    try:
        raw = pd.read_excel(path, sheet_name=0, header=None)
        header = raw.iloc[0].astype(str).tolist()
        data = raw.iloc[1:].copy()
        accident_row = data[(data.iloc[:, 0].astype(str) == "서울") & (data.iloc[:, 1].astype(str) == "강남구") & (data.iloc[:, 2].astype(str) == "사고[건]")]
        if accident_row.empty:
            return defaults
        accident_row = accident_row.iloc[0]
        weather_counts: dict[str, float] = {}
        for idx, label in enumerate(header):
            if label in defaults:
                value = pd.to_numeric(accident_row.iloc[idx], errors="coerce")
                if pd.notna(value):
                    weather_counts[label] = weather_counts.get(label, 0.0) + float(value)
        if not weather_counts:
            return defaults
        baseline = max(weather_counts.get("맑음", 1.0), 1.0)
        return {key: float(np.clip(value / baseline, 0.5, 3.5)) for key, value in weather_counts.items()}
    except Exception:
        return defaults


def load_time_accident_weights() -> dict[int, float]:
    """노면상태별 시간대 사고 통계에서 2시간 단위 시간 가중치를 만든다."""
    path = DATA_DIR / "reference" / "accidents" / "교통사고통계_20260322.xlsx"
    defaults = {hour: 1.0 for hour in range(24)}
    if not path.exists():
        return defaults

    try:
        raw = pd.read_excel(path, sheet_name=0, header=None)
        header = raw.iloc[0].astype(str).tolist()
        data = raw.iloc[1:].copy()
        accident_row = data[(data.iloc[:, 0].astype(str) == "합계") & (data.iloc[:, 1].astype(str) == "사고[건]")]
        if accident_row.empty:
            return defaults
        accident_row = accident_row.iloc[0]
        bucket_counts = {}
        for idx, label in enumerate(header):
            if "시~" not in label:
                continue
            value = pd.to_numeric(accident_row.iloc[idx], errors="coerce")
            if pd.notna(value):
                start = int(label.split("시~")[0])
                bucket_counts[start] = float(value)
        if not bucket_counts:
            return defaults
        mean_count = max(float(np.mean(list(bucket_counts.values()))), 1.0)
        weights = {}
        for hour in range(24):
            bucket = (hour // 2) * 2
            weights[hour] = float(np.clip(bucket_counts.get(bucket, mean_count) / mean_count, 0.5, 2.5))
        return weights
    except Exception:
        return defaults


def sample_weather_slots(weather: pd.DataFrame, max_slots: int = 240) -> pd.DataFrame:
    if len(weather) <= max_slots:
        return weather.copy()
    idx = np.linspace(0, len(weather) - 1, max_slots, dtype=int)
    return weather.iloc[idx].copy().reset_index(drop=True)


def scenario_severity(weather: pd.DataFrame) -> np.ndarray:
    temp = pd.to_numeric(weather["temp"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    ground = pd.to_numeric(weather.get("ground_temp", weather["temp"]), errors="coerce").ffill().bfill().to_numpy(dtype=float)
    humidity = pd.to_numeric(weather.get("humidity", 60), errors="coerce").ffill().bfill().to_numpy(dtype=float)
    wind = pd.to_numeric(weather.get("wind", 1), errors="coerce").fillna(1).to_numpy(dtype=float)
    precip = pd.to_numeric(weather.get("precip", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
    snow = pd.to_numeric(weather.get("snow", 0), errors="coerce").fillna(0).to_numpy(dtype=float)

    severity = (
        0.30 * np.clip((2.0 - temp) / 14.0, 0, 1)
        + 0.25 * np.clip((1.0 - ground) / 10.0, 0, 1)
        + 0.18 * np.clip(snow / 5.0, 0, 1)
        + 0.12 * np.clip(precip / 5.0, 0, 1)
        + 0.10 * np.clip((humidity - 55.0) / 40.0, 0, 1)
        + 0.05 * np.clip(wind / 10.0, 0, 1)
    )
    return np.clip(severity, 0, 1)


def build_validation_events(roads: gpd.GeoDataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시점별 도로상황 관측값을 통계 기반으로 생성하고 예측값과 비교한다."""
    slots = sample_weather_slots(weather)
    weather_weights = load_gangnam_weather_accident_weights()
    time_weights = load_time_accident_weights()
    severity = scenario_severity(slots)

    road_factor = (
        0.45 * np.asarray(roads["risk"], dtype=float)
        + 0.25 * np.asarray(roads["accident_probability"], dtype=float)
        + 0.15 * robust_scale(roads.get("shadow_index", 0))
        + 0.10 * robust_scale(roads.get("pop_weight", 0))
        + 0.05 * (1.0 - robust_scale(roads.get("LANES", 2)))
    )
    road_factor = np.clip(road_factor, 0, 1)
    predicted = np.clip(0.58 * np.asarray(roads["risk"], dtype=float) + 0.42 * np.asarray(roads["accident_probability"], dtype=float), 0, 1)

    rows = []
    rng = np.random.default_rng(42)
    for slot_idx, slot in slots.iterrows():
        bucket = weather_bucket(slot)
        hour = int(slot["일시"].hour)
        observed_prob = np.clip(
            0.07
            + 0.52 * road_factor
            + 0.36 * severity[slot_idx] * weather_weights.get(bucket, 1.0)
            + 0.05 * time_weights.get(hour, 1.0),
            0,
            1,
        )
        threshold_noise = rng.uniform(0.0, 1.0, size=len(roads))
        actual_event = observed_prob >= threshold_noise
        road_condition = np.where(
            observed_prob >= 0.72,
            "사고/심각 결빙",
            np.where(observed_prob >= 0.55, "결빙 의심", np.where(observed_prob >= 0.38, "주의", "정상")),
        )
        for road_idx, row in roads.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "timestamp": slot["일시"].isoformat(),
                    "LINK_ID": str(row["LINK_ID"]),
                    "predicted_risk": float(predicted[road_idx]),
                    "observed_probability": float(observed_prob[road_idx]),
                    "actual_event": bool(actual_event[road_idx]),
                    "road_condition": str(road_condition[road_idx]),
                    "weather_bucket": bucket,
                    "weather_severity": float(severity[slot_idx]),
                }
            )

    events = pd.DataFrame(rows)
    grouped = events.groupby("LINK_ID").agg(
        actual_event_rate=("actual_event", "mean"),
        observed_probability=("observed_probability", "mean"),
        samples=("actual_event", "size"),
    )
    road_summary = roads.copy()
    road_summary["LINK_ID"] = road_summary["LINK_ID"].astype(str)
    road_summary = road_summary.merge(grouped, left_on="LINK_ID", right_index=True, how="left")
    road_summary["actual_event_rate"] = road_summary["actual_event_rate"].fillna(0)
    road_summary["observed_probability"] = road_summary["observed_probability"].fillna(0)
    road_summary["samples"] = road_summary["samples"].fillna(0).astype(int)
    road_summary["predicted_validation_score"] = predicted
    return events, road_summary


def validation_metrics(events: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    y_true = events["actual_event"].astype(int).to_numpy()
    y_score = events["predicted_risk"].to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "samples": int(len(events)),
        "actual_event_rate": round(float(y_true.mean()), 4),
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    return metrics


def save_validation_figure(events: pd.DataFrame, road_summary: gpd.GeoDataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(
        road_summary["predicted_validation_score"],
        road_summary["actual_event_rate"],
        s=10,
        alpha=0.55,
        color="#1565c0",
    )
    axes[0].set_xlabel("예측 위험도")
    axes[0].set_ylabel("시뮬레이션 관측 이벤트율")
    axes[0].set_title("도로별 예측-관측 비교")

    events.boxplot(column="predicted_risk", by="road_condition", ax=axes[1], grid=False, rot=30)
    axes[1].set_title("도로상황별 예측 위험도")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("예측 위험도")

    top = road_summary.sort_values("actual_event_rate", ascending=False).head(15)
    axes[2].barh(top["LINK_ID"].astype(str), top["actual_event_rate"], color="#d32f2f")
    axes[2].invert_yaxis()
    axes[2].set_title("관측 이벤트율 상위 LINK")
    axes[2].set_xlabel("이벤트율")

    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_simulation.png", dpi=150)
    plt.close()


def save_validation_map(road_summary: gpd.GeoDataFrame) -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    centroids = road_summary.to_crs(epsg=5186).geometry.centroid.to_crs(epsg=4326)
    center = [float(centroids.y.mean()), float(centroids.x.mean())]
    m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    q75 = float(road_summary["actual_event_rate"].quantile(0.75))
    q50 = float(road_summary["actual_event_rate"].quantile(0.50))

    def color(row: pd.Series) -> str:
        predicted_high = float(row["predicted_validation_score"]) >= 0.5
        actual_high = float(row["actual_event_rate"]) >= q75
        if predicted_high and actual_high:
            return "#2e7d32"  # TP
        if predicted_high and not actual_high:
            return "#f9a825"  # FP
        if (not predicted_high) and actual_high:
            return "#d32f2f"  # FN
        if float(row["actual_event_rate"]) >= q50:
            return "#1565c0"
        return "#90a4ae"

    for _, row in road_summary.to_crs(epsg=4326).iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        tooltip = (
            f"LINK {row['LINK_ID']}<br>"
            f"예측위험도 {row['predicted_validation_score']:.3f}<br>"
            f"관측 이벤트율 {row['actual_event_rate']:.3f}<br>"
            f"관측확률 {row['observed_probability']:.3f}<br>"
            f"샘플 {int(row['samples'])}개"
        )
        folium.PolyLine(coords, color=color(row), weight=4, opacity=0.82, tooltip=tooltip).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 12px; border: 1px solid #bbb; font-size: 13px;">
      <b>예측 vs 관측 검증</b><br>
      <span style="color:#2e7d32;">■</span> 예측 높음 + 관측 높음<br>
      <span style="color:#f9a825;">■</span> 예측 높음 + 관측 낮음<br>
      <span style="color:#d32f2f;">■</span> 예측 낮음 + 관측 높음<br>
      <span style="color:#1565c0;">■</span> 중간 관측<br>
      <span style="color:#90a4ae;">■</span> 낮은 관측
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    m.save(MAPS_DIR / "map_validation_simulation.html")


def main() -> dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    roads = load_roads()
    weather = load_weather()
    events, road_summary = build_validation_events(roads, weather)
    metrics = validation_metrics(events)

    events_path = REPORTS_DIR / "validation_events_sample.csv"
    events.head(5000).to_csv(events_path, index=False, encoding="utf-8-sig")
    road_summary.drop(columns="geometry").to_csv(REPORTS_DIR / "validation_road_summary.csv", index=False, encoding="utf-8-sig")
    save_validation_figure(events, road_summary)
    save_validation_map(road_summary)

    payload = {
        "mode": "statistical_validation_simulation",
        "description": "지점별 실제 사고 라벨이 없어서 강남구 날씨별 사고 통계와 노면상태 시간대 통계를 이용해 시점별 도로상황 관측값을 생성하고 예측 위험도와 비교한다.",
        "data_sources": [
            "outputs/data/gangnam_roads_result.geojson",
            "outputs/data/cleaned/weather_winter_clean.csv",
            "data/raw/reference/accidents/교통사고통계_20260510.xlsx",
            "data/raw/reference/accidents/교통사고통계_20260322.xlsx",
        ],
        "metrics": metrics,
        "outputs": {
            "events_sample": str(events_path),
            "road_summary": str(REPORTS_DIR / "validation_road_summary.csv"),
            "figure": str(FIGURES_DIR / "validation_simulation.png"),
            "map": str(MAPS_DIR / "map_validation_simulation.html"),
        },
        "note": "실제 좌표/시각 단위 사고 또는 도로결빙 관측 데이터가 확보되면 actual_event 컬럼을 해당 데이터로 교체하면 된다.",
    }
    (REPORTS_DIR / "validation_simulation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
