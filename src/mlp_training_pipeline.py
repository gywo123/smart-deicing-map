"""MLP 학습/예측/산출물 생성을 분리한 재사용 모듈."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from .torch_lightning_risk_model import AccidentRiskLightningModule, IcingRiskLightningModule


FEATURES = [
    "shadow_index",
    "pop_weight",
    "LENGTH",
    "LANES",
    "road_width",
    "area",
    "deicing_cost",
]


@dataclass(frozen=True)
class TrainerDeviceConfig:
    accelerator: str
    devices: int
    description: str


@dataclass(frozen=True)
class MLPTrainingConfig:
    max_epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 2e-3
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.15
    threshold: float = 0.5
    validation_ratio: float = 0.2
    random_state: int = 42


def get_trainer_device_config() -> TrainerDeviceConfig:
    """CUDA 사용 가능 여부에 맞춰 Lightning 학습 장치를 고른다."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        return TrainerDeviceConfig(
            accelerator="gpu",
            devices=1,
            description=f"GPU - {torch.cuda.get_device_name(0)}",
        )
    return TrainerDeviceConfig(accelerator="cpu", devices=1, description="CPU")


def prepare_icing_features(roads: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, list[float] | float]]:
    """도로 GeoDataFrame에서 MLP 입력 feature와 결빙 pseudo-label을 만든다."""
    missing = [feature for feature in FEATURES if feature not in roads.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if "risk" not in roads.columns:
        raise ValueError("roads must contain a risk column")

    x = roads[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    y_cont = roads["risk"].astype(float).clip(0, 1).to_numpy()

    threshold = float(np.quantile(y_cont, 0.60))
    y = (y_cont >= threshold).astype(np.float32)

    mean = x.mean(axis=0).to_numpy(dtype=np.float32)
    std = x.std(axis=0).replace(0, 1).to_numpy(dtype=np.float32)
    x_norm = ((x.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "label_threshold": threshold}
    return x_norm, y, scaler


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    """numpy feature/label 배열을 PyTorch DataLoader로 변환한다."""
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """확률 예측값에서 이진 분류 평가 지표를 계산한다."""
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 4)
    return metrics


def train_binary_mlp(
    module_cls: type[IcingRiskLightningModule],
    x: np.ndarray,
    y: np.ndarray,
    config: MLPTrainingConfig,
) -> tuple[IcingRiskLightningModule, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """하나의 이진 MLP를 학습하고 전체 데이터 확률 예측값을 반환한다."""
    device_config = get_trainer_device_config()
    print(f"학습 장치: {device_config.description}")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=config.validation_ratio,
        random_state=config.random_state,
        stratify=y,
    )

    pos_weight = float((len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0))
    module = module_cls(
        input_dim=len(FEATURES),
        lr=config.learning_rate,
        pos_weight=max(pos_weight, 1e-3),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        threshold=config.threshold,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=device_config.accelerator,
        devices=device_config.devices,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(
        module,
        make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True),
        make_loader(x_val, y_val, batch_size=config.batch_size),
    )

    module.eval()
    device = module.device
    with torch.no_grad():
        val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
        all_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        val_probs = module(val_tensor).cpu().numpy()
        all_probs = module(all_tensor).cpu().numpy()

    return module, all_probs, val_probs, y_val, classification_metrics(y_val, val_probs)


def build_accident_pseudo_labels(
    roads: gpd.GeoDataFrame,
    icing_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """결빙 확률과 도로 feature로 사고확률 MLP용 pseudo-label을 만든다."""
    pop = roads["pop_weight"].astype(float).clip(0, 1).to_numpy()
    shadow = roads["shadow_index"].astype(float).clip(0, 1).to_numpy()
    lanes = roads["LANES"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(2).to_numpy()
    length = roads["LENGTH"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

    lane_risk = np.clip((3 - np.minimum(lanes, 3)) / 2, 0, 1)
    length_risk = np.clip(length / max(float(np.nanpercentile(length, 95)), 1.0), 0, 1)
    accident_score = np.clip(
        0.45 * icing_probs
        + 0.30 * pop
        + 0.15 * shadow
        + 0.06 * length_risk
        + 0.04 * lane_risk,
        0,
        1,
    )
    threshold = float(np.quantile(accident_score, 0.60))
    y = (accident_score >= threshold).astype(np.float32)
    return y, accident_score, threshold


def soften_probabilities(probs: np.ndarray, temperature: float = 0.65) -> np.ndarray:
    """MLP 확률이 0/1로 과하게 붙는 현상을 줄여 지도 색상 쏠림을 완화한다."""
    clipped = np.clip(probs.astype(float), 1e-4, 1 - 1e-4)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-logits * temperature))


def robust_scale(values: Any, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
    """이상치 한두 개가 전체 점수를 지배하지 않게 분위수 기반으로 0~1 정규화한다."""
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0)
    low = float(np.nanpercentile(arr, lower_q))
    high = float(np.nanpercentile(arr, upper_q))
    return np.clip((arr - low) / max(high - low, 1e-6), 0, 1)


def build_physical_icing_baseline(roads: gpd.GeoDataFrame) -> np.ndarray:
    """그림자, 도로 폭/길이, 유동인구를 이용한 연속형 결빙 위험 baseline을 만든다."""
    shadow = robust_scale(roads["shadow_index"].astype(float).to_numpy())
    pop = robust_scale(roads["pop_weight"].astype(float).to_numpy())
    length = robust_scale(roads["LENGTH"].astype(float).to_numpy())
    lanes = roads["LANES"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(2).to_numpy()
    width = roads["road_width"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(8).to_numpy()

    lane_risk = np.clip((3.5 - np.minimum(lanes, 3.5)) / 2.5, 0, 1)
    narrow_road_risk = 1.0 - robust_scale(width)
    baseline = (
        0.42 * shadow
        + 0.18 * pop
        + 0.16 * length
        + 0.14 * lane_risk
        + 0.10 * narrow_road_risk
    )
    return np.clip(0.05 + 0.90 * baseline, 0.05, 0.95)


def save_mlp_checkpoint(
    path: Path,
    module: IcingRiskLightningModule,
    scaler: dict[str, list[float] | float],
    metrics: dict[str, float],
    config: MLPTrainingConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """학습된 MLP와 feature metadata를 하나의 pt 파일로 저장한다."""
    payload: dict[str, Any] = {
        "state_dict": module.state_dict(),
        "input_dim": len(FEATURES),
        "hidden_dims": config.hidden_dims,
        "dropout": config.dropout,
        "features": FEATURES,
        "scaler": scaler,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def apply_risk_outputs(
    roads: gpd.GeoDataFrame,
    icing_probs: np.ndarray,
    accident_probs: np.ndarray,
    accident_baseline: np.ndarray,
    pipeline: Any,
) -> gpd.GeoDataFrame:
    """MLP 예측값을 도로 데이터에 반영하고 최적화용 점수를 갱신한다."""
    roads = roads.copy()
    physical_baseline = build_physical_icing_baseline(roads)
    calibrated_icing = 0.40 * soften_probabilities(np.clip(icing_probs, 0, 1), temperature=0.45) + 0.60 * physical_baseline
    calibrated_accident = (
        0.70 * soften_probabilities(np.clip(accident_probs, 0, 1))
        + 0.30 * np.clip(accident_baseline, 0, 1)
    )

    roads["risk"] = np.clip(calibrated_icing, 0, 1)
    roads["accident_probability"] = np.clip(calibrated_accident, 0, 1)
    roads["accident_pseudo_baseline"] = accident_baseline
    roads = pipeline.calculate_priority(roads)

    pop_weight = roads["pop_weight"].astype(float).clip(0, 1)
    roads["safety_gain_score"] = np.clip(
        (0.45 * roads["risk"] + 0.55 * roads["accident_probability"]) * (0.7 + 0.3 * pop_weight),
        0,
        1,
    )
    roads["priority_score"] = np.clip(
        (0.55 * roads["risk"] + 0.45 * roads["accident_probability"]) * (0.7 + 0.3 * pop_weight),
        0,
        1,
    )
    return roads


def save_training_figure(
    figure_path: Path,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    val_accident_probs: np.ndarray,
    y_accident_val: np.ndarray,
    metrics: dict[str, float],
    accident_metrics: dict[str, float],
) -> None:
    """결빙/사고확률 MLP 검증 결과를 그림으로 저장한다."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(val_probs[y_val == 0], bins=30, alpha=0.7, label="normal", color="#1565c0")
    axes[0].hist(val_probs[y_val == 1], bins=30, alpha=0.7, label="high icing", color="#d32f2f")
    axes[0].set_title("MLP Risk Probability Distribution")
    axes[0].set_xlabel("Predicted probability")
    axes[0].legend()

    axes[1].hist(val_accident_probs[y_accident_val == 0], bins=30, alpha=0.7, label="normal", color="#2e7d32")
    axes[1].hist(val_accident_probs[y_accident_val == 1], bins=30, alpha=0.7, label="high accident", color="#ef6c00")
    axes[1].set_title("MLP Accident Probability Distribution")
    axes[1].set_xlabel("Predicted probability")
    axes[1].legend()

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    axes[2].bar(metric_labels, [metrics.get(key, 0) for key in metric_keys], label="Icing MLP")
    axes[2].plot(
        metric_labels,
        [accident_metrics.get(key, 0) for key in metric_keys],
        marker="o",
        color="#ef6c00",
        label="Accident MLP",
    )
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].set_title("MLP Validation Metrics")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def resolve_roads_geojson_path(pipeline: Any) -> Path:
    """현재 루트 구조의 processed GeoJSON을 찾고, 없으면 이전 위치를 fallback으로 쓴다."""
    current_path = Path(pipeline.OUTPUT_DATA_DIR) / "gangnam_roads_result.geojson"
    if current_path.exists():
        return current_path

    legacy_path = Path(pipeline.BASE_DIR) / "src" / "outputs" / "data" / "gangnam_roads_result.geojson"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        "gangnam_roads_result.geojson을 찾을 수 없습니다. "
        f"확인 위치: {current_path}, {legacy_path}"
    )


def refresh_shadow_index_if_possible(roads: gpd.GeoDataFrame, pipeline: Any) -> tuple[gpd.GeoDataFrame, str]:
    """건물 원본 데이터가 실제 파일이면 shadow_utils 기반 그림자 지수를 다시 계산한다."""
    if not hasattr(pipeline, "load_buildings_and_shadow"):
        return roads, "existing_shadow_index"

    try:
        updated_roads = pipeline.load_buildings_and_shadow(roads)
    except FileNotFoundError as exc:
        print(f"건물 그림자 재계산 건너뜀: {exc}")
        return roads, "existing_shadow_index_raw_buildings_missing"
    except (KeyError, ValueError) as exc:
        print(f"건물 그림자 재계산 건너뜀: 건물 데이터 형식 확인 필요 - {exc}")
        return roads, "existing_shadow_index_building_schema_issue"

    return updated_roads, "recalculated_from_buildings"


def train_mlp_pipeline(pipeline: Any, config: MLPTrainingConfig | None = None) -> dict[str, object]:
    """결빙 MLP와 사고확률 MLP를 학습하고 지도/리포트 산출물을 갱신한다."""
    config = config or MLPTrainingConfig()
    pl.seed_everything(config.random_state, workers=True)

    try:
        roads_path = resolve_roads_geojson_path(pipeline)
    except FileNotFoundError:
        if not hasattr(pipeline, "main"):
            raise
        print("processed GeoJSON이 없어 기본 파이프라인을 먼저 실행합니다.")
        pipeline.main()
        roads_path = resolve_roads_geojson_path(pipeline)

    roads = gpd.read_file(roads_path).to_crs(epsg=4326)
    roads, shadow_source = refresh_shadow_index_if_possible(roads, pipeline)
    x, y, scaler = prepare_icing_features(roads)

    module, all_probs, val_probs, y_val, metrics = train_binary_mlp(IcingRiskLightningModule, x, y, config)
    model_path = Path(pipeline.MODELS_DIR) / "icing_risk_mlp.pt"
    save_mlp_checkpoint(model_path, module, scaler, metrics, config)

    y_accident, accident_baseline, accident_threshold = build_accident_pseudo_labels(roads, np.clip(all_probs, 0, 1))
    accident_module, all_accident_probs, val_accident_probs, y_accident_val, accident_metrics = train_binary_mlp(
        AccidentRiskLightningModule,
        x,
        y_accident,
        config,
    )

    accident_model_path = Path(pipeline.MODELS_DIR) / "accident_risk_mlp.pt"
    save_mlp_checkpoint(
        accident_model_path,
        accident_module,
        scaler,
        accident_metrics,
        config,
        extra={"pseudo_label_threshold": accident_threshold},
    )

    roads = apply_risk_outputs(roads, all_probs, all_accident_probs, accident_baseline, pipeline)
    roads, selected = pipeline.hybrid_bmc_knapsack_optimize(roads, budget_ratio=0.4)
    vehicle_route_roads, vehicle_route_coords, vehicle_meta = pipeline.vrp_route(roads, selected, n_vehicles=4)

    route_coords_flat: list[tuple[float, float]] = []
    for coords in vehicle_route_coords:
        route_coords_flat.extend(coords)

    results = pipeline.run_simulation(roads)
    pipeline.create_maps(roads, route_coords_flat, vehicle_route_roads, vehicle_route_coords, vehicle_meta, results)
    pipeline.generate_navigation_report(roads, vehicle_route_roads, vehicle_route_coords, vehicle_meta, results)

    export = roads.drop(columns=["risk_bin"], errors="ignore").copy()
    export.to_file(Path(pipeline.OUTPUT_DATA_DIR) / "gangnam_roads_result.geojson", driver="GeoJSON")

    save_training_figure(
        Path(pipeline.FIGURES_DIR) / "mlp_model_evaluation.png",
        val_probs,
        y_val,
        val_accident_probs,
        y_accident_val,
        metrics,
        accident_metrics,
    )

    summary = {
        "mode": "local_icing_and_accident_mlp_from_processed_geojson",
        "model_paths": {
            "icing": str(model_path),
            "accident": str(accident_model_path),
        },
        "features": FEATURES,
        "icing_pseudo_label_threshold": round(float(scaler["label_threshold"]), 4),
        "accident_pseudo_label_threshold": round(float(accident_threshold), 4),
        "icing_metrics": metrics,
        "accident_metrics": accident_metrics,
        "selected_roads": len(selected),
        "shadow_source": shadow_source,
        "maps_dir": pipeline.MAPS_DIR,
    }
    summary_path = Path(pipeline.REPORTS_DIR) / "mlp_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
