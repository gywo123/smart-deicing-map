"""기상청 도로관측 노면온도를 학습하는 시간 홀드아웃 residual MLP."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

from .ground_temperature_mlp import (
    FEATURES as ASOS_FEATURES,
    GroundTemperatureMLPConfig,
    _bootstrap_intervals,
    _fit_freeze_calibrator,
    _freeze_metrics,
    _loader,
    _regression_metrics,
    _trainer_device,
    freeze_probability,
    weather_moisture_index,
)
from .torch_lightning_risk_model import GroundTemperatureLightningModule


MODEL_VERSION = "road_surface_temperature_residual_mlp_v2"
MANIFEST_NAME = "road_surface_model_manifest.json"
FORECAST_HORIZON_HOURS = 3
ROAD_FEATURES = [
    *ASOS_FEATURES,
    "road_surface_temp_now",
    "road_surface_temp_lag_1h",
    "road_surface_temp_lag_2h",
    "road_surface_temp_lag_3h",
    "road_surface_temp_lag_6h",
    "road_surface_temp_mean_3h",
    "road_surface_temp_mean_6h",
    "road_surface_temp_change_1h",
    "road_surface_temp_change_3h",
]


def prepare_road_surface_features(
    weather: pd.DataFrame,
    road_weather: pd.DataFrame,
) -> pd.DataFrame:
    """ASOS 시간 피처와 강남 인접 도로기상 관측을 같은 시각으로 결합한다."""
    weather_frame = weather.copy()
    weather_frame["일시"] = pd.to_datetime(weather_frame["일시"], errors="coerce")
    weather_frame["ground_temp_now"] = pd.to_numeric(
        weather_frame["ground_temp"], errors="coerce"
    )

    road = road_weather.copy()
    road["hour"] = pd.to_datetime(road["hour"], errors="coerce")
    road["road_surface_temp"] = pd.to_numeric(
        road["road_surface_temp"], errors="coerce"
    )
    road = road.dropna(subset=["hour", "road_surface_temp"]).sort_values("hour")
    for lag in (1, 2, 3, 6):
        contiguous = (road["hour"] - road["hour"].shift(lag)).eq(
            pd.Timedelta(hours=lag)
        )
        road[f"road_surface_temp_lag_{lag}h"] = road["road_surface_temp"].shift(
            lag
        ).where(contiguous)
    for window in (3, 6):
        contiguous = (road["hour"] - road["hour"].shift(window - 1)).eq(
            pd.Timedelta(hours=window - 1)
        )
        road[f"road_surface_temp_mean_{window}h"] = (
            road["road_surface_temp"]
            .rolling(window, min_periods=window)
            .mean()
            .where(contiguous)
        )
    road["road_surface_temp_change_1h"] = (
        road["road_surface_temp"] - road["road_surface_temp_lag_1h"]
    )
    road["road_surface_temp_change_3h"] = (
        road["road_surface_temp"] - road["road_surface_temp_lag_3h"]
    )
    road = road.rename(columns={"road_surface_temp": "road_surface_temp_now"})
    columns = [
        "hour",
        "road_surface_temp_now",
        "road_surface_temp_lag_1h",
        "road_surface_temp_lag_2h",
        "road_surface_temp_lag_3h",
        "road_surface_temp_lag_6h",
        "road_surface_temp_mean_3h",
        "road_surface_temp_mean_6h",
        "road_surface_temp_change_1h",
        "road_surface_temp_change_3h",
        "station_id",
        "station_name",
        "road_name",
        "lat",
        "lon",
    ]
    available = [column for column in columns if column in road.columns]
    merged = weather_frame.merge(
        road[available],
        left_on="일시",
        right_on="hour",
        how="inner",
        validate="one_to_one",
    )
    return merged.drop(columns=["hour"], errors="ignore").sort_values("일시")


def build_road_surface_forecast_dataset(
    weather: pd.DataFrame,
    road_weather: pd.DataFrame,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """발행 시각의 관측 피처와 정확히 horizon 뒤 실측 노면온도를 결합한다."""
    if horizon_hours <= 0:
        raise ValueError("예측 horizon은 1시간 이상이어야 합니다.")
    issue = prepare_road_surface_features(weather, road_weather)
    missing = sorted(set(ROAD_FEATURES) - set(issue.columns))
    if missing:
        raise ValueError(f"노면온도 MLP 입력 컬럼 누락: {missing}")
    issue["issue_time"] = issue["일시"]
    issue["valid_time"] = issue["issue_time"] + pd.to_timedelta(
        horizon_hours, unit="h"
    )
    issue = issue[["issue_time", "valid_time", *ROAD_FEATURES]].dropna()

    target = road_weather[["hour", "road_surface_temp"]].copy()
    target["hour"] = pd.to_datetime(target["hour"], errors="coerce")
    target["road_surface_temp"] = pd.to_numeric(
        target["road_surface_temp"], errors="coerce"
    )
    target = target.rename(
        columns={
            "hour": "valid_time",
            "road_surface_temp": "target_road_surface_temp",
        }
    ).dropna()
    target = target.drop_duplicates("valid_time", keep="last")
    dataset = issue.merge(target, on="valid_time", how="inner", validate="many_to_one")
    dataset["target_delta"] = (
        dataset["target_road_surface_temp"] - dataset["road_surface_temp_now"]
    )
    return dataset.sort_values("issue_time").reset_index(drop=True)


def split_temporal_months(
    dataset: pd.DataFrame,
    minimum_rows_per_month: int = 24 * 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """최신 두 개의 충분한 월을 보정·테스트로 분리한다."""
    periods = dataset["valid_time"].dt.to_period("M")
    counts = periods.value_counts().sort_index()
    eligible = counts[counts >= minimum_rows_per_month].index.tolist()
    if len(eligible) < 3:
        raise ValueError("노면온도 시간 홀드아웃에는 충분한 관측이 있는 월이 3개 이상 필요합니다.")
    calibration_period = eligible[-2]
    test_period = eligible[-1]
    development = dataset[periods < calibration_period].copy()
    calibration = dataset[periods == calibration_period].copy()
    test = dataset[periods == test_period].copy()
    if development.empty or calibration.empty or test.empty:
        raise ValueError("노면온도 시간 분할 결과에 빈 데이터가 있습니다.")
    return development, calibration, test, {
        "calibration_period": str(calibration_period),
        "test_period": str(test_period),
    }


def _fit_scalers(frame: pd.DataFrame) -> dict[str, Any]:
    x = frame[ROAD_FEATURES].to_numpy(dtype=np.float32)
    y = frame["target_delta"].to_numpy(dtype=np.float32)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    return {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": float(y.mean()),
        "y_std": max(float(y.std()), 1e-6),
    }


def _transform(frame: pd.DataFrame, scaler: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[ROAD_FEATURES].to_numpy(dtype=np.float32)
    x = (x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(
        scaler["x_std"], dtype=np.float32
    )
    y = frame["target_delta"].to_numpy(dtype=np.float32)
    y = (y - float(scaler["y_mean"])) / float(scaler["y_std"])
    return x.astype(np.float32), y.astype(np.float32)


def _fit_with_validation(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    config: GroundTemperatureMLPConfig,
) -> tuple[GroundTemperatureLightningModule, dict[str, Any], int]:
    scaler = _fit_scalers(development)
    x_train, y_train = _transform(development, scaler)
    x_val, y_val = _transform(validation, scaler)
    accelerator, devices, description = _trainer_device()
    print(f"노면온도 MLP 학습 장치: {description}")
    module = GroundTemperatureLightningModule(
        input_dim=len(ROAD_FEATURES),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    with tempfile.TemporaryDirectory(prefix="icezero_road_mlp_") as temp_dir:
        checkpoint = ModelCheckpoint(
            dirpath=temp_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            min_delta=1e-4,
        )
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=False,
            callbacks=[checkpoint, early_stop],
            enable_model_summary=False,
            enable_progress_bar=False,
            deterministic=True,
        )
        trainer.fit(
            module,
            _loader(x_train, y_train, config.batch_size, shuffle=True),
            _loader(x_val, y_val, config.batch_size, shuffle=False),
        )
        payload = torch.load(
            checkpoint.best_model_path, map_location="cpu", weights_only=False
        )
        best_epochs = int(payload["epoch"]) + 1
        best_module = GroundTemperatureLightningModule.load_from_checkpoint(
            checkpoint.best_model_path,
            map_location="cpu",
        )
    return best_module.eval(), scaler, best_epochs


def _fit_fixed_epochs(
    frame: pd.DataFrame,
    config: GroundTemperatureMLPConfig,
    epochs: int,
) -> tuple[GroundTemperatureLightningModule, dict[str, Any]]:
    scaler = _fit_scalers(frame)
    x, y = _transform(frame, scaler)
    accelerator, devices, _ = _trainer_device()
    module = GroundTemperatureLightningModule(
        input_dim=len(ROAD_FEATURES),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        use_plateau_scheduler=False,
    )
    trainer = pl.Trainer(
        max_epochs=max(int(epochs), 1),
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(module, _loader(x, y, config.batch_size, shuffle=True))
    return module.eval().cpu(), scaler


def predict_road_surface_temperature(
    module: GroundTemperatureLightningModule,
    scaler: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    x = frame[ROAD_FEATURES].to_numpy(dtype=np.float32)
    x = (x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(
        scaler["x_std"], dtype=np.float32
    )
    device = next(module.parameters()).device
    with torch.no_grad():
        normalized = module(torch.from_numpy(x).to(device)).cpu().numpy()
    delta = normalized * float(scaler["y_std"]) + float(scaler["y_mean"])
    return frame["road_surface_temp_now"].to_numpy(dtype=float) + delta


def _slice_metrics(
    test: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    near_freezing = np.abs(observed) <= 3.0
    return {
        "near_freezing_observed_minus3_to_plus3_c": {
            "rows": int(near_freezing.sum()),
            **_regression_metrics(observed[near_freezing], predicted[near_freezing]),
        }
    }


def _save_figure(
    path: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    test_period: str,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = observed - predicted
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(observed, predicted, s=8, alpha=0.3, color="#1565c0")
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    axes[0].plot([low, high], [low, high], "r--", linewidth=1)
    axes[0].set_xlabel("Observed road-surface temperature (C)")
    axes[0].set_ylabel("MLP predicted temperature (C)")
    axes[0].set_title(f"{test_period} holdout, R2={metrics['regression']['r2']:.3f}")
    axes[1].scatter(predicted, residual, s=8, alpha=0.3, color="#2e7d32")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted road-surface temperature (C)")
    axes[1].set_ylabel("Observed - predicted (C)")
    axes[1].set_title(
        f"RMSE={metrics['regression']['rmse_c']:.3f}C, "
        f"MAE={metrics['regression']['mae_c']:.3f}C"
    )
    labels = (observed <= 0).astype(int)
    axes[2].hist(probability[labels == 0], bins=25, alpha=0.65, label="above 0C")
    axes[2].hist(probability[labels == 1], bins=25, alpha=0.65, label="0C or below")
    axes[2].set_xlabel("Calibrated freezing probability")
    axes[2].set_title(
        f"PR-AUC={metrics['freezing']['pr_auc']:.3f}, "
        f"Brier={metrics['freezing']['brier_score']:.3f}"
    )
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_road_surface_temperature_mlp(
    weather: pd.DataFrame,
    road_weather: pd.DataFrame,
    model_path: Path,
    reports_dir: Path,
    figures_dir: Path,
    config: GroundTemperatureMLPConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """실측 노면온도 MLP를 학습하고 최신 월 홀드아웃 성능을 저장한다."""
    config = config or GroundTemperatureMLPConfig()
    pl.seed_everything(config.random_state, workers=True)
    dataset = build_road_surface_forecast_dataset(
        weather, road_weather, config.forecast_horizon_hours
    )
    development, calibration, test, split = split_temporal_months(dataset)
    tuned_model, tuned_scaler, best_epochs = _fit_with_validation(
        development, calibration, config
    )
    calibration_prediction = predict_road_surface_temperature(
        tuned_model, tuned_scaler, calibration
    )
    calibrator = _fit_freeze_calibrator(
        calibration_prediction,
        calibration["target_road_surface_temp"].to_numpy(dtype=float),
    )

    evaluation_train = pd.concat([development, calibration], ignore_index=True)
    evaluation_model, evaluation_scaler = _fit_fixed_epochs(
        evaluation_train, config, best_epochs
    )
    test_prediction = predict_road_surface_temperature(
        evaluation_model, evaluation_scaler, test
    )
    observed = test["target_road_surface_temp"].to_numpy(dtype=float)
    probability = freeze_probability(test_prediction, calibrator)
    persistence = test["road_surface_temp_now"].to_numpy(dtype=float)
    metrics = {
        "schema_version": 3,
        "model_version": MODEL_VERSION,
        "model_type": "PyTorch Lightning residual MLP regression",
        "target": f"KMA observed road-surface temperature at +{config.forecast_horizon_hours}h",
        "label_source": "KMA road weather station 13402 observed road-surface temperature",
        "pseudo_label_used": False,
        "evaluation_scope": "observed_road_surface_temperature_temporal_holdout",
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "features": ROAD_FEATURES,
        "development_range": [
            development["valid_time"].min().isoformat(),
            development["valid_time"].max().isoformat(),
        ],
        **split,
        "development_rows": int(len(development)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "selected_epochs": int(best_epochs),
        "regression": _regression_metrics(observed, test_prediction),
        "freezing": _freeze_metrics(observed, probability),
        "baselines": {
            "current_road_surface_temperature_persistence": _regression_metrics(
                observed, persistence
            ),
            "issue_air_temperature": _regression_metrics(
                observed, test["temp"].to_numpy(dtype=float)
            ),
            "issue_asos_ground_temperature": _regression_metrics(
                observed, test["ground_temp_now"].to_numpy(dtype=float)
            ),
        },
        "holdout_used_for_training_or_calibration": False,
        "actual_road_surface_temperature_validated": True,
        "road_icing_state_validated": False,
        "interpretation": (
            "지표는 강남 인접 경부선 원지 관측소의 실측 노면온도 및 0C 이하 사건 성능이다. "
            "강남 전체 도로별 결빙 정확도나 사고확률로 해석하지 않는다."
        ),
    }
    metrics["regression"]["skill_vs_road_persistence_rmse"] = float(
        1
        - metrics["regression"]["rmse_c"]
        / metrics["baselines"]["current_road_surface_temperature_persistence"]["rmse_c"]
    )
    metrics["uncertainty"] = _bootstrap_intervals(
        observed, test_prediction, probability
    )
    metrics["slices"] = _slice_metrics(test, observed, test_prediction)

    operational_model, operational_scaler = _fit_fixed_epochs(
        dataset, config, best_epochs
    )
    bundle = {
        "model_version": MODEL_VERSION,
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "features": ROAD_FEATURES,
        "architecture": {
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "residual_base": "road_surface_temp_now",
        },
        "operational_state_dict": operational_model.state_dict(),
        "operational_scaler": operational_scaler,
        "evaluation_state_dict": evaluation_model.state_dict(),
        "evaluation_scaler": evaluation_scaler,
        "freeze_calibrator": calibrator,
        "metrics": metrics,
        "config": asdict(config),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_path)
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    project_dir = reports_dir.parents[1]
    metrics["model_artifact"] = {
        "path": model_path.relative_to(project_dir).as_posix(),
        "sha256": sha256,
        "bytes": model_path.stat().st_size,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "road_surface_model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "model_version": MODEL_VERSION,
        "artifact": model_path.name,
        "sha256": sha256,
        "bytes": model_path.stat().st_size,
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "features": ROAD_FEATURES,
        "architecture": bundle["architecture"],
        "target": metrics["target"],
        "pseudo_label_used": False,
        "calibration_period": metrics["calibration_period"],
        "test_period": metrics["test_period"],
        "test_rows": metrics["test_rows"],
        "rmse_c": metrics["regression"]["rmse_c"],
        "r2": metrics["regression"]["r2"],
        "freeze_pr_auc": metrics["freezing"].get("pr_auc"),
        "claim_scope": "one nearby KMA road station; not road-by-road icing accuracy",
    }
    (model_path.parent / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "model_version": MODEL_VERSION,
        "purpose": "3시간 후 실측 노면온도를 예측해 제설 상대 위험도 입력으로 사용",
        "label_source": metrics["label_source"],
        "pseudo_label_used": False,
        "temporal_split": {
            "development_range": metrics["development_range"],
            "calibration_period": metrics["calibration_period"],
            "test_period": metrics["test_period"],
        },
        "metrics": {
            "regression": metrics["regression"],
            "freezing": metrics["freezing"],
            "uncertainty": metrics["uncertainty"],
            "slices": metrics["slices"],
        },
        "allowed_claims": [
            "미학습 월의 원지 관측소 실측 노면온도로 3시간 예측 성능을 평가했다.",
            "예측 노면온도와 수분 및 그림자를 결합해 도로의 상대 제설 순위를 만들었다.",
        ],
        "prohibited_claims": [
            "단일 인접 관측소 성능을 강남 모든 도로의 결빙 정확도로 표현",
            "도로 risk를 사고확률 또는 절대 결빙확률로 표현",
        ],
        "spatial_limit": "경부선 원지 관측소는 강남 중심에서 약 5.25km 떨어진 단일 지점",
    }
    (reports_dir / "road_surface_model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_figure(
        figures_dir / "road_surface_mlp_evaluation.png",
        observed,
        test_prediction,
        probability,
        metrics["test_period"],
        metrics,
    )
    return bundle, metrics


def validate_model_bundle(bundle: dict[str, Any]) -> None:
    required = {
        "model_version",
        "forecast_horizon_hours",
        "features",
        "architecture",
        "operational_state_dict",
        "operational_scaler",
        "evaluation_state_dict",
        "evaluation_scaler",
        "freeze_calibrator",
        "metrics",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"노면온도 모델 번들 필수 항목 누락: {missing}")
    if bundle["model_version"] != MODEL_VERSION:
        raise ValueError("노면온도 모델 버전이 현재 코드와 다릅니다.")
    if list(bundle["features"]) != ROAD_FEATURES:
        raise ValueError("노면온도 모델 입력 피처 순서가 현재 코드와 다릅니다.")


def load_model_bundle(model_path: Path, verify_manifest: bool = True) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"노면온도 모델 파일이 없습니다: {model_path}")
    if verify_manifest:
        manifest_path = model_path.parent / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"노면온도 모델 manifest가 없습니다: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if manifest.get("artifact") != model_path.name or manifest.get("sha256") != actual_hash:
            raise ValueError("노면온도 모델과 manifest의 파일명 또는 SHA-256이 다릅니다.")
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    validate_model_bundle(bundle)
    return bundle


def _restore_model(bundle: dict[str, Any], state_key: str) -> GroundTemperatureLightningModule:
    validate_model_bundle(bundle)
    architecture = bundle["architecture"]
    module = GroundTemperatureLightningModule(
        input_dim=len(ROAD_FEATURES),
        hidden_dims=tuple(architecture["hidden_dims"]),
        dropout=float(architecture["dropout"]),
        use_plateau_scheduler=False,
    )
    module.load_state_dict(bundle[state_key])
    return module.eval()


def apply_mlp_road_risk(
    roads: Any,
    weather: pd.DataFrame,
    road_weather: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    historical_holdout: bool = True,
) -> Any:
    """실측 노면온도 MLP 예측과 수분·그림자를 상대 위험지수로 변환한다."""
    validate_model_bundle(bundle)
    frame = prepare_road_surface_features(weather, road_weather)
    frame = frame.dropna(subset=ROAD_FEATURES).copy()
    valid_time = frame["일시"] + pd.to_timedelta(
        int(bundle["forecast_horizon_hours"]), unit="h"
    )
    if historical_holdout:
        test_period = str(bundle["metrics"]["test_period"])
        frame = frame[valid_time.dt.to_period("M").astype(str) == test_period].copy()
        model = _restore_model(bundle, "evaluation_state_dict")
        scaler = bundle["evaluation_scaler"]
        risk_basis = f"{test_period}_road_surface_temporal_holdout_predictions"
    else:
        model = _restore_model(bundle, "operational_state_dict")
        scaler = bundle["operational_scaler"]
        risk_basis = "operational_road_surface_predictions"
    if frame.empty:
        raise ValueError("노면 위험도 계산에 사용할 결합 기상 행이 없습니다.")
    predicted = predict_road_surface_temperature(model, scaler, frame)
    moisture = weather_moisture_index(frame)
    event_mask = (predicted <= 3.0) & (moisture >= 0.05)
    if int(event_mask.sum()) < 20:
        raise ValueError("노면 결빙 후보 기상 사건이 20개 미만입니다.")

    event_temp = predicted[event_mask]
    event_moisture = moisture[event_mask]
    event_solar = frame.loc[event_mask, "solar_3h_sum"].to_numpy(dtype=float)
    shadow = np.clip(roads["shadow_index"].to_numpy(dtype=float), 0, 1)
    effective_temp = event_temp[:, None] - shadow[None, :] * (
        0.25 + 0.65 * np.clip(event_solar[:, None] / 2.5, 0, 1.5)
    )
    freeze = freeze_probability(effective_temp, bundle["freeze_calibrator"])
    hazard = np.clip(
        freeze * event_moisture[:, None] * (0.85 + 0.15 * shadow[None, :]), 0, 1
    )
    roads = roads.copy()
    roads["risk"] = np.clip(
        0.45 * hazard.mean(axis=0)
        + 0.35 * np.quantile(hazard, 0.90, axis=0)
        + 0.20 * np.quantile(hazard, 0.99, axis=0),
        0,
        1,
    )
    roads["risk_source"] = MODEL_VERSION
    roads["risk_interpretation"] = "relative_treatment_priority_not_absolute_probability"
    roads["forecast_horizon_hours"] = int(bundle["forecast_horizon_hours"])
    roads["predicted_road_surface_temp_event_mean"] = float(np.mean(event_temp))
    roads["risk_weather_basis"] = risk_basis
    roads["risk_weather_rows"] = int(len(frame))
    roads["risk_event_rows"] = int(event_mask.sum())
    return roads


__all__ = [
    "MODEL_VERSION",
    "ROAD_FEATURES",
    "apply_mlp_road_risk",
    "build_road_surface_forecast_dataset",
    "load_model_bundle",
    "prepare_road_surface_features",
    "split_temporal_months",
    "train_road_surface_temperature_mlp",
    "validate_model_bundle",
]
