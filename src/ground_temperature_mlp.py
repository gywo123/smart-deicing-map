"""실제 ASOS 지면온도를 학습하는 시간 홀드아웃 MLP 파이프라인."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader, TensorDataset

from .torch_lightning_risk_model import GroundTemperatureLightningModule


MODEL_VERSION = "ground_temperature_residual_mlp_v1"
FORECAST_HORIZON_HOURS = 3
FEATURES = [
    "temp",
    "ground_temp_now",
    "ground_temp_lag_1h",
    "temp_6h_mean",
    "humidity",
    "dewpoint_depression",
    "wind",
    "precip_6h",
    "snow",
    "new_snow_6h",
    "solar",
    "solar_3h_sum",
    "sunshine",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


@dataclass(frozen=True)
class GroundTemperatureMLPConfig:
    max_epochs: int = 180
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.08
    early_stopping_patience: int = 24
    random_state: int = 42
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS


def build_forecast_dataset(
    weather: pd.DataFrame,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """발행 시각 피처와 정확히 horizon 뒤의 실제 지면온도를 결합한다."""
    required = {"일시", "ground_temp", *[f for f in FEATURES if f != "ground_temp_now"]}
    missing = sorted(required - set(weather.columns))
    if missing:
        raise ValueError(f"지면온도 MLP 입력 컬럼 누락: {missing}")
    if horizon_hours <= 0:
        raise ValueError("예측 horizon은 1시간 이상이어야 합니다.")

    issue = weather.copy()
    issue["일시"] = pd.to_datetime(issue["일시"], errors="coerce")
    issue["ground_temp_now"] = pd.to_numeric(issue["ground_temp"], errors="coerce")
    issue["issue_time"] = issue["일시"]
    issue["valid_time"] = issue["issue_time"] + pd.to_timedelta(horizon_hours, unit="h")
    issue = issue[["issue_time", "valid_time", *FEATURES]].dropna()

    target = weather[["일시", "ground_temp"]].copy()
    target["일시"] = pd.to_datetime(target["일시"], errors="coerce")
    target["ground_temp"] = pd.to_numeric(target["ground_temp"], errors="coerce")
    target = target.rename(columns={"일시": "valid_time", "ground_temp": "target_ground_temp"})
    target = target.dropna().drop_duplicates("valid_time", keep="last")

    dataset = issue.merge(target, on="valid_time", how="inner", validate="many_to_one")
    dataset["target_delta"] = dataset["target_ground_temp"] - dataset["ground_temp_now"]
    return dataset.sort_values("issue_time").reset_index(drop=True)


def split_temporal_years(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """최신 연도는 테스트, 직전 연도는 튜닝/확률보정 전용으로 분리한다."""
    years = sorted(dataset["valid_time"].dt.year.unique().tolist())
    if len(years) < 4:
        raise ValueError("시간 홀드아웃에는 최소 4개 연도의 관측값이 필요합니다.")
    test_year = int(years[-1])
    calibration_year = int(years[-2])
    development = dataset[dataset["valid_time"].dt.year < calibration_year].copy()
    calibration = dataset[dataset["valid_time"].dt.year == calibration_year].copy()
    test = dataset[dataset["valid_time"].dt.year == test_year].copy()
    if development.empty or calibration.empty or test.empty:
        raise ValueError("연도 분할 결과에 빈 데이터가 있습니다.")
    return development, calibration, test, {
        "calibration_year": calibration_year,
        "test_year": test_year,
    }


def _fit_scalers(frame: pd.DataFrame) -> dict[str, Any]:
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame["target_delta"].to_numpy(dtype=np.float32)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(y.mean())
    y_std = max(float(y.std()), 1e-6)
    return {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _transform(frame: pd.DataFrame, scaler: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    x = (x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(
        scaler["x_std"], dtype=np.float32
    )
    y = frame["target_delta"].to_numpy(dtype=np.float32)
    y = (y - float(scaler["y_mean"])) / float(scaler["y_std"])
    return x.astype(np.float32), y.astype(np.float32)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _trainer_device() -> tuple[str, int, str]:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        return "gpu", 1, f"GPU - {torch.cuda.get_device_name(0)}"
    return "cpu", 1, "CPU"


def _fit_with_validation(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    config: GroundTemperatureMLPConfig,
) -> tuple[GroundTemperatureLightningModule, dict[str, Any], int]:
    scaler = _fit_scalers(development)
    x_train, y_train = _transform(development, scaler)
    x_val, y_val = _transform(validation, scaler)
    accelerator, devices, description = _trainer_device()
    print(f"MLP 학습 장치: {description}")

    module = GroundTemperatureLightningModule(
        input_dim=len(FEATURES),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    with tempfile.TemporaryDirectory(prefix="icezero_mlp_") as temp_dir:
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
        checkpoint_payload = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        best_epochs = int(checkpoint_payload["epoch"]) + 1
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
        input_dim=len(FEATURES),
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


def predict_ground_temperature(
    module: GroundTemperatureLightningModule,
    scaler: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    x = (x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(
        scaler["x_std"], dtype=np.float32
    )
    device = next(module.parameters()).device
    with torch.no_grad():
        normalized = module(torch.from_numpy(x).to(device)).cpu().numpy()
    delta = normalized * float(scaler["y_std"]) + float(scaler["y_mean"])
    return frame["ground_temp_now"].to_numpy(dtype=float) + delta


def _fit_freeze_calibrator(predicted_temp: np.ndarray, observed_temp: np.ndarray) -> dict[str, float]:
    labels = (np.asarray(observed_temp, dtype=float) <= 0).astype(int)
    estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    estimator.fit(np.asarray(predicted_temp, dtype=float).reshape(-1, 1), labels)
    return {
        "coefficient": float(estimator.coef_[0, 0]),
        "intercept": float(estimator.intercept_[0]),
    }


def freeze_probability(predicted_temp: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    logits = calibrator["intercept"] + calibrator["coefficient"] * np.asarray(
        predicted_temp, dtype=float
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse_c": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae_c": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias_observed_minus_predicted_c": float(np.mean(observed - predicted)),
    }


def _freeze_metrics(observed_temp: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = (np.asarray(observed_temp, dtype=float) <= 0).astype(int)
    predictions = (np.asarray(probability, dtype=float) >= 0.5).astype(int)
    result = {
        "rows": int(len(labels)),
        "observed_freeze_rate": float(labels.mean()),
        "brier_score": float(brier_score_loss(labels, probability)),
        "decision_threshold": 0.5,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        result["pr_auc"] = float(average_precision_score(labels, probability))
        result["roc_auc"] = float(roc_auc_score(labels, probability))
    return result


def _bootstrap_intervals(
    observed: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """시간 홀드아웃 지표의 표본 불확실성을 재현 가능한 구간으로 요약한다."""
    rng = np.random.default_rng(seed)
    labels = (observed <= 0).astype(int)
    rmse_values: list[float] = []
    mae_values: list[float] = []
    pr_auc_values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(observed), size=len(observed))
        sampled_observed = observed[indices]
        sampled_predicted = predicted[indices]
        sampled_labels = labels[indices]
        rmse_values.append(float(np.sqrt(mean_squared_error(sampled_observed, sampled_predicted))))
        mae_values.append(float(mean_absolute_error(sampled_observed, sampled_predicted)))
        if len(np.unique(sampled_labels)) > 1:
            pr_auc_values.append(
                float(average_precision_score(sampled_labels, probability[indices]))
            )

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "rmse_c_95pct": interval(rmse_values),
        "mae_c_95pct": interval(mae_values),
        "freeze_pr_auc_95pct": interval(pr_auc_values),
        "bootstrap_iterations": iterations,
        "seed": seed,
    }


def _slice_metrics(
    test: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    near_freezing = np.abs(observed) <= 3.0
    monthly: dict[str, dict[str, float]] = {}
    months = test["valid_time"].dt.month.to_numpy(dtype=int)
    for month in sorted(np.unique(months)):
        mask = months == month
        monthly[str(int(month))] = {
            "rows": int(mask.sum()),
            **_regression_metrics(observed[mask], predicted[mask]),
        }
    return {
        "near_freezing_observed_minus3_to_plus3_c": {
            "rows": int(near_freezing.sum()),
            **_regression_metrics(observed[near_freezing], predicted[near_freezing]),
        },
        "monthly": monthly,
    }


def _save_evaluation_figure(
    path: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    test_year: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = observed - predicted
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sample = min(3000, len(observed))
    axes[0].scatter(observed[:sample], predicted[:sample], s=7, alpha=0.25, color="#1565c0")
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    axes[0].plot([low, high], [low, high], "r--", linewidth=1)
    axes[0].set_xlabel("ASOS observed ground temperature (C)")
    axes[0].set_ylabel("MLP predicted ground temperature (C)")
    axes[0].set_title(f"{test_year} temporal holdout, R2={metrics['regression']['r2']:.3f}")

    axes[1].scatter(predicted[:sample], residual[:sample], s=7, alpha=0.25, color="#2e7d32")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted ground temperature (C)")
    axes[1].set_ylabel("Observed - predicted (C)")
    axes[1].set_title(f"RMSE={metrics['regression']['rmse_c']:.3f}C, MAE={metrics['regression']['mae_c']:.3f}C")

    labels = (observed <= 0).astype(int)
    axes[2].hist(probability[labels == 0], bins=30, alpha=0.65, label="above 0C", color="#78909c")
    axes[2].hist(probability[labels == 1], bins=30, alpha=0.65, label="0C or below", color="#ef6c00")
    axes[2].set_xlabel("Calibrated freezing probability")
    axes[2].set_title(f"PR-AUC={metrics['freezing']['pr_auc']:.3f}, Brier={metrics['freezing']['brier_score']:.3f}")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_ground_temperature_mlp(
    weather: pd.DataFrame,
    model_path: Path,
    reports_dir: Path,
    figures_dir: Path,
    config: GroundTemperatureMLPConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """실측 지면온도 MLP를 학습하고 독립 연도 성능과 모델을 저장한다."""
    config = config or GroundTemperatureMLPConfig()
    pl.seed_everything(config.random_state, workers=True)
    dataset = build_forecast_dataset(weather, config.forecast_horizon_hours)
    development, calibration, test, split = split_temporal_years(dataset)

    tuned_model, tuned_scaler, best_epochs = _fit_with_validation(development, calibration, config)
    calibration_prediction = predict_ground_temperature(tuned_model, tuned_scaler, calibration)
    calibrator = _fit_freeze_calibrator(
        calibration_prediction,
        calibration["target_ground_temp"].to_numpy(dtype=float),
    )

    evaluation_train = pd.concat([development, calibration], ignore_index=True)
    evaluation_model, evaluation_scaler = _fit_fixed_epochs(
        evaluation_train, config, best_epochs
    )
    test_prediction = predict_ground_temperature(evaluation_model, evaluation_scaler, test)
    observed = test["target_ground_temp"].to_numpy(dtype=float)
    probability = freeze_probability(test_prediction, calibrator)
    air_baseline = test["temp"].to_numpy(dtype=float)
    persistence_baseline = test["ground_temp_now"].to_numpy(dtype=float)

    metrics = {
        "schema_version": 2,
        "model_version": MODEL_VERSION,
        "model_type": "PyTorch Lightning residual MLP regression",
        "target": f"ASOS observed ground temperature at +{config.forecast_horizon_hours}h",
        "pseudo_label_used": False,
        "evaluation_scope": "observed_asos_ground_temperature_temporal_holdout",
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "features": FEATURES,
        "development_years": sorted(development["valid_time"].dt.year.unique().astype(int).tolist()),
        "calibration_year": split["calibration_year"],
        "test_year": split["test_year"],
        "development_rows": int(len(development)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "selected_epochs": int(best_epochs),
        "regression": _regression_metrics(observed, test_prediction),
        "freezing": _freeze_metrics(observed, probability),
        "baselines": {
            "issue_air_temperature": _regression_metrics(observed, air_baseline),
            "current_ground_temperature_persistence": _regression_metrics(
                observed, persistence_baseline
            ),
        },
        "holdout_used_for_training_or_calibration": False,
        "road_icing_ground_truth_validated": False,
        "interpretation": (
            "지표는 ASOS 지면온도 및 0C 이하 사건 예측 성능이다. "
            "도로별 실제 결빙 정확도로 해석하지 않는다."
        ),
    }
    metrics["regression"]["skill_vs_persistence_rmse"] = float(
        1.0
        - metrics["regression"]["rmse_c"]
        / metrics["baselines"]["current_ground_temperature_persistence"]["rmse_c"]
    )
    metrics["uncertainty"] = _bootstrap_intervals(
        observed,
        test_prediction,
        probability,
    )
    metrics["slices"] = _slice_metrics(test, observed, test_prediction)
    if {"지점", "지점명"}.issubset(weather.columns):
        station_rows = weather[["지점", "지점명"]].dropna().drop_duplicates()
        metrics["source_stations"] = [
            {"station_id": int(station), "station_name": str(name)}
            for station, name in station_rows.itertuples(index=False)
        ]

    operational_model, operational_scaler = _fit_fixed_epochs(dataset, config, best_epochs)
    bundle = {
        "model_version": MODEL_VERSION,
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "features": FEATURES,
        "architecture": {
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "residual_base": "ground_temp_now",
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
    (reports_dir / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "model_version": MODEL_VERSION,
        "artifact": model_path.name,
        "sha256": sha256,
        "bytes": model_path.stat().st_size,
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "features": FEATURES,
        "architecture": bundle["architecture"],
        "target": metrics["target"],
        "pseudo_label_used": False,
        "development_years": metrics["development_years"],
        "calibration_year": metrics["calibration_year"],
        "test_year": metrics["test_year"],
        "test_rows": metrics["test_rows"],
        "rmse_c": metrics["regression"]["rmse_c"],
        "r2": metrics["regression"]["r2"],
        "freeze_pr_auc": metrics["freezing"].get("pr_auc"),
        "source_stations": metrics.get("source_stations", []),
        "claim_scope": "ASOS ground temperature only; not road-icing accuracy",
    }
    (model_path.parent / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "model_version": MODEL_VERSION,
        "model_type": metrics["model_type"],
        "purpose": "3시간 후 ASOS 지면온도를 예측해 제설 의사결정의 기상 입력으로 사용",
        "label_source": "기상청 ASOS 시간별 관측 지면온도(C)",
        "pseudo_label_used": False,
        "temporal_split": {
            "development_years": metrics["development_years"],
            "calibration_year": metrics["calibration_year"],
            "test_year": metrics["test_year"],
        },
        "metrics": {
            "regression": metrics["regression"],
            "freezing": metrics["freezing"],
            "uncertainty": metrics["uncertainty"],
            "slices": metrics["slices"],
        },
        "source_stations": metrics.get("source_stations", []),
        "allowed_claims": [
            "2025년 미학습 ASOS 관측에서 3시간 후 지면온도 성능을 평가했다.",
            "예측 지면온도와 수분 및 그림자를 결합해 도로의 상대 제설 순위를 만들었다.",
        ],
        "prohibited_claims": [
            "ASOS 지면온도 성능을 도로 결빙 정확도로 표현",
            "도로 risk를 절대 결빙 확률로 표현",
            "사고 감소율을 현장 실증 결과로 표현",
        ],
        "remaining_limit": "도로별 시각 단위 실제 결빙 라벨은 아직 없음",
        "spatial_limit": "지면온도 정답은 서울 ASOS 관측소 자료이며 강남구 도로 노면센서가 아님",
    }
    (reports_dir / "model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_evaluation_figure(
        figures_dir / "mlp_model_evaluation.png",
        observed,
        test_prediction,
        probability,
        split["test_year"],
        metrics,
    )
    return bundle, metrics


def validate_model_bundle(bundle: dict[str, Any]) -> None:
    """저장 모델이 현재 코드의 버전과 입력 스키마에 맞는지 확인한다."""
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
        raise ValueError(f"모델 번들 필수 항목 누락: {missing}")
    if bundle["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"모델 버전 불일치: {bundle['model_version']} != {MODEL_VERSION}"
        )
    if list(bundle["features"]) != FEATURES:
        raise ValueError("모델 입력 피처 순서가 현재 코드와 다릅니다. 모델을 다시 학습하세요.")


def load_model_bundle(model_path: Path, verify_manifest: bool = True) -> dict[str, Any]:
    """모델 파일의 해시와 내부 스키마를 확인한 뒤 로드한다."""
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    if verify_manifest:
        manifest_path = model_path.parent / "model_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"모델 manifest가 없습니다: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if manifest.get("artifact") != model_path.name:
            raise ValueError("manifest의 모델 파일명이 실제 파일과 다릅니다.")
        if manifest.get("sha256") != actual_hash:
            raise ValueError("모델 SHA-256이 manifest와 다릅니다. 손상되었거나 다른 파일입니다.")
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    validate_model_bundle(bundle)
    return bundle


def _restore_model(
    bundle: dict[str, Any],
    state_key: str,
) -> GroundTemperatureLightningModule:
    validate_model_bundle(bundle)
    architecture = bundle["architecture"]
    module = GroundTemperatureLightningModule(
        input_dim=len(bundle["features"]),
        hidden_dims=tuple(architecture["hidden_dims"]),
        dropout=float(architecture["dropout"]),
        use_plateau_scheduler=False,
    )
    module.load_state_dict(bundle[state_key])
    return module.eval()


def restore_operational_model(bundle: dict[str, Any]) -> GroundTemperatureLightningModule:
    return _restore_model(bundle, "operational_state_dict")


def restore_evaluation_model(bundle: dict[str, Any]) -> GroundTemperatureLightningModule:
    return _restore_model(bundle, "evaluation_state_dict")


def weather_moisture_index(weather: pd.DataFrame) -> np.ndarray:
    liquid = 1.0 - np.exp(-np.maximum(weather["precip_6h"].to_numpy(dtype=float), 0) / 0.7)
    snowpack = 1.0 - np.exp(
        -(
            np.maximum(weather["snow"].to_numpy(dtype=float), 0) / 1.5
            + np.maximum(weather["new_snow_6h"].to_numpy(dtype=float), 0) / 0.5
        )
    )
    humidity = weather["humidity"].to_numpy(dtype=float)
    depression = weather["dewpoint_depression"].to_numpy(dtype=float)
    condensation = (1 / (1 + np.exp(-(humidity - 88) / 4))) * (
        1 / (1 + np.exp(-(2.5 - depression) / 0.8))
    )
    return np.clip(np.maximum(np.maximum(liquid, snowpack), 0.5 * condensation), 0, 1)


def apply_mlp_road_risk(
    roads: Any,
    weather: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    historical_holdout: bool = True,
) -> Any:
    """MLP 예측 지면온도와 수분·그림자를 도로 상대 위험지수로 변환한다.

    기본값은 모델 성능 검증과 같은 미학습 연도의 예측만 사용한다. 실시간으로
    새로 들어온 미래 기상 입력에는 historical_holdout=False를 사용한다.
    """
    validate_model_bundle(bundle)
    frame = weather.copy()
    frame["일시"] = pd.to_datetime(frame["일시"], errors="coerce")
    frame["ground_temp_now"] = pd.to_numeric(frame["ground_temp"], errors="coerce")
    frame = frame.dropna(subset=bundle["features"]).copy()
    if historical_holdout:
        valid_time = frame["일시"] + pd.to_timedelta(
            int(bundle["forecast_horizon_hours"]), unit="h"
        )
        test_year = int(bundle["metrics"]["test_year"])
        frame = frame[valid_time.dt.year == test_year].copy()
        model = restore_evaluation_model(bundle)
        scaler = bundle["evaluation_scaler"]
        risk_basis = f"{test_year}_temporal_holdout_predictions"
    else:
        model = restore_operational_model(bundle)
        scaler = bundle["operational_scaler"]
        risk_basis = "operational_predictions"
    if frame.empty:
        raise ValueError("도로 위험도 계산에 사용할 기상 행이 없습니다.")
    predicted = predict_ground_temperature(model, scaler, frame)
    moisture = weather_moisture_index(frame)
    event_mask = (predicted <= 3.0) & (moisture >= 0.05)
    if int(event_mask.sum()) < 100:
        raise ValueError("결빙 후보 기상 사건이 100개 미만이라 도로 위험도를 계산할 수 없습니다.")

    event_temp = predicted[event_mask]
    event_moisture = moisture[event_mask]
    event_solar = frame.loc[event_mask, "solar_3h_sum"].to_numpy(dtype=float)
    shadow = np.clip(roads["shadow_index"].to_numpy(dtype=float), 0, 1)
    effective_temp = event_temp[:, None] - shadow[None, :] * (
        0.25 + 0.65 * np.clip(event_solar[:, None] / 2.5, 0, 1.5)
    )
    freeze = freeze_probability(effective_temp, bundle["freeze_calibrator"])
    hazard = np.clip(
        freeze * event_moisture[:, None] * (0.85 + 0.15 * shadow[None, :]),
        0,
        1,
    )
    risk_mean = hazard.mean(axis=0)
    risk_p90 = np.quantile(hazard, 0.90, axis=0)
    risk_p99 = np.quantile(hazard, 0.99, axis=0)
    roads = roads.copy()
    roads["risk"] = np.clip(0.45 * risk_mean + 0.35 * risk_p90 + 0.20 * risk_p99, 0, 1)
    roads["risk_source"] = MODEL_VERSION
    roads["risk_interpretation"] = "relative_treatment_priority_not_absolute_probability"
    roads["forecast_horizon_hours"] = int(bundle["forecast_horizon_hours"])
    roads["predicted_ground_temp_event_mean"] = float(np.mean(event_temp))
    roads["risk_weather_basis"] = risk_basis
    roads["risk_weather_rows"] = int(len(frame))
    roads["risk_event_rows"] = int(event_mask.sum())
    return roads
