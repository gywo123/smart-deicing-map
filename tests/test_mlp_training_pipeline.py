import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.ground_temperature_mlp import (  # noqa: E402
    FEATURES,
    MODEL_VERSION,
    build_forecast_dataset,
    freeze_probability,
    split_temporal_years,
    validate_model_bundle,
    weather_moisture_index,
)
from src.mlp_training_pipeline import load_enriched_weather  # noqa: E402


def _weather_frame() -> pd.DataFrame:
    rows = []
    for year in [2021, 2022, 2023, 2024]:
        for hour in range(8):
            timestamp = pd.Timestamp(year=year, month=1, day=1, hour=hour)
            temp = -4 + hour * 0.5
            ground = temp - 1
            rows.append(
                {
                    "일시": timestamp,
                    "temp": temp,
                    "ground_temp": ground,
                    "ground_temp_lag_1h": ground - 0.5,
                    "temp_6h_mean": temp - 0.5,
                    "humidity": 85,
                    "dewpoint_depression": 1.5,
                    "wind": 2,
                    "precip_6h": 0.5,
                    "snow": 1,
                    "new_snow_6h": 0.2,
                    "solar": 0,
                    "solar_3h_sum": 0,
                    "sunshine": 0,
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "day_sin": 0,
                    "day_cos": 1,
                }
            )
    return pd.DataFrame(rows)


def test_features_exclude_cost_and_formula_outputs():
    forbidden = {"deicing_cost", "area", "priority_score", "risk", "accident_probability"}
    assert forbidden.isdisjoint(FEATURES)
    assert {"temp", "ground_temp_now", "precip_6h", "snow", "solar"}.issubset(FEATURES)


def test_forecast_dataset_uses_exact_future_timestamp():
    dataset = build_forecast_dataset(_weather_frame(), horizon_hours=3)

    assert not dataset.empty
    assert (dataset["valid_time"] - dataset["issue_time"] == pd.Timedelta(hours=3)).all()
    assert np.allclose(dataset["target_delta"], 1.5)


def test_temporal_split_reserves_latest_two_years():
    dataset = build_forecast_dataset(_weather_frame(), horizon_hours=3)
    development, calibration, test, years = split_temporal_years(dataset)

    assert set(development["valid_time"].dt.year) == {2021, 2022}
    assert set(calibration["valid_time"].dt.year) == {2023}
    assert set(test["valid_time"].dt.year) == {2024}
    assert years == {"calibration_year": 2023, "test_year": 2024}


def test_freeze_probability_decreases_as_temperature_rises():
    calibrator = {"coefficient": -2.0, "intercept": 0.0}
    probabilities = freeze_probability(np.array([-2.0, 0.0, 2.0]), calibrator)

    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_moisture_index_responds_to_precipitation_and_snow():
    dry = _weather_frame().iloc[:1].copy()
    wet = dry.copy()
    dry[["precip_6h", "snow", "new_snow_6h"]] = 0
    dry["humidity"] = 40
    dry["dewpoint_depression"] = 10
    wet[["precip_6h", "snow", "new_snow_6h"]] = [3, 4, 2]

    assert weather_moisture_index(wet)[0] > weather_moisture_index(dry)[0]


def test_load_enriched_weather_rejects_old_schema(tmp_path):
    output = tmp_path / "outputs" / "data" / "cleaned"
    output.mkdir(parents=True)
    pd.DataFrame({"일시": ["2025-01-01"], "ground_temp": [-1]}).to_csv(
        output / "weather_winter_clean.csv", index=False
    )

    with pytest.raises(ValueError, match="이전 형식"):
        load_enriched_weather(tmp_path)


def test_model_bundle_schema_rejects_wrong_feature_order():
    bundle = {
        "model_version": MODEL_VERSION,
        "forecast_horizon_hours": 3,
        "features": list(reversed(FEATURES)),
        "architecture": {},
        "operational_state_dict": {},
        "operational_scaler": {},
        "evaluation_state_dict": {},
        "evaluation_scaler": {},
        "freeze_calibrator": {},
        "metrics": {},
    }

    with pytest.raises(ValueError, match="피처 순서"):
        validate_model_bundle(bundle)
