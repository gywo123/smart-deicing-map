import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.ground_temperature_mlp import FEATURES as ASOS_FEATURES  # noqa: E402
from src.road_surface_temperature_mlp import (  # noqa: E402
    ROAD_FEATURES,
    build_road_surface_forecast_dataset,
    prepare_road_surface_features,
    split_temporal_months,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2024-11-01", periods=24 * 95, freq="h")
    hour = timestamps.hour.to_numpy()
    weather = pd.DataFrame(
        {
            "일시": timestamps,
            "ground_temp": -2 + np.sin(hour / 24 * 2 * np.pi),
        }
    )
    for feature in ASOS_FEATURES:
        if feature != "ground_temp_now":
            weather[feature] = 0.5
    weather["temp"] = weather["ground_temp"] + 1
    road = pd.DataFrame(
        {
            "hour": timestamps,
            "road_surface_temp": weather["ground_temp"] - 0.5,
            "station_id": "13402",
            "station_name": "원지(도)",
            "road_name": "경부선",
            "lat": 37.4538,
            "lon": 127.0488,
        }
    )
    return weather, road


def test_prepare_features_adds_current_and_lagged_road_temperature():
    weather, road = _frames()

    frame = prepare_road_surface_features(weather, road)

    assert "road_surface_temp_now" in frame
    assert "road_surface_temp_lag_1h" in frame
    assert "road_surface_temp_mean_6h" in frame
    assert pd.isna(frame.iloc[0]["road_surface_temp_lag_1h"])
    assert frame.iloc[1]["road_surface_temp_lag_1h"] == frame.iloc[0]["road_surface_temp_now"]


def test_road_forecast_target_is_exactly_three_hours_ahead():
    weather, road = _frames()

    dataset = build_road_surface_forecast_dataset(weather, road, horizon_hours=3)

    assert set(ROAD_FEATURES).issubset(dataset.columns)
    assert (dataset["valid_time"] - dataset["issue_time"] == pd.Timedelta(hours=3)).all()
    target_by_time = road.set_index("hour")["road_surface_temp"]
    expected = target_by_time.loc[dataset["valid_time"]].to_numpy()
    assert np.allclose(dataset["target_road_surface_temp"], expected)
    assert np.allclose(
        dataset["target_delta"], expected - dataset["road_surface_temp_now"].to_numpy()
    )


def test_month_split_uses_latest_complete_month_as_test():
    timestamps = pd.date_range("2024-11-01", "2025-01-31 23:00", freq="h")
    dataset = pd.DataFrame({"valid_time": timestamps})

    development, calibration, test, split = split_temporal_months(dataset)

    assert split == {"calibration_period": "2024-12", "test_period": "2025-01"}
    assert development["valid_time"].dt.to_period("M").unique().tolist() == [
        pd.Period("2024-11", freq="M")
    ]
    assert calibration["valid_time"].dt.month.unique().tolist() == [12]
    assert test["valid_time"].dt.month.unique().tolist() == [1]
