import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.mlp_training_pipeline import (  # noqa: E402
    FEATURES,
    build_accident_pseudo_labels,
    build_physical_icing_baseline,
    classification_metrics,
    get_trainer_device_config,
    prepare_icing_features,
    refresh_shadow_index_if_possible,
    resolve_roads_geojson_path,
    soften_probabilities,
)


def _roads_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shadow_index": [0.1, 0.3, 0.8, 1.0, 0.0],
            "pop_weight": [0.2, 0.4, 0.9, 1.0, 0.1],
            "LENGTH": [100, 200, 600, 800, 150],
            "LANES": [4, 3, 2, 1, 5],
            "road_width": [8, 8, 10, 6, 12],
            "area": [800, 1600, 6000, 4800, 1800],
            "deicing_cost": [1000, 2000, 6000, 8000, 1500],
            "risk": [0.1, 0.3, 0.7, 0.9, 0.2],
        }
    )


def test_prepare_icing_features_normalizes_and_labels():
    x, y, scaler = prepare_icing_features(_roads_frame())

    assert x.shape == (5, len(FEATURES))
    assert y.shape == (5,)
    assert set(y.tolist()) == {0.0, 1.0}
    assert scaler["label_threshold"] == pytest.approx(0.46)
    assert len(scaler["mean"]) == len(FEATURES)


def test_build_accident_pseudo_labels_outputs_probability_range():
    roads = _roads_frame()
    labels, scores, threshold = build_accident_pseudo_labels(
        roads,
        icing_probs=np.array([0.1, 0.3, 0.7, 0.9, 0.2], dtype=np.float32),
    )

    assert labels.shape == (5,)
    assert scores.shape == (5,)
    assert 0 <= threshold <= 1
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[3] > scores[0]


def test_soften_probabilities_reduces_extreme_values():
    softened = soften_probabilities(np.array([0.01, 0.5, 0.99]))

    assert softened[0] > 0.01
    assert softened[1] == pytest.approx(0.5)
    assert softened[2] < 0.99


def test_build_physical_icing_baseline_is_continuous_probability():
    baseline = build_physical_icing_baseline(_roads_frame())

    assert baseline.shape == (5,)
    assert np.all((baseline >= 0.05) & (baseline <= 0.95))
    assert len(np.unique(np.round(baseline, 3))) > 2


def test_classification_metrics_handles_single_class_without_auc():
    metrics = classification_metrics(
        np.array([1, 1, 1], dtype=np.float32),
        np.array([0.6, 0.7, 0.9], dtype=np.float32),
    )

    assert metrics["accuracy"] == 1.0
    assert "roc_auc" not in metrics


def test_get_trainer_device_config_returns_lightning_values():
    config = get_trainer_device_config()

    assert config.accelerator in {"cpu", "gpu"}
    assert config.devices == 1
    assert config.description


def test_refresh_shadow_index_keeps_existing_values_when_raw_buildings_missing():
    roads = _roads_frame()

    class PipelineStub:
        @staticmethod
        def load_buildings_and_shadow(_roads):
            raise FileNotFoundError("missing building shp")

    updated, source = refresh_shadow_index_if_possible(roads, PipelineStub)

    assert updated is roads
    assert source == "existing_shadow_index_raw_buildings_missing"


def test_resolve_roads_geojson_path_prefers_current_outputs(tmp_path):
    current = tmp_path / "outputs" / "data"
    legacy = tmp_path / "src" / "outputs" / "data"
    current.mkdir(parents=True)
    legacy.mkdir(parents=True)
    current_file = current / "gangnam_roads_result.geojson"
    legacy_file = legacy / "gangnam_roads_result.geojson"
    current_file.write_text("{}", encoding="utf-8")
    legacy_file.write_text("{}", encoding="utf-8")

    class PipelineStub:
        BASE_DIR = str(tmp_path)
        OUTPUT_DATA_DIR = str(current)

    assert resolve_roads_geojson_path(PipelineStub) == current_file


def test_resolve_roads_geojson_path_uses_legacy_outputs(tmp_path):
    current = tmp_path / "outputs" / "data"
    legacy = tmp_path / "src" / "outputs" / "data"
    legacy.mkdir(parents=True)
    legacy_file = legacy / "gangnam_roads_result.geojson"
    legacy_file.write_text("{}", encoding="utf-8")

    class PipelineStub:
        BASE_DIR = str(tmp_path)
        OUTPUT_DATA_DIR = str(current)

    assert resolve_roads_geojson_path(PipelineStub) == legacy_file
