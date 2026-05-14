import pytest
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import LineString

from scripts import main_pipeline
from scripts.main_pipeline import (
    build_road_network_router,
    choose_vehicle_count,
    require_real_data_file,
)
from scripts.route_visualization import _estimate_deicing_kg, bearing_label
from scripts.shadow_utils import solar_position_kst
from scripts.validation_simulation import scenario_severity, weather_bucket


def test_bearing_label_cardinal_directions():
    assert bearing_label((0, 0), (0, 1)) == "북"
    assert bearing_label((0, 0), (1, 0)) == "동"
    assert bearing_label((0, 0), (0, -1)) == "남"
    assert bearing_label((0, 0), (-1, 0)) == "서"


def test_estimate_deicing_kg_uses_pipeline_default_when_rate_missing():
    assert _estimate_deicing_kg({"area": 2000}) == pytest.approx(60)


def test_choose_vehicle_count_scales_with_workload():
    rows = []
    for idx in range(600):
        rows.append(
            {
                "F_NODE": f"N{idx}",
                "T_NODE": f"N{idx + 1}",
                "LENGTH": 500,
                "deicing_kg": 120,
            }
        )

    assert choose_vehicle_count(pd.DataFrame(rows)) >= 6


def test_solar_position_kst_returns_reasonable_winter_noon_elevation():
    elevation, azimuth = solar_position_kst(37.514, 127.047, 1, 15, 12)

    assert 20 <= elevation <= 40
    assert 0 <= azimuth <= 360


def test_require_real_data_file_rejects_git_lfs_pointer(tmp_path):
    pointer = tmp_path / "MOCT_LINK.shp"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        require_real_data_file(str(pointer), "도로 Shapefile")


def test_road_router_uses_directed_node_link_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(main_pipeline, "DATA_DIR", str(tmp_path))
    roads = gpd.GeoDataFrame(
        {
            "LINK_ID": ["L1"],
            "F_NODE": ["A"],
            "T_NODE": ["B"],
            "LENGTH": [100.0],
            "ROAD_USE": [0],
            "geometry": [LineString([(127.0, 37.0), (127.001, 37.0)])],
        },
        crs="EPSG:4326",
    )

    graph, _ = build_road_network_router(roads)

    assert isinstance(graph, nx.DiGraph)
    assert graph.has_edge("A", "B")
    assert not graph.has_edge("B", "A")


def test_validation_weather_bucket_and_severity():
    weather = pd.DataFrame(
        [
            {"일시": pd.Timestamp("2025-01-01 06:00"), "temp": -5, "ground_temp": -2, "humidity": 90, "wind": 4, "precip": 0, "snow": 2},
            {"일시": pd.Timestamp("2025-01-01 12:00"), "temp": 3, "ground_temp": 2, "humidity": 50, "wind": 1, "precip": 0, "snow": 0},
        ]
    )

    assert weather_bucket(weather.iloc[0]) == "눈"
    assert weather_bucket(weather.iloc[1]) == "맑음"
    severity = scenario_severity(weather)
    assert severity[0] > severity[1]
