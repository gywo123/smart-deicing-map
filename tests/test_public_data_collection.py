import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from scripts.collect_public_data import (
    aggregate_kma_hourly,
    gangnam_icing_segments,
    match_segments_to_roads,
    parse_kma_road_observations,
    select_nearest_icing_station,
    winter_days,
)
from datetime import date


def test_gangnam_icing_segments_filters_region_and_builds_lines():
    frame = pd.DataFrame(
        [
            {
                "구간번호": "G1",
                "관리청": "서울특별시",
                "도로분류": "시도",
                "대표지역": "서울특별시 강남구",
                "도로(노선)명": "테스트로",
                "총길이(km)": 1.0,
                "기점 위도(WGS84(4326))": 37.50,
                "기점 경도(WGS84(4326))": 127.04,
                "종점 위도(WGS84(4326))": 37.51,
                "종점 경도(WGS84(4326))": 127.05,
            },
            {
                "구간번호": "S1",
                "관리청": "서울특별시",
                "도로분류": "시도",
                "대표지역": "서울특별시 서초구",
                "도로(노선)명": "다른로",
                "총길이(km)": 1.0,
                "기점 위도(WGS84(4326))": 37.50,
                "기점 경도(WGS84(4326))": 127.04,
                "종점 위도(WGS84(4326))": 37.51,
                "종점 경도(WGS84(4326))": 127.05,
            },
        ]
    )

    selected = gangnam_icing_segments(frame)

    assert selected["구간번호"].tolist() == ["G1"]
    assert selected.geometry.iloc[0].geom_type == "LineString"


def test_match_segments_to_roads_keeps_every_link_inside_buffer():
    segments = gpd.GeoDataFrame(
        {
            "구간번호": ["G1"],
            "대표지역": ["서울특별시 강남구"],
            "도로(노선)명": ["테스트로"],
            "총길이(km)": [1.0],
            "geometry": [LineString([(127.04, 37.50), (127.05, 37.50)])],
        },
        crs="EPSG:4326",
    )
    roads = gpd.GeoDataFrame(
        {
            "LINK_ID": ["near-1", "near-2", "far"],
            "geometry": [
                LineString([(127.041, 37.50), (127.042, 37.50)]),
                LineString([(127.043, 37.50005), (127.044, 37.50005)]),
                LineString([(127.04, 37.51), (127.05, 37.51)]),
            ],
        },
        crs="EPSG:4326",
    )

    matched = match_segments_to_roads(segments, roads, buffer_m=20)

    assert matched.loc[0, "matched_link_count"] == 2
    assert set(matched.loc[0, "matched_link_ids"].split("|")) == {"near-1", "near-2"}


def test_parse_kma_response_decodes_korean_and_replaces_missing_values():
    payload = (
        "# comment\n"
        "202501150000, 13402, 원지(도), 경부선, 목표(결빙), "
        "37.4538, 127.0488, 35, 1, -0.3, -999, -999.0, 0.0, "
        "-999.0, -2.1, -999, -999, -999, -999, -999, -999, -999, -999, -999\n"
    ).encode("euc-kr")

    frame = parse_kma_road_observations(payload)

    assert frame.loc[0, "station_name"] == "원지(도)"
    assert frame.loc[0, "road_surface_temp"] == -0.3
    assert pd.isna(frame.loc[0, "visibility_m"])


def test_select_nearest_icing_station_uses_coordinates():
    stations = pd.DataFrame(
        {
            "station_id": ["near", "far"],
            "station_name": ["가까움", "멀리"],
            "road_name": ["경부선", "다른선"],
            "station_type": ["목표(결빙)", "목표(결빙)"],
            "lat": [37.49, 36.0],
            "lon": [127.03, 128.0],
        }
    )

    selected = select_nearest_icing_station(stations)

    assert selected["station_id"] == "near"
    assert selected["distance_to_gangnam_km"] < 2


def test_winter_days_excludes_non_winter_months_and_end_date():
    days = winter_days(date(2025, 2, 27), date(2025, 11, 3))

    assert days == [
        date(2025, 2, 27),
        date(2025, 2, 28),
        date(2025, 11, 1),
        date(2025, 11, 2),
    ]


def test_aggregate_kma_hourly_uses_actual_observations_only():
    frame = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                ["2025-01-15 00:00", "2025-01-15 00:30", "2025-01-15 01:00"]
            ),
            "station_id": ["13402"] * 3,
            "station_name": ["원지(도)"] * 3,
            "road_name": ["경부선"] * 3,
            "station_type": ["목표(결빙)"] * 3,
            "lat": [37.4538] * 3,
            "lon": [127.0488] * 3,
            "height_m": [35.0] * 3,
            "road_surface_state": pd.array([1, 1, 2], dtype="Int64"),
            "road_surface_temp": [-1.0, 1.0, 2.0],
        }
    )

    hourly = aggregate_kma_hourly(frame)

    assert len(hourly) == 2
    assert hourly.loc[0, "hour"] == pd.Timestamp("2025-01-15 01:00")
    assert hourly.loc[0, "road_surface_temp"] == 0.0
    assert hourly.loc[0, "freezing_ratio"] == 0.5
    assert hourly.loc[0, "observation_count"] == 2
