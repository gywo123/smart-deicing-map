"""건물 그림자 지수를 계산하는 유틸리티."""

from __future__ import annotations

import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def require_real_data_file(path: str, label: str) -> None:
    """Git LFS pointer나 누락 파일을 명확한 오류로 알려준다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {path}")
    with open(path, "rb") as f:
        head = f.read(128)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise FileNotFoundError(
            f"{label} 파일이 실제 데이터가 아니라 Git LFS pointer입니다. "
            f"git lfs pull 또는 원본 데이터 복원이 필요합니다: {path}"
        )


def solar_position_kst(lat_deg: float, lon_deg: float, month: int, day: int, hour_kst: float) -> tuple[float, float]:
    """한국 표준시 기준 태양 고도각과 방위각을 계산한다."""
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = sum(days_per_month[:month]) + day

    declination = -23.45 * math.cos(math.radians(360.0 / 365.0 * (doy + 10)))
    b = math.radians(360.0 / 365.0 * (doy - 81))
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    solar_time = hour_kst + (lon_deg - 135.0) / 15.0 + eot_min / 60.0
    hour_angle = 15.0 * (solar_time - 12.0)

    lat_r = math.radians(lat_deg)
    dec_r = math.radians(declination)
    ha_r = math.radians(hour_angle)

    sin_elev = math.sin(lat_r) * math.sin(dec_r) + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    cos_elev = math.cos(math.radians(elevation))
    if cos_elev < 1e-9:
        return elevation, 0.0

    cos_az = max(
        -1.0,
        min(1.0, (math.sin(dec_r) - math.sin(lat_r) * sin_elev) / (math.cos(lat_r) * cos_elev)),
    )
    azimuth = math.degrees(math.acos(cos_az))
    if hour_angle > 0:
        azimuth = 360.0 - azimuth

    return elevation, azimuth


def load_buildings_and_shadow(
    roads_gdf: gpd.GeoDataFrame,
    data_dir: str | None = None,
    gangnam_code: str = "11680",
    date: tuple[int, int] = (1, 15),
    hours: list[int] | None = None,
) -> gpd.GeoDataFrame:
    """건물 높이와 시간대별 태양 위치를 반영해 도로별 그림자 지수를 계산한다."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if hours is None:
        hours = list(range(24))
    if not hours:
        raise ValueError("hours must contain at least one hour")

    print("[2/6] 건물 데이터 + 시간대별 그림자 지수 계산...")
    building_path = os.path.join(data_dir, "buildings", "AL_D162_11_20260115", "AL_D162_11_20260115.shp")
    require_real_data_file(building_path, "건물 Shapefile")
    bldg = gpd.read_file(building_path)
    gn_bldg = bldg[bldg["A39"].astype(str).str.startswith(gangnam_code)].copy()

    gn_bldg["height"] = pd.to_numeric(gn_bldg["A31"], errors="coerce").fillna(0)
    mask_zero = gn_bldg["height"] == 0
    gn_bldg.loc[mask_zero, "height"] = (
        pd.to_numeric(gn_bldg.loc[mask_zero, "A18"], errors="coerce").fillna(1) * 3
    )
    print(f"  → 강남구 건물: {len(gn_bldg)}개, 평균 높이: {gn_bldg['height'].mean():.1f}m")

    gn_bldg_5186 = gn_bldg[["height", "geometry"]].to_crs(epsg=5186)
    roads_5186 = roads_gdf[["LINK_ID", "geometry"]].copy().to_crs(epsg=5186)

    bldg_centroids = gn_bldg_5186.geometry.centroid
    bldg_cx = np.asarray(bldg_centroids.x, dtype=float)
    bldg_cy = np.asarray(bldg_centroids.y, dtype=float)
    heights = gn_bldg_5186["height"].values.astype(float)
    road_centroids = roads_5186.geometry.centroid
    road_cx = np.asarray(road_centroids.x, dtype=float)
    road_cy = np.asarray(road_centroids.y, dtype=float)

    from shapely import STRtree

    height_p95 = float(np.nanpercentile(heights, 95)) if len(heights) else 30.0
    max_shadow_m = float(np.clip(height_p95 * 12.0, 180.0, 800.0))
    tree = STRtree(list(bldg_centroids))
    search_radius_m = max(200.0, max_shadow_m)

    road_nearby = []
    for rx, ry in zip(road_cx, road_cy):
        idxs = np.array(tree.query(Point(rx, ry).buffer(search_radius_m)), dtype=int)
        if len(idxs):
            road_nearby.append((rx, ry, bldg_cx[idxs], bldg_cy[idxs], heights[idxs]))
        else:
            road_nearby.append((rx, ry, np.array([]), np.array([]), np.array([])))

    roads_lonlat = roads_gdf[["geometry"]].copy().to_crs(epsg=4326)
    road_centroids_lonlat = roads_lonlat.geometry.centroid
    road_lat = np.asarray(road_centroids_lonlat.y, dtype=float)
    road_lon = np.asarray(road_centroids_lonlat.x, dtype=float)
    if "road_width" in roads_gdf.columns:
        road_width = pd.to_numeric(roads_gdf["road_width"], errors="coerce").fillna(8).to_numpy(dtype=float)
    else:
        road_width = np.full(len(roads_gdf), 8.0, dtype=float)
    shadow_width_m = np.clip(road_width * 1.5 + 8.0, 14.0, 60.0)
    min_elevation = np.clip(2.0 + road_width / 12.0, 3.0, 8.0)

    shadow_count = np.zeros(len(road_cx), dtype=int)
    valid_slots = np.zeros(len(road_cx), dtype=int)
    print(f"  → 시간대별 그림자 계산: KST {hours[0]}~{hours[-1]}시 ({date[0]}월 {date[1]}일 기준)...")

    for hour in hours:
        solar = np.array(
            [solar_position_kst(lat, lon, date[0], date[1], hour) for lat, lon in zip(road_lat, road_lon)],
            dtype=float,
        )
        elevations = solar[:, 0]
        azimuths = solar[:, 1]
        valid_mask = elevations >= min_elevation
        valid_slots += valid_mask.astype(int)

        for road_idx, (rx, ry, bx_arr, by_arr, h_arr) in enumerate(road_nearby):
            if len(bx_arr) == 0 or not valid_mask[road_idx]:
                continue

            tan_elevation = math.tan(math.radians(elevations[road_idx]))
            shadow_azimuth_rad = math.radians((azimuths[road_idx] + 180.0) % 360.0)
            shadow_dx = math.sin(shadow_azimuth_rad)
            shadow_dy = math.cos(shadow_azimuth_rad)
            dx = rx - bx_arr
            dy = ry - by_arr
            shadow_len = np.minimum(h_arr / tan_elevation, max_shadow_m)
            dot = dx * shadow_dx + dy * shadow_dy
            perp = np.abs(dx * shadow_dy - dy * shadow_dx)

            if np.any((dot > 0) & (dot <= shadow_len) & (perp < shadow_width_m[road_idx])):
                shadow_count[road_idx] += 1

    shadow_index = shadow_count / np.maximum(valid_slots, 1)
    roads_gdf = roads_gdf.copy()
    roads_gdf["shadow_index"] = shadow_index

    print(
        f"  → 도로 위치별 태양각 사용 | 평균 유효 시간대: {valid_slots.mean():.1f}개 | "
        f"그림자 지수 평균: {shadow_index.mean():.3f}, 최대: {shadow_index.max():.3f}"
    )
    print(
        f"  → 종일 그림자(≥80%) 도로: {(shadow_index >= 0.8).sum()}개 | "
        f"절반+ 그림자(≥50%) 도로: {(shadow_index >= 0.5).sum()}개 / {len(shadow_index)}개 총"
    )
    return roads_gdf
