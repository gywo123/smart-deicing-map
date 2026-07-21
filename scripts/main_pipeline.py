"""
AI 제설 맵핑 시스템 — 전체 파이프라인
강남구 도로 결빙 위험도 예측 → 최적 제설 경로 도출
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import networkx as nx
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from scripts.route_visualization import create_navigation_simulation_map
    from scripts.shadow_utils import load_buildings_and_shadow as load_time_based_shadow
except ModuleNotFoundError:
    from route_visualization import create_navigation_simulation_map
    from shadow_utils import load_buildings_and_shadow as load_time_based_shadow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scripts' else SCRIPT_DIR
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MAPS_DIR = os.path.join(OUTPUTS_DIR, 'maps')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
OUTPUT_DATA_DIR = os.path.join(OUTPUTS_DIR, 'data')
CLEANED_DATA_DIR = os.path.join(OUTPUT_DATA_DIR, 'cleaned')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
DEFAULT_DEICING_COST_CONFIG_PATH = os.path.join(CONFIG_DIR, 'deicing_costs.json')

for directory in [MODELS_DIR, MAPS_DIR, FIGURES_DIR, OUTPUT_DATA_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

GN_LON_MIN, GN_LON_MAX = 127.01, 127.09
GN_LAT_MIN, GN_LAT_MAX = 37.47, 37.53
GANGNAM_CODE = '11680'

DEFAULT_DEICING_COST_CONFIG = {
    'unit_spread_kg_per_m2': 0.03,
    'material_cost_won_per_kg': 300.0,
    'labor_cost_won_per_km': 50000.0,
    'environmental_cost_won_per_kg': 0.0,
    'default_road_width_m': 8.0,
    'road_width_by_rank_m': {
        '101': 30.0,
        '102': 25.0,
        '103': 20.0,
        '104': 8.0,
        '105': 6.0,
        '106': 6.0,
        '107': 4.0,
        '108': 4.0,
    },
}


def load_deicing_cost_config(path=None):
    """제설 비용/살포량 설정을 JSON에서 읽고 누락값은 기본값으로 보완한다."""
    config_path = path or os.environ.get('DEICING_COST_CONFIG', DEFAULT_DEICING_COST_CONFIG_PATH)
    config = {
        **DEFAULT_DEICING_COST_CONFIG,
        'road_width_by_rank_m': DEFAULT_DEICING_COST_CONFIG['road_width_by_rank_m'].copy(),
    }
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        config.update({key: value for key, value in loaded.items() if key != 'road_width_by_rank_m'})
        if isinstance(loaded.get('road_width_by_rank_m'), dict):
            config['road_width_by_rank_m'].update({
                str(key): float(value)
                for key, value in loaded['road_width_by_rank_m'].items()
            })

    positive_keys = [
        'unit_spread_kg_per_m2',
        'material_cost_won_per_kg',
        'labor_cost_won_per_km',
        'default_road_width_m',
    ]
    for key in positive_keys:
        config[key] = float(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    config['environmental_cost_won_per_kg'] = float(config.get('environmental_cost_won_per_kg', 0.0))
    if config['environmental_cost_won_per_kg'] < 0:
        raise ValueError("environmental_cost_won_per_kg must be non-negative")
    return config


def ensure_output_dirs():
    """현재 루트 구조 기준 산출물 폴더를 항상 준비한다."""
    for directory in [MODELS_DIR, MAPS_DIR, FIGURES_DIR, OUTPUT_DATA_DIR, REPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)


def require_real_data_file(path, label):
    """Git LFS pointer나 누락 파일을 사람이 이해하기 쉬운 오류로 바꾼다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {path}")
    with open(path, "rb") as f:
        head = f.read(128)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise FileNotFoundError(
            f"{label} 파일이 실제 데이터가 아니라 Git LFS pointer입니다. "
            f"실제 파일을 받으려면 git lfs pull 또는 원본 데이터 복원이 필요합니다: {path}"
        )


def filter_drivable_links(gdf):
    """표준 링크 속성의 도로사용여부를 반영해 차량 통행 가능한 링크만 남긴다."""
    if 'ROAD_USE' not in gdf.columns:
        return gdf
    road_use = pd.to_numeric(gdf['ROAD_USE'], errors='coerce').fillna(0)
    return gdf[road_use == 0].copy()


def load_standard_node_positions():
    """MOCT_NODE의 실제 좌표를 노드 그래프 기준점으로 읽는다."""
    node_path = os.path.join(DATA_DIR, 'roads', '[2024-03-25]NODELINKDATA', 'MOCT_NODE.shp')
    if not os.path.exists(node_path):
        return {}
    require_real_data_file(node_path, "노드 Shapefile")
    nodes_raw = gpd.read_file(node_path)
    nodes_metric = nodes_raw.to_crs(epsg=5179)
    nodes_lonlat = nodes_raw.to_crs(epsg=4326)
    positions = {}
    for (_, metric_row), (_, lonlat_row) in zip(nodes_metric.iterrows(), nodes_lonlat.iterrows()):
        positions[str(metric_row['NODE_ID'])] = {
            'x': float(metric_row.geometry.x),
            'y': float(metric_row.geometry.y),
            'lon': float(lonlat_row.geometry.x),
            'lat': float(lonlat_row.geometry.y),
        }
    return positions


def snap_linestring_endpoints_to_node_positions(row, node_positions):
    """LineString 첫/끝점을 F_NODE/T_NODE 좌표와 맞춘다."""
    geom = row.geometry
    if geom is None or geom.is_empty or geom.geom_type != 'LineString':
        return geom
    coords = list(geom.coords)
    if len(coords) < 2:
        return geom
    f_node = node_positions.get(str(row['F_NODE']))
    t_node = node_positions.get(str(row['T_NODE']))
    if f_node:
        coords[0] = (f_node['lon'], f_node['lat'])
    if t_node:
        coords[-1] = (t_node['lon'], t_node['lat'])
    return LineString(coords)


def configure_korean_font():
    """저장되는 matplotlib 차트의 한글 라벨이 깨지지 않도록 한글 폰트를 설정한다."""
    font_candidates = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False


configure_korean_font()


# ============================================================
# 1. 도로 네트워크 전처리
# ============================================================
def load_roads():
    print("[1/6] 도로 네트워크 로딩...")
    clean_path = os.path.join(CLEANED_DATA_DIR, 'gangnam_roads_clean.geojson')
    if os.path.exists(clean_path):
        gn_links = gpd.read_file(clean_path).to_crs(epsg=4326)
        gn_links = filter_drivable_links(gn_links)
        print(f"  → 정제 도로 사용: {len(gn_links)}개")
        return gn_links

    road_path = os.path.join(DATA_DIR, 'roads', '[2024-03-25]NODELINKDATA', 'MOCT_LINK.shp')
    require_real_data_file(road_path, "도로 Shapefile")
    links = gpd.read_file(road_path)
    links = links.to_crs(epsg=4326)

    candidate_links = links.cx[GN_LON_MIN:GN_LON_MAX, GN_LAT_MIN:GN_LAT_MAX].copy()
    metric_centroids = candidate_links.to_crs(epsg=5186).geometry.centroid
    lonlat_centroids = gpd.GeoSeries(metric_centroids, crs='EPSG:5186').to_crs(epsg=4326)
    candidate_links['cent_lon'] = lonlat_centroids.x.values
    candidate_links['cent_lat'] = lonlat_centroids.y.values
    gn_links = candidate_links[
        (candidate_links['cent_lon'].between(GN_LON_MIN, GN_LON_MAX)) &
        (candidate_links['cent_lat'].between(GN_LAT_MIN, GN_LAT_MAX))
    ].copy()
    keep_cols = ['LINK_ID', 'F_NODE', 'T_NODE', 'LANES', 'ROAD_RANK', 'ROAD_TYPE',
                 'ROAD_NAME', 'ROAD_USE', 'CONNECT', 'MAX_SPD', 'REST_VEH',
                 'REST_W', 'REST_H', 'LENGTH', 'cent_lon', 'cent_lat', 'geometry']
    gn_links = gn_links[[col for col in keep_cols if col in gn_links.columns]].reset_index(drop=True)

    gn_links['LANES'] = pd.to_numeric(gn_links['LANES'], errors='coerce').fillna(2).astype(int)
    gn_links['LENGTH'] = pd.to_numeric(gn_links['LENGTH'], errors='coerce')
    gn_links = filter_drivable_links(gn_links)
    gn_links = gn_links[gn_links['LENGTH'] > 0].drop_duplicates('LINK_ID').reset_index(drop=True)
    gn_links['ROAD_RANK'] = gn_links['ROAD_RANK'].astype(str)

    print(f"  → 강남구 도로 링크: {len(gn_links)}개")
    return gn_links


# ============================================================
# 2. 건물 데이터 → 그림자 지수
# ============================================================
def load_buildings_and_shadow(roads_gdf):
    return load_time_based_shadow(roads_gdf, data_dir=DATA_DIR, gangnam_code=GANGNAM_CODE)


# ============================================================
# 3. 기상 데이터 전처리
# ============================================================
def load_weather():
    print("[3/6] 기상 데이터 로딩...")
    clean_path = os.path.join(CLEANED_DATA_DIR, 'weather_winter_clean.csv')
    if os.path.exists(clean_path):
        winter = pd.read_csv(clean_path)
        if '일시' in winter.columns:
            winter['일시'] = pd.to_datetime(winter['일시'], errors='coerce')
        print(f"  → 정제 겨울철 기상 데이터 사용: {len(winter)}행")
        print(f"  → 영하 비율: {(winter['temp'] <= 0).mean():.1%}")
        return winter

    asos_files = sorted(glob.glob(os.path.join(DATA_DIR, 'weather', 'OBS_ASOS_TIM_*.csv')))

    main_cols = ['일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
                 '적설(cm)', '지면온도(°C)']

    dfs = []
    for f in asos_files:
        df = pd.read_csv(f, encoding='cp949')
        if '기온(°C)' not in df.columns:
            print(f"  → 보조 OBS 파일 제외: {os.path.basename(f)}")
            continue
        available = [c for c in main_cols if c in df.columns]
        dfs.append(df[available])

    weather = pd.concat(dfs, ignore_index=True)
    weather['일시'] = pd.to_datetime(weather['일시'])
    before_dedup = len(weather)
    weather = weather.sort_values('일시').drop_duplicates('일시', keep='last')
    if len(weather) != before_dedup:
        print(f"  → 중복 시간 {before_dedup - len(weather)}행 제거")
    weather['month'] = weather['일시'].dt.month

    winter = weather[weather['month'].isin([11, 12, 1, 2])].copy()

    for col in ['강수량(mm)', '적설(cm)']:
        if col in winter.columns:
            winter[col] = pd.to_numeric(winter[col], errors='coerce').fillna(0)
    for col in ['기온(°C)', '습도(%)', '풍속(m/s)', '지면온도(°C)']:
        if col in winter.columns:
            winter[col] = pd.to_numeric(winter[col], errors='coerce')
            winter[col] = winter[col].ffill().bfill()

    winter = winter.rename(columns={
        '기온(°C)': 'temp', '강수량(mm)': 'precip', '풍속(m/s)': 'wind',
        '습도(%)': 'humidity', '적설(cm)': 'snow', '지면온도(°C)': 'ground_temp'
    })

    print(f"  → 겨울철 기상 데이터: {len(winter)}행")
    print(f"  → 영하 비율: {(winter['temp'] <= 0).mean():.1%}")
    return winter


# ============================================================
# 4. 유동인구 데이터 전처리 + 도로 매핑
# ============================================================
def load_population_and_map(roads_gdf):
    print("[4/6] 유동인구 데이터 로딩 + 도로 매핑...")
    clean_path = os.path.join(CLEANED_DATA_DIR, 'population_grid_clean.csv')
    if os.path.exists(clean_path):
        valid_grids = pd.read_csv(clean_path)
        print(f"  → 정제 생활인구 격자 사용: {len(valid_grids)}개")
    else:
        pop_dirs = [
            os.path.join(DATA_DIR, 'population', '250_LOCAL_RESD_202501'),
            os.path.join(DATA_DIR, 'population', '250_LOCAL_RESD_202512'),
        ]

        dfs = []
        for d in pop_dirs:
            for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
                df = pd.read_csv(f, encoding='cp949')
                df.columns = [c.strip().strip('"') for c in df.columns]
                dfs.append(df[['행정동코드', '250M격자', '시간', '생활인구합계']])

        pop = pd.concat(dfs, ignore_index=True)
        pop['행정동코드'] = pop['행정동코드'].astype(str)
        gn_pop = pop[pop['행정동코드'].str.startswith(GANGNAM_CODE)].copy()
        gn_pop['생활인구합계'] = pd.to_numeric(
            gn_pop['생활인구합계'].astype(str).str.replace('*', '0'), errors='coerce'
        ).fillna(0)

        grid_avg = gn_pop.groupby('250M격자')['생활인구합계'].mean().reset_index()
        grid_avg.columns = ['grid_id', 'avg_pop']

        import pyproj
        transformer = pyproj.Transformer.from_crs('EPSG:5179', 'EPSG:4326', always_xy=True)

        grid_lons_list, grid_lats_list = [], []
        for gid in grid_avg['grid_id']:
            gid_str = str(gid)
            nums = gid_str.replace('다사', '')
            if len(nums) >= 8:
                x_5179 = 900000 + int(nums[:4]) * 10
                y_5179 = 1900000 + int(nums[4:8]) * 10
                lon, lat = transformer.transform(x_5179, y_5179)
                grid_lons_list.append(lon)
                grid_lats_list.append(lat)
            else:
                grid_lons_list.append(0)
                grid_lats_list.append(0)

        grid_avg['grid_lon'] = grid_lons_list
        grid_avg['grid_lat'] = grid_lats_list

        valid_grids = grid_avg[
            (grid_avg['grid_lon'] >= GN_LON_MIN) & (grid_avg['grid_lon'] <= GN_LON_MAX) &
            (grid_avg['grid_lat'] >= GN_LAT_MIN) & (grid_avg['grid_lat'] <= GN_LAT_MAX)
        ].copy()

        if len(valid_grids) == 0:
            valid_grids = grid_avg.copy()

        pop_max = valid_grids['avg_pop'].max()
        valid_grids['pop_weight'] = valid_grids['avg_pop'] / max(pop_max, 1)

    pop_weights = []
    grid_lons = valid_grids['grid_lon'].values
    grid_lats = valid_grids['grid_lat'].values
    grid_pops = valid_grids['pop_weight'].values

    for _, row in roads_gdf.iterrows():
        dists = np.sqrt((grid_lons - row['cent_lon'])**2 + (grid_lats - row['cent_lat'])**2)
        near_mask = dists < 0.005  # ~500m
        if near_mask.any():
            weights = 1.0 / (dists[near_mask] + 1e-6)
            pop_w = np.average(grid_pops[near_mask], weights=weights)
        else:
            nearest = np.argmin(dists)
            pop_w = grid_pops[nearest] * 0.5
        pop_weights.append(pop_w)

    roads_gdf['pop_weight'] = pop_weights

    print(f"  → 강남구 격자: {len(valid_grids)}개")
    print(f"  → 도로 유동인구 가중치: 평균 {np.mean(pop_weights):.3f}, "
          f"최대 {np.max(pop_weights):.3f}")
    return roads_gdf


# ============================================================
# 6. 우선순위 산정 + 비용 계산
# ============================================================
def calculate_priority(roads_gdf):
    print("\n우선순위 산정...")
    roads_gdf['priority_score'] = roads_gdf['risk'] * (0.7 + 0.3 * roads_gdf['pop_weight'])

    cost_config = load_deicing_cost_config()
    road_width_map = cost_config['road_width_by_rank_m']
    roads_gdf['road_width'] = roads_gdf['ROAD_RANK'].astype(str).map(road_width_map).fillna(
        cost_config['default_road_width_m']
    )
    roads_gdf['area'] = roads_gdf['LENGTH'] * roads_gdf['road_width']

    unit_spread = cost_config['unit_spread_kg_per_m2']
    material_cost = cost_config['material_cost_won_per_kg']
    environmental_cost = cost_config['environmental_cost_won_per_kg']
    labor_per_km = cost_config['labor_cost_won_per_km']
    roads_gdf['deicing_kg'] = roads_gdf['area'] * unit_spread
    roads_gdf['deicing_cost'] = (
        roads_gdf['deicing_kg'] * (material_cost + environmental_cost) +
        roads_gdf['LENGTH'] / 1000 * labor_per_km
    )

    print(
        f"  → 비용 설정: 살포량={unit_spread:.3f}kg/m², "
        f"재료비={material_cost:.0f}원/kg, 인건비={labor_per_km:.0f}원/km"
    )
    print(f"  → 총 제설 비용 (전체): {roads_gdf['deicing_cost'].sum()/1e6:.1f}백만원")
    print(f"  → 우선순위 점수: 평균={roads_gdf['priority_score'].mean():.3f}, "
          f"최대={roads_gdf['priority_score'].max():.3f}")
    return roads_gdf


# ============================================================
# 9. Knapsack + Budgeted Maximum Coverage 최적화
# ============================================================
def build_coverage_sets(roads_gdf, cell_size=250, buffer_m=125):
    """Create road-to-grid coverage sets for budgeted maximum coverage."""
    roads_5179 = roads_gdf[['LINK_ID', 'priority_score', 'pop_weight', 'geometry']].copy().to_crs(epsg=5179)
    road_cells = {}
    cell_weights = {}

    pop_max = max(float(roads_gdf['pop_weight'].max()), 1e-9)

    for idx, row in roads_5179.iterrows():
        geom_buffer = row.geometry.buffer(buffer_m)
        minx, miny, maxx, maxy = geom_buffer.bounds
        x0, x1 = int(np.floor(minx / cell_size)), int(np.floor(maxx / cell_size))
        y0, y1 = int(np.floor(miny / cell_size)), int(np.floor(maxy / cell_size))

        cells = set()
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                center = Point((gx + 0.5) * cell_size, (gy + 0.5) * cell_size)
                if geom_buffer.contains(center) or geom_buffer.intersects(center):
                    cells.add((gx, gy))

        if not cells:
            centroid = row.geometry.centroid
            cells.add((int(np.floor(centroid.x / cell_size)),
                       int(np.floor(centroid.y / cell_size))))

        road_cells[idx] = cells
        pop_factor = 0.65 + 0.35 * (float(row['pop_weight']) / pop_max)
        road_value = float(row['priority_score']) * pop_factor
        for cell in cells:
            cell_weights[cell] = max(cell_weights.get(cell, 0.0), road_value)

    return road_cells, cell_weights


def hybrid_bmc_knapsack_optimize(
    roads_gdf,
    budget_ratio=0.4,
    coverage_weight=0.55,
    knapsack_weight=0.45,
):
    print("\nKnapsack + Budgeted Maximum Coverage 최적화...")

    total_cost = roads_gdf['deicing_cost'].sum()
    budget = total_cost * budget_ratio
    print(f"  예산: {budget/1e6:.1f}백만원 (전체의 {budget_ratio:.0%})")

    road_cells, cell_weights = build_coverage_sets(roads_gdf)
    max_priority = max(float(roads_gdf['priority_score'].max()), 1e-9)
    initial_coverage_gain = {
        idx: sum(cell_weights[cell] for cell in cells)
        for idx, cells in road_cells.items()
    }
    max_coverage_gain = max(initial_coverage_gain.values()) if initial_coverage_gain else 1.0
    max_coverage_gain = max(max_coverage_gain, 1e-9)

    selected_indices = []
    selected_ids = []
    covered_cells = set()
    remaining_budget = budget
    candidates = set(roads_gdf.index)

    while candidates:
        best_idx = None
        best_score = -1.0

        for idx in candidates:
            cost = float(roads_gdf.at[idx, 'deicing_cost'])
            if cost > remaining_budget:
                continue

            new_cells = road_cells[idx] - covered_cells
            coverage_gain = sum(cell_weights[cell] for cell in new_cells)
            coverage_norm = coverage_gain / max_coverage_gain
            priority_norm = float(roads_gdf.at[idx, 'priority_score']) / max_priority

            hybrid_value = (
                coverage_weight * coverage_norm +
                knapsack_weight * priority_norm
            )
            score = hybrid_value / max(cost, 1.0)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        selected_ids.append(roads_gdf.at[best_idx, 'LINK_ID'])
        covered_cells.update(road_cells[best_idx])
        remaining_budget -= float(roads_gdf.at[best_idx, 'deicing_cost'])
        candidates.remove(best_idx)

    spent = budget - remaining_budget
    roads_gdf['selected'] = roads_gdf.index.isin(selected_indices).astype(int)
    roads_gdf['coverage_cells'] = roads_gdf.index.map(lambda idx: len(road_cells.get(idx, set())))
    roads_gdf['selection_method'] = np.where(
        roads_gdf['selected'] == 1,
        'hybrid_bmc_knapsack',
        'not_selected'
    )

    print(f"  → 커버 격자: {len(covered_cells)}개 / 전체 {len(cell_weights)}개")
    print(f"  → 선택 도로: {len(selected_ids)}개 / {len(roads_gdf)}개")
    print(f"  → 사용 예산: {spent/1e6:.1f}백만원")
    print(f"  → 잔여 예산: {remaining_budget/1e6:.2f}백만원")

    return roads_gdf, selected_ids


# ============================================================
# 10. VRP 경로 생성 (다중 제설차)
# ============================================================
def choose_vehicle_count(sel, min_vehicles=2, max_vehicles=8):
    """작업량과 연결 컴포넌트 규모에 따라 필요한 제설차 대수를 자동 추정한다."""
    total_length_km = float(sel['LENGTH'].sum() / 1000.0)
    total_deicing_kg = float(sel['deicing_kg'].sum())
    by_length = int(np.ceil(total_length_km / 55.0))
    by_load = int(np.ceil(total_deicing_kg / 12000.0))
    by_road_count = int(np.ceil(len(sel) / 280.0))

    selected_graph = nx.Graph()
    for _, row in sel.iterrows():
        selected_graph.add_edge(str(row['F_NODE']), str(row['T_NODE']))
    significant_components = 0
    if selected_graph.number_of_edges() > 0:
        component_nodes = list(nx.connected_components(selected_graph))
        for nodes in component_nodes:
            edge_count = selected_graph.subgraph(nodes).number_of_edges()
            if edge_count >= 15:
                significant_components += 1

    estimated = max(min_vehicles, by_length, by_load, by_road_count, min(significant_components, max_vehicles))
    return int(np.clip(estimated, min_vehicles, max_vehicles))


def priority_biased_nearest_route(vehicle_df, depot_xy):
    coords = vehicle_df[['x', 'y']].values
    priority = vehicle_df['priority_score'].values
    priority_norm = priority / max(float(priority.max()), 1e-9)
    n = len(vehicle_df)

    radial_dist = np.sqrt(((coords - depot_xy) ** 2).sum(axis=1))
    radial_norm = radial_dist / max(float(radial_dist.max()), 1e-9)
    start_score = 0.70 * radial_norm + 0.30 * priority_norm
    start_pos = int(np.argmax(start_score))
    visited = np.zeros(n, dtype=bool)
    route = [start_pos]
    visited[start_pos] = True

    for _ in range(n - 1):
        curr = route[-1]
        candidates = np.where(~visited)[0]
        if len(candidates) == 0:
            break

        deltas = coords[candidates] - coords[curr]
        dists = np.sqrt((deltas * deltas).sum(axis=1))
        priority_bonus = 0.55 + 0.45 * priority_norm[candidates]
        next_pos = int(candidates[np.argmin(dists / priority_bonus)])
        route.append(next_pos)
        visited[next_pos] = True

    route_xy = coords[route]
    depot_to_start = float(np.linalg.norm(route_xy[0] - depot_xy))
    inner_dist = float(np.sqrt(((route_xy[1:] - route_xy[:-1]) ** 2).sum(axis=1)).sum()) if len(route_xy) > 1 else 0.0
    return_to_depot = float(np.linalg.norm(route_xy[-1] - depot_xy))
    total_dist_m = depot_to_start + inner_dist + return_to_depot

    return route, total_dist_m


def rebalance_vehicle_loads(sel, n_vehicles, vehicle_capacity_kg):
    labels = sel['vehicle_id'].values.copy()

    for _ in range(500):
        loads = sel.groupby(labels)['deicing_kg'].sum().reindex(range(n_vehicles), fill_value=0.0)
        overloaded = loads[loads > vehicle_capacity_kg].sort_values(ascending=False)
        if overloaded.empty:
            break

        moved = False
        for vehicle_id in overloaded.index:
            vehicle_rows = sel[labels == vehicle_id].copy()
            if len(vehicle_rows) <= 1:
                continue

            center = vehicle_rows[['x', 'y']].mean().values
            vehicle_rows['dist_from_center'] = np.sqrt(
                ((vehicle_rows[['x', 'y']].values - center) ** 2).sum(axis=1)
            )
            vehicle_rows = vehicle_rows.sort_values('dist_from_center', ascending=False)

            for row_idx, row in vehicle_rows.iterrows():
                target_candidates = loads[loads + row['deicing_kg'] <= vehicle_capacity_kg]
                target_candidates = target_candidates.drop(index=vehicle_id, errors='ignore')
                if target_candidates.empty:
                    continue

                target_id = int(target_candidates.idxmin())
                labels[row_idx] = target_id
                moved = True
                break

            if moved:
                break

        if not moved:
            break

    return labels


def build_road_network_router(roads_gdf):
    """F_NODE/T_NODE 차량흐름 방향을 반영한 표준 노드·링크 그래프를 만든다."""
    node_positions = load_standard_node_positions()
    router_cols = ['LINK_ID', 'F_NODE', 'T_NODE', 'LENGTH', 'ROAD_USE', 'geometry']
    router_source = roads_gdf[[col for col in router_cols if col in roads_gdf.columns]].copy().to_crs(epsg=4326)

    raw_road_path = os.path.join(DATA_DIR, 'roads', '[2024-03-25]NODELINKDATA', 'MOCT_LINK.shp')
    try:
        if os.path.exists(raw_road_path):
            require_real_data_file(raw_road_path, "도로 Shapefile")
            raw_roads = gpd.read_file(raw_road_path).to_crs(epsg=4326)
            raw_roads = filter_drivable_links(raw_roads)
            margin = 0.015
            raw_roads = raw_roads.cx[
                GN_LON_MIN - margin:GN_LON_MAX + margin,
                GN_LAT_MIN - margin:GN_LAT_MAX + margin,
            ].copy()
            raw_roads = raw_roads[[col for col in router_cols if col in raw_roads.columns]]
            raw_roads['LENGTH'] = pd.to_numeric(raw_roads['LENGTH'], errors='coerce')
            router_source = pd.concat([router_source, raw_roads], ignore_index=True)
            router_source = router_source.drop_duplicates('LINK_ID').dropna(subset=['LENGTH'])
    except Exception as exc:
        print(f"  → 원본 도로망 보강 생략: {exc}")

    router_source = filter_drivable_links(router_source)
    if node_positions:
        router_source['geometry'] = router_source.apply(
            lambda row: snap_linestring_endpoints_to_node_positions(row, node_positions),
            axis=1,
        )

    roads_5179 = router_source[['LINK_ID', 'F_NODE', 'T_NODE', 'LENGTH', 'geometry']].copy().to_crs(epsg=5179)
    roads_4326 = router_source[['LINK_ID', 'F_NODE', 'T_NODE', 'LENGTH', 'geometry']].copy().to_crs(epsg=4326)
    graph = nx.DiGraph()

    for (_, metric_row), (_, lonlat_row) in zip(roads_5179.iterrows(), roads_4326.iterrows()):
        metric_coords = list(metric_row.geometry.coords)
        lonlat_coords = list(lonlat_row.geometry.coords)
        if len(metric_coords) < 2:
            continue

        node_a = str(metric_row['F_NODE'])
        node_b = str(metric_row['T_NODE'])
        start_pos = node_positions.get(node_a)
        end_pos = node_positions.get(node_b)
        if start_pos:
            graph.add_node(node_a, **start_pos)
        else:
            x1, y1 = metric_coords[0]
            lon_a, lat_a = lonlat_coords[0]
            graph.add_node(node_a, x=float(x1), y=float(y1), lon=float(lon_a), lat=float(lat_a))
        if end_pos:
            graph.add_node(node_b, **end_pos)
        else:
            x2, y2 = metric_coords[-1]
            lon_b, lat_b = lonlat_coords[-1]
            graph.add_node(node_b, x=float(x2), y=float(y2), lon=float(lon_b), lat=float(lat_b))

        length_m = float(metric_row['LENGTH']) if pd.notna(metric_row['LENGTH']) else float(metric_row.geometry.length)
        edge_coords = [(float(lon), float(lat)) for lon, lat in lonlat_coords]
        if graph.has_edge(node_a, node_b):
            if length_m < graph[node_a][node_b]['weight']:
                graph[node_a][node_b]['weight'] = length_m
                graph[node_a][node_b]['coords'] = edge_coords
                graph[node_a][node_b]['link_id'] = str(metric_row['LINK_ID'])
        else:
            graph.add_edge(node_a, node_b, weight=length_m, coords=edge_coords, link_id=str(metric_row['LINK_ID']))

    node_keys = list(graph.nodes)
    node_xy = np.asarray([[graph.nodes[node]['x'], graph.nodes[node]['y']] for node in node_keys], dtype=float)

    def nearest_node(x, y):
        if len(node_xy) == 0:
            return None
        deltas = node_xy - np.array([x, y], dtype=float)
        return node_keys[int(np.argmin((deltas * deltas).sum(axis=1)))]

    return graph, nearest_node


def append_unique_coords(target, coords):
    """연속 좌표 배열에 중복 끝점을 만들지 않고 좌표를 붙인다."""
    for coord in coords:
        coord = [float(coord[0]), float(coord[1])]
        if not target or target[-1] != coord:
            target.append(coord)


def path_nodes_to_coords(graph, path_nodes):
    """최단경로 노드열을 실제 링크 segment 좌표열로 변환한다."""
    coords = []
    for start, end in zip(path_nodes[:-1], path_nodes[1:]):
        edge_coords = graph[start][end].get('coords')
        if not edge_coords:
            start_attrs = graph.nodes[start]
            end_attrs = graph.nodes[end]
            edge_coords = [
                (float(start_attrs['lon']), float(start_attrs['lat'])),
                (float(end_attrs['lon']), float(end_attrs['lat'])),
            ]

        start_attrs = graph.nodes[start]
        expected_start = [float(start_attrs['lon']), float(start_attrs['lat'])]
        segment = [[float(lon), float(lat)] for lon, lat in edge_coords]
        if segment and segment[0] != expected_start:
            segment = list(reversed(segment))
        append_unique_coords(coords, segment)
    return coords


def road_network_movement_path(route_roads, graph, nearest_node):
    """차량 이동/도포 전체 궤적을 실제 도로망과 링크 geometry만으로 만든다."""
    if len(route_roads) == 0 or graph.number_of_nodes() == 0:
        return [], 0.0, 0

    route_metric = route_roads[['F_NODE', 'T_NODE', 'geometry']].copy().to_crs(epsg=5179)
    route_lonlat = route_roads[['F_NODE', 'T_NODE', 'geometry']].copy().to_crs(epsg=4326)
    movement_coords = []
    total_distance_m = 0.0
    skipped_connectors = 0
    current_node = None

    for (_, metric_row), (_, lonlat_row) in zip(route_metric.iterrows(), route_lonlat.iterrows()):
        metric_geom = metric_row.geometry
        lonlat_geom = lonlat_row.geometry
        metric_coords = list(metric_geom.coords)
        lonlat_coords = [[float(lon), float(lat)] for lon, lat in lonlat_geom.coords]
        if len(metric_coords) < 2 or len(lonlat_coords) < 2:
            continue

        forward_start = str(metric_row['F_NODE'])
        forward_end = str(metric_row['T_NODE'])
        if forward_start not in graph:
            forward_start = nearest_node(metric_coords[0][0], metric_coords[0][1])
        if forward_end not in graph:
            forward_end = nearest_node(metric_coords[-1][0], metric_coords[-1][1])
        if forward_start is None or forward_end is None:
            continue

        service_options = [(forward_start, forward_end, lonlat_coords)]

        if current_node is None:
            start_node, end_node, service_coords = service_options[0]
        else:
            ranked_options = []
            for start_node_option, end_node_option, coords_option in service_options:
                try:
                    connector_len = float(nx.shortest_path_length(graph, current_node, start_node_option, weight='weight'))
                    ranked_options.append((connector_len, start_node_option, end_node_option, coords_option))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            if not ranked_options:
                skipped_connectors += 1
                continue

            _, start_node, end_node, service_coords = min(ranked_options, key=lambda item: item[0])
            connector_nodes = nx.shortest_path(graph, current_node, start_node, weight='weight')
            connector_coords = path_nodes_to_coords(graph, connector_nodes)
            append_unique_coords(movement_coords, connector_coords)
            total_distance_m += float(nx.shortest_path_length(graph, current_node, start_node, weight='weight'))

        append_unique_coords(movement_coords, service_coords)
        total_distance_m += float(metric_geom.length)
        current_node = end_node

    return movement_coords, total_distance_m, skipped_connectors


def vrp_route(roads_gdf, selected_ids, n_vehicles=None):
    print("\nVRP 경로 도출 (다중 제설차 휴리스틱)...")

    sel = roads_gdf[roads_gdf['LINK_ID'].isin(selected_ids)].copy()
    if len(sel) == 0:
        print("  → 선택 도로가 없어 VRP를 건너뜁니다.")
        return [], [], []

    sel['orig_index'] = sel.index
    sel = sel.reset_index(drop=True)
    sel_metric = sel[['LINK_ID', 'geometry']].copy().to_crs(epsg=5179)
    centroids = sel_metric.geometry.centroid
    sel['x'] = centroids.x.values
    sel['y'] = centroids.y.values
    if 'deicing_kg' not in sel.columns:
        cost_config = load_deicing_cost_config()
        sel['deicing_kg'] = sel['area'] * cost_config['unit_spread_kg_per_m2']

    if n_vehicles is None or n_vehicles == 'auto':
        n_vehicles = choose_vehicle_count(sel)
        print(f"  → 자동 산정 제설차: {n_vehicles}대")
    n_vehicles = min(int(n_vehicles), len(sel))

    coords = sel[['x', 'y']].values
    kmeans = KMeans(n_clusters=n_vehicles, random_state=42, n_init=10)
    sel['vehicle_id'] = kmeans.fit_predict(coords)

    total_deicing_kg = float(sel['deicing_kg'].sum())
    vehicle_capacity_kg = total_deicing_kg / n_vehicles * 1.15
    sel['vehicle_id'] = rebalance_vehicle_loads(sel, n_vehicles, vehicle_capacity_kg)

    depot_xy = coords.mean(axis=0)
    colors = ['#1565c0', '#e53935', '#2e7d32', '#8e24aa',
              '#f57c00', '#00897b', '#6d4c41', '#3949ab']
    road_graph, nearest_road_node = build_road_network_router(roads_gdf)

    vehicle_route_roads = []
    vehicle_route_coords = []
    vehicle_meta = []

    roads_gdf['vehicle_id'] = 0
    roads_gdf['route_order'] = np.nan

    for vehicle_id in range(n_vehicles):
        vehicle_df = sel[sel['vehicle_id'] == vehicle_id].copy().reset_index(drop=True)
        if len(vehicle_df) == 0:
            continue

        route, dist_m = priority_biased_nearest_route(vehicle_df, depot_xy)
        route_roads = vehicle_df.iloc[route].reset_index(drop=True)
        route_coords = route_roads[['cent_lon', 'cent_lat']].values.tolist()
        movement_coords, movement_dist_m, skipped_connectors = road_network_movement_path(
            route_roads,
            road_graph,
            nearest_road_node,
        )
        if movement_coords:
            dist_m = movement_dist_m

        original_indices = route_roads['orig_index'].values
        roads_gdf.loc[original_indices, 'vehicle_id'] = vehicle_id + 1
        roads_gdf.loc[original_indices, 'route_order'] = np.arange(1, len(route_roads) + 1)

        meta = {
            'name': f'제설차 {vehicle_id + 1}',
            'color': colors[vehicle_id % len(colors)],
            'roads': int(len(route_roads)),
            'distance_km': round(dist_m / 1000, 1),
            'movement_coords': movement_coords,
            'movement_points': int(len(movement_coords)),
            'skipped_connectors': int(skipped_connectors),
            'avg_risk': float(route_roads['risk'].mean()),
            'total_cacl2_kg': float(route_roads['deicing_kg'].sum()),
            'vehicle_capacity_kg': float(vehicle_capacity_kg),
            'load_pct': float(route_roads['deicing_kg'].sum() / vehicle_capacity_kg),
            'budget_million_won': float(route_roads['deicing_cost'].sum() / 1e6),
            'start_lat': float(movement_coords[0][1]) if movement_coords else float(route_coords[0][1]),
            'start_lon': float(movement_coords[0][0]) if movement_coords else float(route_coords[0][0]),
        }

        vehicle_route_roads.append(route_roads)
        vehicle_route_coords.append(route_coords)
        vehicle_meta.append(meta)

        print(f"  {meta['name']}: {meta['roads']}개 도로, "
              f"{meta['distance_km']}km, 적재율={meta['load_pct']:.0%}, "
              f"평균 위험도={meta['avg_risk']:.3f}")
        if skipped_connectors:
            print(f"    - 도로망 연결 실패 구간 {skipped_connectors}개는 직선 이동선 없이 제외")

    total_dist = sum(v['distance_km'] for v in vehicle_meta)
    print(f"\n  → 전체 VRP 경로: {total_dist:.1f}km ({len(vehicle_meta)}대 제설차)")
    return vehicle_route_roads, vehicle_route_coords, vehicle_meta


# ============================================================
# 10-1. VRP 네비게이션 리포트 생성
# ============================================================
def generate_navigation_report(roads_gdf, vehicle_route_roads, vehicle_route_coords, vehicle_meta, results=None):
    print("\nVRP 네비게이션 리포트 생성...")

    total_roads = len(roads_gdf)
    total_cost = roads_gdf['deicing_cost'].sum()
    total_risk_sum = roads_gdf['risk'].sum()

    rank_map = {'101': '고속도로', '102': '도시고속', '103': '일반국도',
                '104': '특별시도', '105': '광역시도', '106': '지방도',
                '107': '시군도', '108': '이면도로'}

    vehicle_reports = []
    for zi, (route_roads, route_coords, meta) in enumerate(
            zip(vehicle_route_roads, vehicle_route_coords, vehicle_meta)):

        segments = []
        cumul_dist = 0.0

        for i in range(len(route_roads)):
            row = route_roads.iloc[i]
            road_type = rank_map.get(str(row['ROAD_RANK']), '기타')

            if i > 0:
                seg_dist = np.sqrt(
                    (route_coords[i][0] - route_coords[i-1][0])**2 +
                    (route_coords[i][1] - route_coords[i-1][1])**2
                ) * 111000
                cumul_dist += seg_dist

            bearing = ""
            if i < len(route_roads) - 1:
                dx = route_coords[i+1][0] - route_coords[i][0]
                dy = route_coords[i+1][1] - route_coords[i][1]
                angle = np.degrees(np.arctan2(dx, dy)) % 360
                if angle < 22.5 or angle >= 337.5: bearing = "북"
                elif angle < 67.5: bearing = "북동"
                elif angle < 112.5: bearing = "동"
                elif angle < 157.5: bearing = "남동"
                elif angle < 202.5: bearing = "남"
                elif angle < 247.5: bearing = "남서"
                elif angle < 292.5: bearing = "서"
                else: bearing = "북서"

            segments.append({
                'order': i + 1,
                'link_id': str(row['LINK_ID']),
                'road_type': road_type,
                'lanes': int(row['LANES']),
                'length_m': float(round(row['LENGTH'], 1)),
                'risk': float(round(float(row['risk']), 3)),
                'deicing_kg': float(round(float(row['area']) * 0.03, 1)),
                'lat': float(round(route_coords[i][1], 6)),
                'lon': float(round(route_coords[i][0], 6)),
                'direction': bearing,
                'cumul_dist_m': float(round(cumul_dist, 0)),
            })

        vehicle_reports.append({
            'vehicle': meta['name'],
            'color': meta['color'],
            'summary': {
                'roads': meta['roads'],
                'distance_km': meta['distance_km'],
                'avg_risk': round(meta['avg_risk'], 3),
                'cacl2_kg': round(meta['total_cacl2_kg'], 1),
                'capacity_kg': round(meta['vehicle_capacity_kg'], 1),
                'load_pct': round(meta['load_pct'], 3),
                'budget_million_won': round(meta['budget_million_won'], 2),
            },
            'start': {'lat': segments[0]['lat'], 'lon': segments[0]['lon']},
            'end': {'lat': segments[-1]['lat'], 'lon': segments[-1]['lon']},
            'route': segments,
        })

    selected_count = sum(m['roads'] for m in vehicle_meta)
    total_dist = sum(m['distance_km'] for m in vehicle_meta)
    selected_cost = sum(
        df['deicing_cost'].sum() for df in vehicle_route_roads
    )

    if results:
        risk_msg = (f"{results['baseline']['risk_coverage']*100:.1f}% → "
                    f"{results['ai']['risk_coverage']*100:.1f}% "
                    f"(+{results['improvement']['risk_coverage_pp']:.1f}%p)")
        cacl2_msg = (f"{results['baseline']['cacl2_tons']:.1f}톤 → "
                     f"{results['ai']['cacl2_tons']:.1f}톤 "
                     f"(-{results['improvement']['cacl2_reduction_pct']:.1f}%)")
        pop_msg = (f"{results['baseline']['pop_coverage']*100:.1f}% → "
                   f"{results['ai']['pop_coverage']*100:.1f}% "
                   f"(+{results['improvement']['pop_coverage_pp']:.1f}%p)")
    else:
        risk_msg = '기존 대비 위험구간 커버율 개선'
        cacl2_msg = '기존 대비 염화칼슘 사용량 절감'
        pop_msg = '기존 대비 유동인구 보호율 개선'

    report = {
        'system': 'AI 제설 맵핑 — VRP 차량별 제설 네비게이션',
        'area': '서울특별시 강남구',
        'selection_algorithm': 'Knapsack + Budgeted Maximum Coverage Hybrid',
        'routing_algorithm': 'Multi-vehicle VRP heuristic',
        'overview': {
            'total_roads': total_roads,
            'selected_roads': selected_count,
            'vehicles': len(vehicle_meta),
            'total_route_km': round(total_dist, 1),
            'budget_million_won': round(float(selected_cost) / 1e6, 1),
        },
        'improvement': {
            'message': '이 경로를 따라 제설 시 기존 대비 개선 효과',
            'risk_coverage': risk_msg,
            'cacl2_reduction': cacl2_msg,
            'pop_coverage': pop_msg,
        },
        'vehicles': vehicle_reports,
    }

    out_path = os.path.join(REPORTS_DIR, 'route_navigation.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → route_navigation.json 저장")

    print(f"\n{'='*60}")
    print("  AI 최적 제설 경로 — VRP 차량별 네비게이션")
    print(f"{'='*60}")
    print(f"  총 경로: {total_dist:.1f}km | 도로: {selected_count}개 | "
          f"예산: {float(selected_cost)/1e6:.1f}백만원 | 차량: {len(vehicle_meta)}대")
    print(f"\n  [이 경로를 따르면]")
    print(f"  • 위험구간 커버: {risk_msg}")
    print(f"  • 염화칼슘:     {cacl2_msg}")
    print(f"  • 유동인구 보호: {pop_msg}")

    for vr in vehicle_reports:
        print(f"\n  ── {vr['vehicle']} ({vr['summary']['roads']}개 도로, "
              f"{vr['summary']['distance_km']}km, 적재율 {vr['summary']['load_pct']:.0%}) ──")
        for seg in vr['route'][:5]:
            print(f"    {seg['order']:>3}. → {seg['direction']:<3} "
                  f"{seg['road_type']} {seg['lanes']}차선 {seg['length_m']:.0f}m "
                  f"[위험도 {seg['risk']:.2f}] "
                  f"살포 {seg['deicing_kg']:.0f}kg")
        if len(vr['route']) > 5:
            print(f"    ... 외 {len(vr['route'])-5}개 구간")

    return report


# ============================================================
# 11. Folium 시각화 (VRP 네비게이션 지도 포함)
# ============================================================
def create_maps(roads_gdf, route_coords, vehicle_route_roads=None, vehicle_route_coords=None, vehicle_meta=None, results=None):
    ensure_output_dirs()
    print("\n지도 시각화 생성...")
    center = [roads_gdf['cent_lat'].mean(), roads_gdf['cent_lon'].mean()]

    def road_latlon(row):
        return [(c[1], c[0]) for c in row.geometry.coords]

    def first_route_point(route_roads_z):
        if len(route_roads_z) == 0:
            return None
        row = route_roads_z.iloc[0]
        return [float(row['cent_lat']), float(row['cent_lon'])]

    def last_route_point(route_roads_z):
        if len(route_roads_z) == 0:
            return None
        row = route_roads_z.iloc[-1]
        return [float(row['cent_lat']), float(row['cent_lon'])]

    # --- 1) 결빙 위험도 맵 ---
    m1 = folium.Map(location=center, zoom_start=14, tiles='cartodbpositron')

    def risk_color(risk):
        if risk >= 0.8: return '#d32f2f'
        if risk >= 0.6: return '#f57c00'
        if risk >= 0.4: return '#fbc02d'
        if risk >= 0.2: return '#388e3c'
        return '#1565c0'

    for _, row in roads_gdf.iterrows():
        coords = list(row.geometry.coords)
        folium.PolyLine(
            locations=[(c[1], c[0]) for c in coords],
            color=risk_color(row['risk']),
            weight=3 if row['risk'] >= 0.5 else 2,
            opacity=0.8,
            tooltip=(f"LINK: {row['LINK_ID']}<br>"
                     f"위험도: {row['risk']:.2f}<br>"
                     f"우선순위: {row['priority_score']:.2f}<br>"
                     f"그림자지수: {row['shadow_index']:.3f}<br>"
                     f"차선: {row['LANES']}")
        ).add_to(m1)

    legend_html = '''
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:10px; border-radius:5px; border:1px solid #ccc;
         font-size:12px;">
    <b>결빙 위험도</b><br>
    <span style="color:#d32f2f">■</span> 매우 높음 (≥0.8)<br>
    <span style="color:#f57c00">■</span> 높음 (0.6~0.8)<br>
    <span style="color:#fbc02d">■</span> 보통 (0.4~0.6)<br>
    <span style="color:#388e3c">■</span> 낮음 (0.2~0.4)<br>
    <span style="color:#1565c0">■</span> 매우 낮음 (<0.2)
    </div>
    '''
    m1.get_root().html.add_child(folium.Element(legend_html))
    m1.save(os.path.join(MAPS_DIR, 'map_freezing_risk.html'))
    print("  → map_freezing_risk.html 저장")

    # --- 2) 최적 제설 경로 맵 ---
    m2 = folium.Map(location=center, zoom_start=14, tiles='cartodbpositron')

    for _, row in roads_gdf.iterrows():
        coords = list(row.geometry.coords)
        if row['selected'] == 1:
            color = '#e53935'
            weight = 4
        else:
            color = '#bdbdbd'
            weight = 1
        folium.PolyLine(
            locations=[(c[1], c[0]) for c in coords],
            color=color, weight=weight, opacity=0.7,
            tooltip=(f"LINK: {row['LINK_ID']}<br>"
                     f"선택: {'O' if row['selected'] else 'X'}<br>"
                     f"우선순위: {row['priority_score']:.3f}<br>"
                     f"비용: {row['deicing_cost']/1000:.0f}천원")
        ).add_to(m2)

    if vehicle_route_coords:
        for route_roads_z, meta in zip(vehicle_route_roads, vehicle_meta):
            if len(route_roads_z) == 0:
                continue
            movement_coords = meta.get('movement_coords') or []
            if len(movement_coords) >= 2:
                movement_latlon = [(c[1], c[0]) for c in movement_coords]
                folium.PolyLine(
                    movement_latlon,
                    color=meta['color'],
                    weight=2,
                    dash_array='8',
                    opacity=0.45,
                    tooltip=f"{meta['name']} 도로망 이동 경로",
                ).add_to(m2)
            start_point = [float(meta.get('start_lat', route_roads_z.iloc[0]['cent_lat'])),
                           float(meta.get('start_lon', route_roads_z.iloc[0]['cent_lon']))]
            end_point = last_route_point(route_roads_z)
            folium.Marker(start_point, icon=folium.Icon(color='green'),
                          popup=f"{meta['name']} 출발").add_to(m2)
            folium.Marker(end_point, icon=folium.Icon(color='red'),
                          popup=f"{meta['name']} 종점").add_to(m2)
    elif route_coords:
        route_latlon = [(c[1], c[0]) for c in route_coords]
        folium.PolyLine(route_latlon, color='#1565c0', weight=2,
                       dash_array='10', opacity=0.5).add_to(m2)
        folium.Marker(route_latlon[0], icon=folium.Icon(color='green'),
                     popup='출발').add_to(m2)
        folium.Marker(route_latlon[-1], icon=folium.Icon(color='red'),
                     popup='종점').add_to(m2)

    legend2 = '''
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:10px; border-radius:5px; border:1px solid #ccc;
         font-size:12px;">
    <b>제설 경로</b><br>
    <span style="color:#e53935">━━</span> AI 선택 도로<br>
    <span style="color:#bdbdbd">━━</span> 미선택 도로<br>
    <span style="color:#1565c0">╌╌</span> 도로망 이동 경로<br>
    <span style="color:#2e7d32">●</span> 차량별 출발/종점
    </div>
    '''
    m2.get_root().html.add_child(folium.Element(legend2))
    m2.save(os.path.join(MAPS_DIR, 'map_optimal_route.html'))
    print("  → map_optimal_route.html 저장")

    # --- 3) 우선순위 히트맵 ---
    m3 = folium.Map(location=center, zoom_start=14, tiles='cartodbpositron')
    priority_roads = folium.FeatureGroup(name='우선순위 도로', show=True)
    for _, row in roads_gdf.iterrows():
        if row['priority_score'] <= 0:
            continue
        folium.PolyLine(
            locations=road_latlon(row),
            color=risk_color(row['priority_score']),
            weight=3 if row['priority_score'] >= 0.5 else 2,
            opacity=0.62,
            tooltip=(f"LINK: {row['LINK_ID']}<br>"
                     f"우선순위: {row['priority_score']:.3f}<br>"
                     f"결빙위험: {row['risk']:.3f}<br>"
                     f"그림자지수: {row['shadow_index']:.3f}")
        ).add_to(priority_roads)
    priority_roads.add_to(m3)
    heat_data = roads_gdf[roads_gdf['priority_score'] > 0][
        ['cent_lat', 'cent_lon', 'priority_score']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=17).add_to(m3)
    folium.LayerControl(collapsed=True).add_to(m3)
    m3.save(os.path.join(MAPS_DIR, 'map_priority_heatmap.html'))
    print("  → map_priority_heatmap.html 저장")

    # --- 4) 네비게이션 경로 맵 (차량별 VRP) ---
    if vehicle_route_roads is not None and vehicle_meta is not None:
        m4 = folium.Map(location=center, zoom_start=14, tiles='cartodbpositron')

        for _, row in roads_gdf.iterrows():
            coords = list(row.geometry.coords)
            folium.PolyLine(
                locations=[(c[1], c[0]) for c in coords],
                color='#eeeeee', weight=1, opacity=0.3
            ).add_to(m4)

        for zi, (route_roads_z, route_coords_z, meta) in enumerate(
                zip(vehicle_route_roads, vehicle_route_coords, vehicle_meta)):
            color = meta['color']

            for _, row in route_roads_z.iterrows():
                folium.PolyLine(
                    locations=road_latlon(row),
                    color=color, weight=4, opacity=0.85,
                    tooltip=(f"{meta['name']} | 위험도: {row['risk']:.2f}<br>"
                             f"차선: {row['LANES']} | 길이: {row['LENGTH']:.0f}m")
                ).add_to(m4)

            movement_coords = meta.get('movement_coords') or []
            if len(movement_coords) >= 2:
                movement_latlon = [(c[1], c[0]) for c in movement_coords]
                folium.PolyLine(
                    movement_latlon,
                    color=color,
                    weight=2,
                    dash_array='8',
                    opacity=0.5,
                    tooltip=f"{meta['name']} 도로망 이동 경로",
                ).add_to(m4)

            start_point = [float(meta.get('start_lat', route_roads_z.iloc[0]['cent_lat'])),
                           float(meta.get('start_lon', route_roads_z.iloc[0]['cent_lon']))]
            folium.Marker(
                start_point,
                icon=folium.DivIcon(html=(
                    f'<div style="background:{color};color:white;border-radius:50%;'
                    f'width:28px;height:28px;text-align:center;line-height:28px;'
                    f'font-weight:bold;font-size:13px;border:2px solid white;'
                    f'box-shadow:0 2px 4px rgba(0,0,0,0.3);">{zi+1}</div>'
                ))
            ).add_to(m4)

            n_wp = min(8, len(route_coords_z))
            step = max(1, len(route_coords_z) // n_wp)
            for i in range(step, len(route_coords_z) - step, step):
                waypoint_row = route_roads_z.iloc[i]
                folium.CircleMarker(
                    location=[float(waypoint_row['cent_lat']), float(waypoint_row['cent_lon'])],
                    radius=4, color=color, fill=True,
                    fill_color='white', fill_opacity=0.9, weight=2,
                    tooltip=f"{meta['name']} 경유지"
                ).add_to(m4)

            mid_idx = len(route_coords_z) // 2
            mid_row = route_roads_z.iloc[mid_idx]
            mid_lat = float(mid_row['cent_lat'])
            mid_lon = float(mid_row['cent_lon'])
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(html=(
                    f'<div style="background:rgba(255,255,255,0.9);padding:3px 8px;'
                    f'border-radius:4px;font-size:11px;font-weight:bold;'
                    f'color:{color};border:1px solid {color};white-space:nowrap;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.2);">'
                    f'{meta["name"]} ({meta["roads"]}개, {meta["distance_km"]}km)</div>'
                ))
            ).add_to(m4)

        vehicle_lines = []
        for idx, meta in enumerate(vehicle_meta):
            vehicle_lines.append(
                f'''<span style="display:inline-block;background:{meta['color']};color:white;
                      border-radius:50%;width:18px;height:18px;text-align:center;
                      line-height:18px;font-size:10px;font-weight:bold;">{idx+1}</span>
                    <b style="color:{meta['color']};">{meta['name']}</b> — 
                    {meta['roads']}개, {meta['distance_km']}km, 적재율 {meta['load_pct']:.0%}<br>'''
            )
        vehicle_html = ''.join(vehicle_lines)

        if results:
            risk_line = f"위험구간 커버 <b>{results['ai']['risk_coverage']*100:.1f}%</b> (기존 {results['baseline']['risk_coverage']*100:.1f}%)"
            cacl2_line = f"염화칼슘 <b>{results['improvement']['cacl2_reduction_pct']:.1f}% 절감</b>"
            pop_line = f"유동인구 보호 <b>{results['ai']['pop_coverage']*100:.1f}%</b> (기존 {results['baseline']['pop_coverage']*100:.1f}%)"
        else:
            risk_line = "위험구간 커버 개선"
            cacl2_line = "염화칼슘 사용량 절감"
            pop_line = "유동인구 보호 개선"

        nav_legend = f'''
        <div style="position:fixed; top:10px; right:10px; z-index:1000;
             background:white; padding:15px; border-radius:10px;
             border:1px solid #ddd; font-size:12px; max-width:280px;
             box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        <div style="font-size:15px;font-weight:bold;margin-bottom:10px;">
        AI 제설 VRP 네비게이션</div>
        <div style="margin-bottom:8px;">
        {vehicle_html}
        </div>
        <hr style="margin:8px 0;">
        <div style="font-size:11px;line-height:1.6;">
        <b>이 경로를 따르면:</b><br>
        • {risk_line}<br>
        • {cacl2_line}<br>
        • {pop_line}<br>
        • Knapsack+BMC 선정 + VRP 경로화
        </div>
        </div>
        '''
        m4.get_root().html.add_child(folium.Element(nav_legend))
        m4.save(os.path.join(MAPS_DIR, 'map_navigation_route.html'))
        print("  → map_navigation_route.html 저장 (네비게이션 지도)")
        create_navigation_simulation_map(
            roads_gdf,
            vehicle_route_roads,
            vehicle_route_coords,
            vehicle_meta,
            results=results,
            maps_dir=MAPS_DIR,
        )


# ============================================================
# 12. 내부 정책 비교 시뮬레이션
# ============================================================
def run_simulation(roads_gdf):
    ensure_output_dirs()
    print("\n" + "="*60)
    print("내부 정책 비교: 도로등급 baseline vs 제안 최적화")
    print("="*60)

    total_cost = roads_gdf['deicing_cost'].sum()
    budget = total_cost * 0.4

    # 기존 방식: 도로등급 순으로 예산 소진
    baseline = roads_gdf.sort_values('ROAD_RANK').copy()
    baseline_selected = []
    baseline_budget = budget
    for _, row in baseline.iterrows():
        if row['deicing_cost'] <= baseline_budget:
            baseline_selected.append(row['LINK_ID'])
            baseline_budget -= row['deicing_cost']
    baseline_mask = roads_gdf['LINK_ID'].isin(baseline_selected)
    ai_mask = roads_gdf['selected'] == 1

    # 지표 1: 위험 구간 커버율 (상위 30%)
    risk_threshold = roads_gdf['risk'].quantile(0.7)
    high_risk = roads_gdf['risk'] >= risk_threshold
    hr_count = max(high_risk.sum(), 1)
    bl_risk_cov = (baseline_mask & high_risk).sum() / hr_count
    ai_risk_cov = (ai_mask & high_risk).sum() / hr_count

    # 가중 위험 커버 (위험도 합)
    total_risk_sum = roads_gdf['risk'].sum()
    bl_risk_sum = roads_gdf.loc[baseline_mask, 'risk'].sum()
    ai_risk_sum = roads_gdf.loc[ai_mask, 'risk'].sum()

    # 지표 2: 염화칼슘 사용량
    if 'deicing_kg' in roads_gdf.columns:
        bl_cacl2 = roads_gdf.loc[baseline_mask, 'deicing_kg'].sum() / 1000
        ai_cacl2 = roads_gdf.loc[ai_mask, 'deicing_kg'].sum() / 1000
    else:
        cost_config = load_deicing_cost_config()
        unit_spread = cost_config['unit_spread_kg_per_m2']
        bl_cacl2 = roads_gdf.loc[baseline_mask, 'area'].sum() * unit_spread / 1000
        ai_cacl2 = roads_gdf.loc[ai_mask, 'area'].sum() * unit_spread / 1000

    # 지표 3: 유동인구 커버율
    total_pop = roads_gdf['pop_weight'].sum()
    bl_pop = roads_gdf.loc[baseline_mask, 'pop_weight'].sum() / max(total_pop, 1)
    ai_pop = roads_gdf.loc[ai_mask, 'pop_weight'].sum() / max(total_pop, 1)

    # 효율성 (위험도 감소 / 사용 비용)
    bl_spent = budget - baseline_budget
    ai_spent = roads_gdf.loc[ai_mask, 'deicing_cost'].sum()
    bl_eff = bl_risk_sum / max(bl_spent, 1) * 1e6
    ai_eff = ai_risk_sum / max(ai_spent, 1) * 1e6

    print(f"\n{'지표':<30} {'도로등급 기준':<15} {'제안 최적화':<15} {'차이':<10}")
    print("-" * 70)
    print(f"{'선택 도로 수':<29} {baseline_mask.sum():<15} {ai_mask.sum():<15}")
    print(f"{'사용 예산(백만원)':<27} {bl_spent/1e6:<15.1f} {ai_spent/1e6:<15.1f}")
    print(f"{'위험구간 커버율':<28} {bl_risk_cov:<15.1%} {ai_risk_cov:<15.1%} {ai_risk_cov-bl_risk_cov:+.1%}")
    print(f"{'위험도 가중 커버':<27} {bl_risk_sum/total_risk_sum:<15.1%} {ai_risk_sum/total_risk_sum:<15.1%} {(ai_risk_sum-bl_risk_sum)/total_risk_sum:+.1%}")
    print(f"{'염화칼슘 사용량(톤)':<26} {bl_cacl2:<15.1f} {ai_cacl2:<15.1f} {(1-ai_cacl2/max(bl_cacl2,1)):+.1%}")
    print(f"{'유동인구 커버율':<28} {bl_pop:<15.1%} {ai_pop:<15.1%} {ai_pop-bl_pop:+.1%}")
    print(f"{'비용 효율성(위험/백만원)':<25} {bl_eff:<15.2f} {ai_eff:<15.2f} {(ai_eff/max(bl_eff,1)-1):+.1%}")

    results = {
        'schema_version': 2,
        'evaluation_type': 'internal_policy_simulation',
        'external_ground_truth_used': False,
        'baseline_definition': 'ROAD_RANK ascending order until the same budget is exhausted',
        'proposed_definition': 'MLP-derived relative risk plus Hybrid BMC/Knapsack selection',
        'warning': (
            '이 결과는 같은 계산 위험도와 비용을 사용한 내부 정책 비교이며 '
            '현장 사고 감소율이나 도로 결빙 정확도가 아니다.'
        ),
        'budget_million_won': round(budget / 1e6, 1),
        'baseline': {
            'roads': int(baseline_mask.sum()),
            'spent_million': round(bl_spent / 1e6, 1),
            'risk_coverage': round(float(bl_risk_cov), 4),
            'weighted_risk_coverage': round(float(bl_risk_sum / total_risk_sum), 4),
            'cacl2_tons': round(float(bl_cacl2), 2),
            'pop_coverage': round(float(bl_pop), 4),
            'efficiency': round(float(bl_eff), 2),
        },
        'ai': {
            'roads': int(ai_mask.sum()),
            'spent_million': round(ai_spent / 1e6, 1),
            'risk_coverage': round(float(ai_risk_cov), 4),
            'weighted_risk_coverage': round(float(ai_risk_sum / total_risk_sum), 4),
            'cacl2_tons': round(float(ai_cacl2), 2),
            'pop_coverage': round(float(ai_pop), 4),
            'efficiency': round(float(ai_eff), 2),
        },
        'improvement': {
            'risk_coverage_pp': round(float(ai_risk_cov - bl_risk_cov) * 100, 1),
            'cacl2_reduction_pct': round(float(1 - ai_cacl2 / max(bl_cacl2, 1)) * 100, 1),
            'pop_coverage_pp': round(float(ai_pop - bl_pop) * 100, 1),
        }
    }

    with open(os.path.join(REPORTS_DIR, 'simulation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n  → simulation_results.json 저장")

    # 시뮬레이션 비교 차트
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ['위험구간\n커버율', '유동인구\n커버율', '비용 효율성\n(정규화)']
    bl_vals = [bl_risk_cov, bl_pop, bl_eff / max(ai_eff, bl_eff)]
    ai_vals = [ai_risk_cov, ai_pop, ai_eff / max(ai_eff, bl_eff)]

    x = np.arange(len(categories))
    w = 0.35
    axes[0].bar(x - w/2, bl_vals, w, label='도로등급 기준', color='#90a4ae')
    axes[0].bar(x + w/2, ai_vals, w, label='제안 최적화', color='#e53935')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].set_title('성능 비교')

    axes[1].bar(['도로등급', '제안'], [bl_cacl2, ai_cacl2], color=['#90a4ae', '#43a047'])
    axes[1].set_title('염화칼슘 사용량 (톤)')
    axes[1].set_ylabel('톤')

    risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    risk_labels = ['<0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '≥0.8']
    roads_gdf['risk_bin'] = pd.cut(roads_gdf['risk'], bins=risk_bins, labels=risk_labels)
    risk_dist = roads_gdf['risk_bin'].value_counts().sort_index()
    axes[2].bar(risk_dist.index, risk_dist.values, color='#1565c0')
    axes[2].set_title('도로별 위험도 분포')
    axes[2].set_ylabel('도로 수')
    axes[2].set_xlabel('위험도')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'simulation_comparison.png'), dpi=150)
    plt.close()
    print("  → simulation_comparison.png 저장")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AI 제설 맵핑 시스템 — 파이프라인 시작")
    print("대상: 서울특별시 강남구")
    print("=" * 60)

    roads = load_roads()
    roads = load_buildings_and_shadow(roads)
    weather = load_weather()
    roads = load_population_and_map(roads)

    model_path = os.path.join(MODELS_DIR, 'road_surface_temperature_mlp.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "실측 노면온도 MLP가 없습니다. "
            "python scripts\\train_mlp_pipeline.py를 먼저 실행하세요."
        )
    from src.road_surface_temperature_mlp import apply_mlp_road_risk, load_model_bundle
    from src.mlp_training_pipeline import load_road_weather

    bundle = load_model_bundle(Path(model_path))
    road_weather = load_road_weather(Path(BASE_DIR))
    roads = apply_mlp_road_risk(roads, weather, road_weather, bundle)
    roads = calculate_priority(roads)
    roads, selected = hybrid_bmc_knapsack_optimize(roads, budget_ratio=0.4)
    vehicle_route_roads, vehicle_route_coords, vehicle_meta = vrp_route(roads, selected)

    route_coords_flat = []
    for vehicle_coords in vehicle_route_coords:
        route_coords_flat.extend(vehicle_coords)

    results = run_simulation(roads)
    create_maps(roads, route_coords_flat, vehicle_route_roads, vehicle_route_coords, vehicle_meta, results)
    nav_report = generate_navigation_report(roads, vehicle_route_roads, vehicle_route_coords, vehicle_meta, results)

    export = roads.drop(columns=['risk_bin'], errors='ignore').copy()
    export.to_file(os.path.join(OUTPUT_DATA_DIR, 'gangnam_roads_result.geojson'), driver='GeoJSON')
    print("\n→ gangnam_roads_result.geojson 저장")

    print("\n" + "=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)

    return roads, results


if __name__ == '__main__':
    roads, results = main()
