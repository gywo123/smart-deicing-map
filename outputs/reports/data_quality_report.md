# Data Quality Report

- [info] roads: 원본 도로 548,668개 중 강남 중심점 기준 2,072개 사용
- [info] roads: MOCT_NODE 기준 링크 시작/종료점 보정 (3845)
- [warning] roads: ROAD_USE 통행불가 링크 제거 (41)
- [warning] roads: bbox와 걸쳤지만 중심점이 강남 범위 밖인 링크 제거 (76)
- [info] buildings: 강남구 건물 12,908개 정제
- [warning] buildings: 건물 높이 결측/0 값을 층수 기반 height_m으로 보정 (2711)
- [warning] buildings: invalid geometry 복구 (1)
- [warning] weather: 기온 컬럼 없는 OBS 보조 파일 제외: OBS_ASOS_TIM_20260510122332.csv
- [info] weather: 풍속(m/s) 결측을 최대 6시간 과거값으로 보완 (8)
- [warning] weather: 인과적 보완 후에도 핵심 기상 피처가 비어 학습 시 제외되는 행 (33)
- [info] weather: 사용 ASOS 관측소: 108 서울
- [info] weather: 겨울철 기상 17,184행 정제
- [info] population: 생활인구 원본 15,839,601행 중 강남 1,086,137행 집계
- [info] population: 강남 250m 격자 472개 정제

## Cleaned Outputs
- `outputs\data\cleaned\gangnam_buildings_clean.geojson`
- `outputs\data\cleaned\gangnam_roads_clean.geojson`
- `outputs\data\cleaned\population_grid_clean.csv`
- `outputs\data\cleaned\weather_winter_clean.csv`