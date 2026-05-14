# 스마트 제설 지도 실행 방법 및 프로젝트 정리

## 1. 프로젝트 개요

이 프로젝트는 강남구 도로 데이터를 기준으로 결빙 위험도를 계산하고, 우선 제설 대상 도로를 선택한 뒤, 제설차별 도포 경로를 지도 HTML로 생성하는 파이프라인이다.

현재 모델/알고리즘 흐름은 다음과 같다.

```text
원본 데이터 정제
-> 건물 그림자/기상/유동인구/도로 특성 계산
-> MLP 기반 결빙 위험도 및 사고 위험도 추정
-> Knapsack + Budgeted Maximum Coverage로 제설 대상 선택
-> VRP 휴리스틱으로 제설차별 경로 생성
-> 지도/리포트 출력
```

현재 기준:

- 메인 모델: PyTorch Lightning 기반 MLP
- XGBoost 사용 안 함
- MCMF 사용 안 함
- 도로 연결: `F_NODE -> T_NODE` 기준 방향 그래프
- 노드 보정: `MOCT_NODE` 좌표 기준으로 링크 시작/종료점 보정
- 그림자 날짜: `1월 15일` 고정
- 그림자 시간대: `0시~23시` 전체 시간대 기준
- 제설차 수: 작업량에 따라 자동 산정

## 2. 폴더 구조

```text
smart-deicing-map-codex-71uq2a/
├─ data/                 원본 데이터
├─ models/               학습된 모델 파일
├─ outputs/              지도, 리포트, 정제 데이터 출력
├─ scripts/              실행 스크립트
├─ src/                  모델/점수/계획 모듈
├─ tests/                테스트 코드
└─ 모델 기획서.md
```

GitHub에는 필요에 따라 코드와 일부 `outputs`만 올린다. 일반적으로 `data/`, `models/`, `.venv/`는 올리지 않는다.

## 3. 실행 환경

권장 환경은 Conda의 `mh_ai311` 환경이다.

PowerShell에서 실행:

```powershell
conda activate mh_ai311
```

또는 Conda activate가 안 잡힐 때는 파이썬을 직접 지정한다.

```powershell
C:\Users\USER\miniconda3\envs\mh_ai311\python.exe --version
```

## 3-1. 새 컴퓨터/새 환경에서 처음 설치

Python 라이브러리를 한 번에 설치하는 파일은 `requirements.txt`다.

가장 간단한 설치:

```powershell
python -m pip install -r requirements.txt
```

다만 Windows에서는 `geopandas`, `pyogrio`, `pyproj`, `shapely` 같은 공간정보 라이브러리가 pip에서 꼬일 수 있다. 그래서 새 환경에서는 Conda용 `environment.yml` 사용을 더 권장한다.

Conda 새 환경 생성:

```powershell
conda env create -f environment.yml
conda activate smart-deicing
```

이미 같은 이름의 환경이 있으면 갱신:

```powershell
conda env update -f environment.yml --prune
conda activate smart-deicing
```

설치 확인:

```powershell
python -c "import geopandas, torch, pytorch_lightning, folium; print('OK')"
python -m pytest tests
```

GPU 확인:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

새 환경 기준 전체 실행:

```powershell
python scripts\preprocess_data.py
python scripts\main_pipeline.py
python scripts\train_mlp_pipeline.py
python scripts\validation_simulation.py
python -m pytest tests
```

## 4. 데이터 전처리

원본 데이터를 정제해서 `outputs/data/cleaned` 아래에 저장한다.

```powershell
python scripts\preprocess_data.py
```

Conda 파이썬 직접 실행:

```powershell
C:\Users\USER\miniconda3\envs\mh_ai311\python.exe scripts\preprocess_data.py
```

전처리 산출물:

```text
outputs/data/cleaned/gangnam_roads_clean.geojson
outputs/data/cleaned/gangnam_buildings_clean.geojson
outputs/data/cleaned/weather_winter_clean.csv
outputs/data/cleaned/population_grid_clean.csv
outputs/reports/data_quality_report.json
outputs/reports/data_quality_report.md
```

전처리에서 하는 일:

- `MOCT_LINK.shp` 도로 링크 로딩
- `MOCT_NODE.shp` 기준으로 링크 시작/종료점 보정
- `ROAD_USE` 통행불가 도로 제거
- 강남구 범위 도로 필터링
- 건물 높이 결측값 보정
- 겨울철 기상 데이터 정리
- 생활인구 250m 격자 데이터 정리

## 5. 전체 파이프라인 실행

전처리된 데이터를 사용해서 위험도 계산, 도로 선택, VRP 경로 생성, 지도 출력을 한 번에 수행한다.

```powershell
python scripts\main_pipeline.py
```

Conda 파이썬 직접 실행:

```powershell
C:\Users\USER\miniconda3\envs\mh_ai311\python.exe scripts\main_pipeline.py
```

주요 출력:

```text
outputs/data/gangnam_roads_result.geojson
outputs/maps/map_freezing_risk.html
outputs/maps/map_optimal_route.html
outputs/maps/map_priority_heatmap.html
outputs/maps/map_navigation_route.html
outputs/maps/map_navigation_simulation.html
outputs/maps/map_validation_simulation.html
outputs/reports/simulation_results.json
outputs/reports/route_navigation.json
outputs/reports/validation_simulation.json
outputs/figures/simulation_comparison.png
outputs/figures/validation_simulation.png
```

## 6. MLP 학습 및 모델 내보내기

MLP 결빙 위험 모델과 사고 위험 모델을 학습하고, 학습된 모델 파일과 지도 산출물을 다시 생성한다.

```powershell
python scripts\train_mlp_pipeline.py
```

Conda 파이썬 직접 실행:

```powershell
C:\Users\USER\miniconda3\envs\mh_ai311\python.exe scripts\train_mlp_pipeline.py
```

GPU가 잡히면 로그에 다음처럼 표시된다.

```text
학습 장치: GPU - NVIDIA GeForce RTX 4060 Laptop GPU
GPU available: True (cuda), used: True
```

모델 출력:

```text
models/icing_risk_mlp.pt
models/accident_risk_mlp.pt
outputs/reports/mlp_training_summary.json
outputs/figures/mlp_model_evaluation.png
```

## 7. 지도 확인

생성된 HTML 파일을 브라우저에서 열면 된다.

```text
outputs/maps/map_freezing_risk.html
outputs/maps/map_optimal_route.html
outputs/maps/map_priority_heatmap.html
outputs/maps/map_navigation_route.html
outputs/maps/map_navigation_simulation.html
```

이미 브라우저에서 열려 있다면 `Ctrl + F5`로 강제 새로고침한다.

지도별 의미:

- `map_freezing_risk.html`: 도로별 결빙 위험도
- `map_priority_heatmap.html`: 우선순위 히트맵
- `map_optimal_route.html`: 제설 대상 도로와 차량별 작업 구역
- `map_navigation_route.html`: 제설차별 경로와 상세 카드
- `map_navigation_simulation.html`: 재생형 제설차 이동 시뮬레이션
- `map_validation_simulation.html`: 예측 위험도와 시점별 관측 상황 비교 지도

## 7-1. 예측 위험도 검증 시뮬레이션

현재 보유한 사고 자료는 도로 좌표별 실제 사고 라벨이 아니라, 강남구 날씨별 사고 통계와 노면상태/시간대별 사고 통계다. 그래서 실제 지점별 사고 라벨을 확보하기 전까지는 통계 기반 검증 시뮬레이션으로 예측 위험도와 관측 상황을 비교한다.

검증 흐름:

```text
예측 위험도 + 사고확률
+ 해당 시점 기상 상황
+ 강남구 날씨별 사고 통계
+ 노면상태/시간대별 사고 통계
-> 시점별/도로별 관측 이벤트 생성
-> 예측 위험도와 관측 이벤트 비교
```

실행:

```powershell
python scripts\validation_simulation.py
```

Conda 파이썬 직접 실행:

```powershell
C:\Users\USER\miniconda3\envs\mh_ai311\python.exe scripts\validation_simulation.py
```

출력:

```text
outputs/reports/validation_simulation.json
outputs/reports/validation_events_sample.csv
outputs/reports/validation_road_summary.csv
outputs/figures/validation_simulation.png
outputs/maps/map_validation_simulation.html
```

주의:

- 이 검증은 실제 좌표별 사고 라벨이 없는 상태의 통계 기반 시뮬레이션이다.
- 실제 사고 좌표/시간 또는 도로 결빙 관측 데이터가 들어오면 `actual_event`를 그 데이터로 교체하면 된다.

## 8. 테스트

전체 테스트:

```powershell
python -m pytest tests
```

Conda 파이썬 직접 실행:

```powershell
C:\Users\USER\miniconda3\envs\mh_ai311\python.exe -m pytest tests
```

캐시 권한 경고가 뜨면 테스트 자체가 실패한 것은 아닐 수 있다. 마지막 줄의 `passed` 여부를 확인한다.

## 9. 실행 순서 추천

데이터를 새로 넣었거나 도로/건물/기상/생활인구가 바뀐 경우:

```powershell
python scripts\preprocess_data.py
python scripts\main_pipeline.py
python scripts\train_mlp_pipeline.py
python scripts\validation_simulation.py
python -m pytest tests
```

코드만 조금 바꿨고 기존 정제 데이터를 그대로 쓸 경우:

```powershell
python scripts\main_pipeline.py
python scripts\train_mlp_pipeline.py
python scripts\validation_simulation.py
python -m pytest tests
```

지도만 다시 보고 싶을 경우:

```powershell
python scripts\main_pipeline.py
```

## 10. 주의사항

- `outputs`는 결과 파일이라 용량이 커질 수 있다.
- `data` 원본은 GitHub에 올리지 않는 것을 권장한다.
- `models` 학습 모델도 보통 GitHub에는 올리지 않는다.
- `MOCT_LINK.shp`, `MOCT_NODE.shp`가 Git LFS pointer 파일이면 실행이 실패한다.
- 네비게이션 경로는 실제 도로망의 `F_NODE`, `T_NODE` 연결을 기준으로 만든다.
- 도로가 실제로 단절되어 있거나 원본 노드링크가 끊긴 경우 일부 이동 경로가 길어질 수 있다.

## 11. 현재 산출물 기준 요약

최근 실행 기준 주요 결과:

- 정제 강남 도로: `2,072개`
- 선택 도로: `1,187개`
- 자동 제설차 수: `5대`
- 총 경로: 약 `553.4km`
- 위험구간 커버: `12.7% -> 97.3%`
- 염화칼슘 사용량: 약 `22.8%` 절감
- 테스트: `29 passed`
