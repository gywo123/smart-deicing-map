# ICE-ZERO

강남구 겨울철 도로의 상대적 결빙 위험 순위를 산정하고, 제한된 예산과 제설차 작업시간 안에서 처리 도로와 이동 경로를 생성하는 프로젝트다.

## 현재 모델

- 모델: PyTorch Lightning residual MLP
- 모델 버전: `road_surface_temperature_residual_mlp_v2`
- 예측 목표: 발행 시각 기준 3시간 후 실측 노면온도
- 실제 정답: 기상청 도로기상관측 원지(도) `노면온도(°C)`
- pseudo-label: 사용하지 않음
- 개발/튜닝/테스트: 2024-11~2025-02 / 2025-11 / 2025-12 시간 분리
- 도로 위험도: MLP 예측 노면온도 + 강수·적설·응결 수분 + 건물 그림자
- XGBoost, MCMF: 사용하지 않음

MLP는 도로 사고확률을 직접 예측하지 않는다. 강남 중심에서 약 5.25km 떨어진 경부선 원지(도) 관측소의 실제 노면온도를 예측한 뒤 물리 조건을 결합해 도로 간 상대 제설 우선순위를 만든다. 기존 ASOS 지면온도 모델은 비교용 소스 코드와 평가 기록만 남기며 구형 모델 바이너리는 배포하지 않는다.

## 독립 평가 결과

2025년 12월은 모델 학습, 조기종료 선택, 동결확률 보정에 사용하지 않은 시간 홀드아웃이다.

| 평가 항목 | 결과 |
|---|---:|
| 테스트 표본 | 724시간 |
| 3시간 후 노면온도 RMSE | 1.064°C |
| RMSE 95% bootstrap 구간 | 0.999~1.129°C |
| MAE | 0.804°C |
| R² | 0.946 |
| 현재 노면온도 지속 기준 대비 RMSE skill | 61.2% |
| 0°C 이하 Precision / Recall / F1 | 0.958 / 0.806 / 0.875 |
| 0°C 이하 사건 PR-AUC | 0.976 |
| ROC-AUC | 0.991 |
| Brier score | 0.0433 |
| 현재 노면온도 지속 기준 RMSE | 2.744°C |
| 발행 시각 기온 기준 RMSE | 3.161°C |

이 수치는 **원지(도) 단일 관측소의 실측 노면온도와 0°C 이하 사건 예측 성능**이다. 강남 전체 도로별 결빙 정확도나 사고 발생 확률로 해석하면 안 된다.

## 실제 사고 자료를 이용한 공간 확인

`data/raw/weather/13_24_freezing.csv`의 강남구 결빙사고 다발지역은 모델 학습과 계수 조정에 사용하지 않고 독립 공간 proxy로만 사용한다.

- 원자료 13건을 동일 장소 기준 6개 군집으로 통합
- 사고지역과 겹치는 모든 도로의 평균 위험 백분위 사용
- 지역 안에서 최고 위험 도로만 고르는 선택 편향 제거
- 무작위 비교도 사고지역별 도로 개수를 동일하게 유지
- 사고건수 가중 평균 위험 백분위 0.717, 무작위 평균 0.501, p=0.0001
- 제안 경로의 사고지역 내부 선택률 70.4%, 전체 선택률 73.6%, p=0.678

사고 다발지역은 개별 시각의 도로 결빙 정답이 아니다. 이 결과는 위험 순위의 공간적 일치도이며 도로 결빙 분류 정확도가 아니다.

위험 순위의 공간 일치도는 확인됐지만, 현재 제안 경로가 사고지역 도로를 무작위 동일 규모 선택보다 더 많이 포함한다는 근거는 없다. `simulation_results.json`의 개선율은 외부 정답이 아닌 동일 계산식 안의 내부 정책 비교다.

## 공식 상습결빙구간 교차검증

`collect_public_data.py`는 행정안전부 상습결빙구간 3,358건을 공식 페이지에서 수집한다. 서울 268건 중 강남구 7개 구간을 추출해 표준 링크 28개와 연결했다. 이 자료도 모델 학습과 계수 조정에는 사용하지 않았다.

| 항목 | 결과 |
|---|---:|
| 강남 공식 상습결빙구간 | 7개 |
| 매칭 표준 링크 | 28개 |
| 평균 위험 백분위 | 0.400 |
| 무작위 평균 | 0.501 |
| permutation p-value | 0.896 |

현재 모델은 공식 상습결빙구간에 일반화되지 않았다. 인접 관측소의 실제 노면온도는 추가됐지만 도로별 경사, 교량·터널, 배수, 포장재와 공간별 노면센서가 부족하다는 근거다. 이 결과를 숨기거나 해당 7개 구간으로 계수를 조정하지 않는다.

## 설치

권장 방식은 Conda 환경이다.

```powershell
conda env create -f environment.yml
conda activate smart-deicing
```

기존 `mh_ai311` 환경을 사용할 수도 있다.

```powershell
conda activate mh_ai311
python -m pip install -r requirements.txt
```

GPU 확인:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 전체 실행

```powershell
python scripts\preprocess_data.py
python scripts\collect_public_data.py
python scripts\train_mlp_pipeline.py
python scripts\validation_simulation.py
python -m pytest tests --basetemp .pytest_run
```

역할:

1. `preprocess_data.py`: 도로·건물·기상·생활인구 정제와 MLP 시간 피처 생성
2. `collect_public_data.py`: 행정안전부 상습결빙구간과 강남 인접 기상청 노면관측 수집·정제
3. `train_mlp_pipeline.py`: 실제 노면온도 MLP 학습, 2025-12 홀드아웃 평가, 도로 위험도·경로·지도 생성
4. `validation_simulation.py`: 사고 다발지역과 공식 상습결빙구간을 이용한 독립 공간 proxy 검증

`main_pipeline.py`는 저장된 `models/road_surface_temperature_mlp.pt`로 지도와 경로를 다시 생성한다. 모델이 없으면 먼저 `train_mlp_pipeline.py`를 실행해야 한다.

## 주요 입력

| 데이터 | 파일 | 역할 |
|---|---|---|
| 도로망 | `MOCT_LINK.shp`, `MOCT_NODE.shp` | 도로 특성·방향 그래프·경로 |
| 건물 | `AL_D162_11_20260115.shp` | 시간대별 건물 그림자 |
| ASOS 기상 | `OBS_ASOS_TIM_*.csv` | 대기·지면온도·수분·일사 입력 피처 |
| 도로기상 | `kma_road_weather_hourly.csv` | 현재 노면온도 입력과 3시간 후 실측 정답 |
| 생활인구 | `250_LOCAL_RESD_*.csv` | 제설 우선순위의 노출도 |
| 결빙사고 다발지역 | `13_24_freezing.csv` | 독립 공간 proxy 검증 |
| 상습결빙구간 | `habitual_icing_segments_20251107.csv` | 두 번째 독립 공간 proxy 검증 |

기상청 도로기상관측은 API Hub 인증키가 있을 때 자동 수집한다. 기본 범위는 2024-11-01부터 2026-02-01 전까지의 겨울철이며, 강남에서 가장 가까운 `목표(결빙)` 관측소를 자동 선택한다. 2026년 2월 원지 관측은 일부 날짜에서 API 시간 초과가 확인되어 기본 범위에서 제외했다. 분 단위 원자료는 `data/raw/weather/road_observations`에 gzip으로 캐시하고, 날짜별 이어받기 캐시도 함께 유지한다. 모델 결합용 시간 단위 자료는 `outputs/data/cleaned/kma_road_weather_hourly.csv`에 저장한다. 강남 도로 자체가 아닌 인접 고속도로 관측이므로 전이학습·외부 보조자료로 사용해야 한다.

CMD에서 실행:

```bat
set KMA_API_KEY=발급받은_인증키
python scripts\collect_public_data.py
```

PowerShell에서 실행:

```powershell
$env:KMA_API_KEY = "발급받은_인증키"
python scripts\collect_public_data.py
```

인증키는 코드·설정 파일·보고서에 저장하지 않는다. 기존 캐시를 무시하고 다시 받을 때만 `--refresh`를 사용한다.

공식 출처:

- 행정안전부 상습결빙구간: https://www.data.go.kr/data/15067396/fileData.do
- 기상청 도로기상관측자료: https://www.data.go.kr/data/15159045/openapi.do
- 기상청 AWS 자료: https://data.kma.go.kr/data/grnd/selectAwsRltmList.do

제설 비용과 살포량은 `config/deicing_costs.json`에서 관리한다. 비용은 MLP 입력에 사용하지 않고 도로 선택 최적화 단계에서만 사용한다.

## 모델 입력

MLP는 다음 26개 피처를 사용한다. 모든 노면 이력 피처는 예측 시각 이전 관측만 사용한다.

```text
temp, ground_temp_now, ground_temp_lag_1h, temp_6h_mean,
humidity, dewpoint_depression, wind, precip_6h, snow, new_snow_6h,
solar, solar_3h_sum, sunshine,
hour_sin, hour_cos, day_sin, day_cos
road_surface_temp_now, road_surface_temp_lag_1h,
road_surface_temp_lag_2h, road_surface_temp_lag_3h,
road_surface_temp_lag_6h, road_surface_temp_mean_3h,
road_surface_temp_mean_6h, road_surface_temp_change_1h,
road_surface_temp_change_3h
```

`deicing_cost`, `area`, `priority_score`, 기존 계산식 `risk`는 모델 입력에서 제외했다.

## 모델 산출물

```text
models/road_surface_temperature_mlp.pt
models/road_surface_model_manifest.json
outputs/reports/road_surface_model_metrics.json
outputs/reports/road_surface_model_card.json
outputs/reports/mlp_training_summary.json
outputs/figures/road_surface_mlp_evaluation.png
```

모델 바이너리 `.pt`는 약 245KB이므로 Git 제외 대상이 아니다. `models/road_surface_model_manifest.json`의 SHA-256, 모델 버전, 실제 정답, 홀드아웃 성능과 함께 보관한다. 실행 시 모델 해시와 26개 입력 피처 순서가 manifest·현재 코드와 일치하는지 확인한다.

## v1 대비 개선

- 시간 평균을 해당 시간의 끝 시각에 붙여 최대 59분의 미래 정보가 들어갈 가능성을 제거했다.
- 과거 2·3·6시간 노면온도, 3·6시간 이동평균, 1·3시간 변화율을 추가했다.
- 동일한 미학습 2025년 12월에서 RMSE `1.212 → 1.064°C`(-12.2%), MAE `0.949 → 0.804°C`(-15.2%), R² `0.929 → 0.946`으로 개선됐다.
- 관측 -3~+3°C 구간 RMSE도 `1.015 → 0.846°C`(-16.7%)로 낮아졌다.
- 자세한 비교는 `outputs/reports/model_improvement_comparison.md`에 기록한다.

## 지도와 리포트

```text
outputs/maps/map_freezing_risk.html
outputs/maps/map_priority_heatmap.html
outputs/maps/map_optimal_route.html
outputs/maps/map_navigation_route.html
outputs/maps/map_navigation_simulation.html
outputs/maps/map_validation_simulation.html
outputs/reports/validation_simulation.json
outputs/reports/validation_hotspot_matches.csv
```

## 해석 가능한 주장

- 실제 기상청 노면온도를 학습한 MLP를 미학습 2025년 12월로 평가했다.
- 미학습 월의 예측 노면온도와 수분·그림자를 결합해 도로의 상대 제설 우선순위를 계산했다.
- 실제 결빙사고 다발지역을 학습에 사용하지 않고 공간 proxy로 비교했다.

현재 주장하면 안 되는 내용:

- 도로별 결빙 정확도 97.5%
- 사고 발생확률 정확도 97.5%
- 제설로 인한 실제 사고 감소율
- `risk`를 절대 결빙 발생확률로 표현

도로×시각 실측 라벨을 확보하면 노면센서·시각이 있는 CCTV 판독·현장 점검 자료를 `LINK_ID`, `timestamp`, `icing` 형태로 연결해 공간 홀드아웃 평가를 추가해야 한다.

현재 정답은 강남 중심에서 약 5.25km 떨어진 원지(도) 단일 노면 관측소 자료다. 실제 노면온도를 학습했지만 강남구 도로별 공간 차이를 직접 학습한 모델은 아니며, 건물 그림자와 생활인구는 MLP 학습 이후의 도로 우선순위 단계에서 반영한다.
