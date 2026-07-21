# ICE-ZERO

강남구 겨울철 도로의 3시간 후 노면온도를 예측하고, 결빙 위험이 높은 도로의 제설 우선순위와 차량 경로를 생성하는 프로젝트다.

## 처리 흐름

```text
기상·노면·도로·건물 데이터
        ↓
Residual MLP로 3시간 후 노면온도 예측
        ↓
수분·적설·건물 그림자를 결합한 상대 위험도 산정
        ↓
BMC/Knapsack으로 제설 대상 선정
        ↓
도로 그래프 기반 VRP로 차량 경로 생성
```

- **MLP**: 노면온도 예측
- **BMC/Knapsack**: 예산 안에서 처리할 도로 결정
- **VRP**: 차량별 작업 순서와 도로 이동 경로 생성
- XGBoost와 MCMF는 사용하지 않는다.

## AI 모델

- 모델: PyTorch Lightning Residual MLP
- 버전: `road_surface_temperature_residual_mlp_v2`
- 입력: 기온, 습도, 강수, 적설, 일사, 시간, 과거 노면온도 등 26개 피처
- 출력: 3시간 후 노면온도(°C)
- 정답: 기상청 원지(도) 도로기상관측 실측 노면온도
- pseudo-label 및 사고확률 계산식: 사용하지 않음

## 검증 결과

학습과 보정에 사용하지 않은 2025년 12월 724시간을 평가했다.

| 지표 | 결과 |
|---|---:|
| RMSE | 1.064°C |
| MAE | 0.804°C |
| R² | 0.946 |
| 지속 예측 대비 RMSE 개선 | 61.2% |
| 0°C 이하 Precision / Recall / F1 | 0.958 / 0.806 / 0.875 |
| PR-AUC | 0.976 |

이 수치는 강남 인접 **원지(도) 단일 관측소**의 노면온도 예측 성능이다. 강남구 전체 도로의 결빙 정확도나 사고 발생확률로 해석하지 않는다.

## 설치

```powershell
conda env create -f environment.yml
conda activate smart-deicing
```

기존 Conda 환경을 사용한다면 다음과 같이 설치한다.

```powershell
python -m pip install -r requirements.txt
```

## 실행

저장된 모델로 지도와 경로만 다시 생성:

```powershell
python scripts\main_pipeline.py
```

데이터 정제부터 학습과 검증까지 전체 실행:

```powershell
python scripts\preprocess_data.py
$env:KMA_API_KEY = "발급받은_인증키"
python scripts\collect_public_data.py
python scripts\train_mlp_pipeline.py
python scripts\validation_simulation.py
python -m pytest tests --basetemp .pytest_run
```

API 인증키는 코드나 설정 파일에 저장하지 않는다.

## 주요 파일

| 경로 | 역할 |
|---|---|
| `scripts/main_pipeline.py` | 저장 모델로 위험도·경로·지도 생성 |
| `scripts/train_mlp_pipeline.py` | MLP 학습과 홀드아웃 평가 |
| `scripts/validation_simulation.py` | 사고 다발지역·상습결빙구간 공간 검증 |
| `src/road_surface_temperature_mlp.py` | Residual MLP 모델 코드 |
| `models/road_surface_temperature_mlp.pt` | 현재 학습 모델 |
| `outputs/maps/` | 결과 지도 |
| `outputs/reports/` | 성능 및 검증 보고서 |

## 주요 결과

- `outputs/maps/map_freezing_risk.html`: 도로 결빙 상대 위험도
- `outputs/maps/map_optimal_route.html`: 차량별 제설 작업 경로
- `outputs/maps/map_navigation_simulation.html`: 차량 이동 시뮬레이션
- `outputs/reports/road_surface_model_metrics.json`: 모델 평가 지표
- `outputs/reports/validation_simulation.json`: 공간 검증 결과

## 한계

- 단일 인접 관측소를 사용하므로 강남구 도로별 공간 차이를 직접 학습하지 못한다.
- 도로별 실제 결빙 상태와 사고 시점 라벨이 없어 위험도는 절대 확률이 아닌 상대 우선순위다.
- 향후 교량, 경사, 배수, 포장재, 도로별 노면센서와 실제 제설 작업 결과를 추가해야 한다.
