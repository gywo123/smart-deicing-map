# Model Artifacts

현재 운영 모델은 `road_surface_temperature_mlp.pt`이며 PyTorch Lightning residual MLP다.

- 예측 목표: 3시간 후 기상청 원지(도) 실측 노면온도
- pseudo-label: 미사용
- 테스트: 미학습 2025년 12월 시간 홀드아웃
- 상세 지표: `../outputs/reports/road_surface_model_metrics.json`
- 재현 메타데이터와 SHA-256: `road_surface_model_manifest.json`

`road_surface_temperature_mlp.pt`는 약 245KB이므로 Git에서 제외하지 않는다. 로드할 때 SHA-256, 모델 버전, 26개 입력 피처 순서를 `road_surface_model_manifest.json` 및 현재 코드와 대조한다.

이전 ASOS 지면온도 모델은 비교용 소스 코드와 평가 기록만 남긴다. 현재 지도·경로 파이프라인에서 사용하지 않는 구형 모델 바이너리와 manifest는 배포하지 않는다.

구형 `icing_risk_mlp.pt`와 `accident_risk_mlp.pt`는 제거했으며 현재 파이프라인에서 사용하지 않는다.
