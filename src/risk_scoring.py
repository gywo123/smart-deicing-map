"""Smart Deicing Map MVP용 위험도 점수 계산 보조 모듈.

이 모듈은 대회 참고 프로젝트의 흐름을 가볍고 의존성 없는 형태로 옮긴다.
실제 도로 결빙 라벨이나 학습된 tabular 모델이 준비되기 전까지 규칙 기반
pseudo-risk 점수를 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .dispatch_planner import DeicingCandidate


@dataclass(frozen=True)
class WeatherFeatures:
    temp_c: float
    ground_temp_c: float
    humidity_pct: float
    wind_m_s: float
    snow_cm: float = 0.0
    precipitation_mm: float = 0.0


@dataclass(frozen=True)
class RoadRiskFeatures:
    segment_id: str
    length_m: float
    lane_count: int
    shadow_index: float = 0.0
    exposure_score: float = 1.0
    road_width_m: float = 8.0
    speed_limit_kmh: int = 50
    accident_history_count: int = 0
    covered_cells: frozenset[str] = field(default_factory=frozenset)
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must not be empty")
        if self.length_m <= 0:
            raise ValueError("length_m must be positive")
        if self.lane_count <= 0:
            raise ValueError("lane_count must be positive")
        if not 0 <= self.shadow_index <= 1:
            raise ValueError("shadow_index must be in the range [0, 1]")
        if self.exposure_score < 0:
            raise ValueError("exposure_score must be non-negative")
        if self.road_width_m <= 0:
            raise ValueError("road_width_m must be positive")
        if self.speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh must be positive")
        if self.accident_history_count < 0:
            raise ValueError("accident_history_count must be non-negative")
        object.__setattr__(self, "covered_cells", frozenset(self.covered_cells))


@dataclass(frozen=True)
class WeatherScenario:
    name: str
    weather: WeatherFeatures
    weight: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


@dataclass(frozen=True)
class RoadRiskResult:
    segment_id: str
    risk: float
    accident_probability: float
    priority_score: float
    safety_gain_score: float
    deicing_cost_won: float
    candidate: DeicingCandidate


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def estimate_icing_risk(weather: WeatherFeatures, road: RoadRiskFeatures) -> float:
    """기상 정보와 도로 특성으로 0~1 범위의 결빙 위험도를 추정한다.

    다운로드한 대회 프로젝트 흐름을 참고한 pseudo-label/규칙 모델이다.
    기상 조건이 기본 위험도를 만들고, 그림자와 도로 특성이 도로별 차이를 만든다.
    """
    weather_score = 0.0
    if weather.temp_c <= -10:
        weather_score += 0.30
    elif weather.temp_c <= -5:
        weather_score += 0.25
    elif weather.temp_c <= 0:
        weather_score += 0.20
    elif weather.temp_c <= 3:
        weather_score += 0.05

    if weather.ground_temp_c <= -5:
        weather_score += 0.20
    elif weather.ground_temp_c <= 0:
        weather_score += 0.15
    elif weather.ground_temp_c <= 2:
        weather_score += 0.05

    if weather.snow_cm > 5:
        weather_score += 0.20
    elif weather.snow_cm > 0:
        weather_score += 0.10

    if weather.humidity_pct > 85:
        weather_score += 0.08
    elif weather.humidity_pct > 70:
        weather_score += 0.04

    if weather.precipitation_mm > 0 and weather.temp_c <= 2:
        weather_score += 0.07

    if weather.wind_m_s > 10:
        weather_score += 0.05
    elif weather.wind_m_s > 5:
        weather_score += 0.02

    road_score = 0.0
    road_score += road.shadow_index * 0.40
    if road.lane_count <= 1:
        road_score += 0.08
    elif road.lane_count <= 2:
        road_score += 0.04
    elif road.lane_count >= 6:
        road_score -= 0.02

    length_norm = _clamp(road.length_m / 3000.0)
    road_score += length_norm * 0.04

    shadow_temp = road.shadow_index * max(0.0, -weather.temp_c) / 15.0 * 0.20
    shadow_humid = road.shadow_index * max(0.0, weather.humidity_pct - 60) / 40.0 * 0.10
    road_score += shadow_temp + shadow_humid

    return _clamp(weather_score + road_score)


def weighted_scenario_risk(road: RoadRiskFeatures, scenarios: list[WeatherScenario]) -> float:
    """여러 겨울 기상 시나리오를 하나의 도로 위험도로 결합한다."""
    total_weight = sum(scenario.weight for scenario in scenarios)
    if total_weight <= 0:
        raise ValueError("at least one scenario must have a positive weight")

    risk = sum(estimate_icing_risk(scenario.weather, road) * scenario.weight for scenario in scenarios)
    return _clamp(risk / total_weight)


def estimate_accident_probability(icing_risk: float, road: RoadRiskFeatures) -> float:
    """우선순위 산정과 검증에 사용할 겨울철 사고 확률을 추정한다.

    실제 사고 라벨을 사용할 수 있게 되면 이 산식은 해당 라벨 기반 모델로
    교체해야 한다. 그 전까지는 사고확률 MLP의 결정론적 pseudo-label 목표값이자
    발표용으로 설명 가능한 baseline 역할을 한다.
    """
    if icing_risk < 0:
        raise ValueError("icing_risk must be non-negative")

    exposure = _clamp(road.exposure_score)
    speed_factor = _clamp((road.speed_limit_kmh - 30) / 70)
    narrow_factor = _clamp((3 - min(road.lane_count, 3)) / 2)
    history_factor = _clamp(road.accident_history_count / 5)

    score = (
        0.45 * _clamp(icing_risk)
        + 0.25 * exposure
        + 0.10 * road.shadow_index
        + 0.08 * speed_factor
        + 0.07 * narrow_factor
        + 0.05 * history_factor
    )
    return _clamp(score)


def calculate_safety_gain_score(
    icing_risk: float,
    accident_probability: float,
    exposure_score: float,
) -> float:
    """해당 구간을 제설했을 때 기대되는 안전 개선 효과를 추정한다."""
    if icing_risk < 0 or accident_probability < 0 or exposure_score < 0:
        raise ValueError("risk inputs must be non-negative")
    exposure_weight = 0.7 + 0.3 * _clamp(exposure_score)
    return _clamp((0.45 * _clamp(icing_risk) + 0.55 * _clamp(accident_probability)) * exposure_weight)


def calculate_priority_score(
    risk: float,
    exposure_score: float,
    accident_probability: float | None = None,
) -> float:
    """결빙 위험도, 사고 확률, 노출 점수를 결합해 제설 우선순위를 계산한다."""
    if risk < 0:
        raise ValueError("risk must be non-negative")
    if exposure_score < 0:
        raise ValueError("exposure_score must be non-negative")
    if accident_probability is not None and accident_probability < 0:
        raise ValueError("accident_probability must be non-negative")

    base_risk = _clamp(risk)
    if accident_probability is not None:
        base_risk = 0.55 * base_risk + 0.45 * _clamp(accident_probability)
    exposure_weight = 0.7 + 0.3 * _clamp(exposure_score)
    return base_risk * exposure_weight


def estimate_deicing_cost_won(
    road: RoadRiskFeatures,
    unit_spread_kg_m2: float = 0.03,
    material_cost_per_kg: float = 300.0,
    labor_cost_per_km: float = 50_000.0,
    environmental_cost_per_kg: float = 0.0,
) -> float:
    """도로 면적, 살포재, 인건비, 환경 비용을 이용해 제설 비용을 추정한다."""
    if unit_spread_kg_m2 <= 0:
        raise ValueError("unit_spread_kg_m2 must be positive")
    if material_cost_per_kg < 0 or labor_cost_per_km < 0 or environmental_cost_per_kg < 0:
        raise ValueError("cost terms must be non-negative")

    area_m2 = road.length_m * road.road_width_m
    material_kg = area_m2 * unit_spread_kg_m2
    material_cost = material_kg * material_cost_per_kg
    environmental_cost = material_kg * environmental_cost_per_kg
    labor_cost = road.length_m / 1000.0 * labor_cost_per_km
    return material_cost + environmental_cost + labor_cost


def build_deicing_candidate(
    road: RoadRiskFeatures,
    risk: float,
    accident_probability: float | None = None,
    cost_unit_won: int = 10_000,
) -> RoadRiskResult:
    """도로 특성과 결빙 위험도에서 배차 최적화용 제설 후보를 만든다."""
    if cost_unit_won <= 0:
        raise ValueError("cost_unit_won must be positive")

    accident_prob = (
        estimate_accident_probability(risk, road)
        if accident_probability is None
        else _clamp(accident_probability)
    )
    priority = calculate_priority_score(risk, road.exposure_score, accident_prob)
    safety_gain = calculate_safety_gain_score(risk, accident_prob, road.exposure_score)
    deicing_cost_won = estimate_deicing_cost_won(road)
    candidate_cost = max(1, math.ceil(deicing_cost_won / cost_unit_won))
    service_time = road.length_m / 1000.0

    candidate = DeicingCandidate(
        candidate_id=road.segment_id,
        cost=candidate_cost,
        value=priority,
        covered_cells=road.covered_cells,
        x=road.x,
        y=road.y,
        service_time=service_time,
        demand=max(1, math.ceil(road.length_m / 1000.0)),
    )

    return RoadRiskResult(
        segment_id=road.segment_id,
        risk=_clamp(risk),
        accident_probability=accident_prob,
        priority_score=priority,
        safety_gain_score=safety_gain,
        deicing_cost_won=deicing_cost_won,
        candidate=candidate,
    )
