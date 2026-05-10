"""Risk scoring helpers for the Smart Deicing Map MVP.

This module mirrors the contest reference pipeline in a lightweight,
dependency-free form. It provides rule-based pseudo-risk scores until real
road-icing labels or a trained tabular model are available.
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
    priority_score: float
    deicing_cost_won: float
    candidate: DeicingCandidate


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def estimate_icing_risk(weather: WeatherFeatures, road: RoadRiskFeatures) -> float:
    """Estimate a 0..1 icing risk score from weather and road features.

    This is a pseudo-label/rule model inspired by the downloaded contest
    project: weather drives the base risk, while shadow and road characteristics
    create road-by-road variation.
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
    """Combine multiple winter scenarios into one road risk score."""
    total_weight = sum(scenario.weight for scenario in scenarios)
    if total_weight <= 0:
        raise ValueError("at least one scenario must have a positive weight")

    risk = sum(estimate_icing_risk(scenario.weather, road) * scenario.weight for scenario in scenarios)
    return _clamp(risk / total_weight)


def calculate_priority_score(risk: float, exposure_score: float) -> float:
    """Combine icing risk and exposure into a treatment priority score."""
    if risk < 0:
        raise ValueError("risk must be non-negative")
    if exposure_score < 0:
        raise ValueError("exposure_score must be non-negative")
    exposure_weight = 0.7 + 0.3 * _clamp(exposure_score)
    return _clamp(risk) * exposure_weight


def estimate_deicing_cost_won(
    road: RoadRiskFeatures,
    unit_spread_kg_m2: float = 0.03,
    material_cost_per_kg: float = 300.0,
    labor_cost_per_km: float = 50_000.0,
    environmental_cost_per_kg: float = 0.0,
) -> float:
    """Estimate treatment cost using road area, material, labor, and environment terms."""
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
    cost_unit_won: int = 10_000,
) -> RoadRiskResult:
    """Build a dispatch candidate from road features and an icing risk score."""
    if cost_unit_won <= 0:
        raise ValueError("cost_unit_won must be positive")

    priority = calculate_priority_score(risk, road.exposure_score)
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
        priority_score=priority,
        deicing_cost_won=deicing_cost_won,
        candidate=candidate,
    )
