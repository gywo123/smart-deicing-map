from src.risk_scoring import (
    RoadRiskFeatures,
    WeatherFeatures,
    WeatherScenario,
    build_deicing_candidate,
    calculate_priority_score,
    estimate_deicing_cost_won,
    estimate_icing_risk,
    weighted_scenario_risk,
)


def test_estimate_icing_risk_increases_for_shadow_and_cold():
    mild_weather = WeatherFeatures(
        temp_c=2,
        ground_temp_c=1,
        humidity_pct=60,
        wind_m_s=1,
        snow_cm=0,
    )
    cold_weather = WeatherFeatures(
        temp_c=-8,
        ground_temp_c=-6,
        humidity_pct=90,
        wind_m_s=7,
        snow_cm=3,
    )
    open_road = RoadRiskFeatures("S1", length_m=500, lane_count=4, shadow_index=0.0)
    shadow_road = RoadRiskFeatures("S1", length_m=500, lane_count=4, shadow_index=0.8)

    assert estimate_icing_risk(cold_weather, shadow_road) > estimate_icing_risk(cold_weather, open_road)
    assert estimate_icing_risk(cold_weather, open_road) > estimate_icing_risk(mild_weather, open_road)


def test_weighted_scenario_risk_and_candidate_creation():
    road = RoadRiskFeatures(
        "S-100",
        length_m=1200,
        lane_count=2,
        shadow_index=0.5,
        exposure_score=0.8,
        road_width_m=8,
        covered_cells=frozenset({"g1", "g2"}),
        x=1,
        y=2,
    )
    scenarios = [
        WeatherScenario("cold", WeatherFeatures(-8, -6, 90, 7, snow_cm=3), weight=0.7),
        WeatherScenario("mild", WeatherFeatures(-1, 0.5, 65, 2), weight=0.3),
    ]

    risk = weighted_scenario_risk(road, scenarios)
    result = build_deicing_candidate(road, risk)

    assert 0 <= risk <= 1
    assert result.segment_id == "S-100"
    assert result.priority_score == calculate_priority_score(risk, 0.8)
    assert result.deicing_cost_won == estimate_deicing_cost_won(road)
    assert result.candidate.candidate_id == "S-100"
    assert result.candidate.covered_cells == frozenset({"g1", "g2"})
