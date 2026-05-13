from datetime import datetime, timedelta, timezone

from src.deicing_models import (
    AccidentPrediction,
    BoundingBox,
    DeicingOperation,
    DeicingZone,
    GridCell,
    GeoPoint,
    MaterialApplication,
    MaterialType,
    MobilitySnapshot,
    OperationStatus,
    Organization,
    RiskPrediction,
    RoadSegment,
    SeverityLevel,
    SmartDeicingMapModel,
    SurfaceState,
    UrbanExposure,
)


def test_model_smoke_detailed():
    now = datetime.now(timezone.utc)

    model = SmartDeicingMapModel(
        generated_at=now,
        organization=Organization(
            org_id="ORG-SEOUL-001",
            org_name="Seoul Winter Ops Center",
            jurisdiction_code="KR-11",
        ),
        zones=[
            DeicingZone(
                zone_id="Z-01",
                org_id="ORG-SEOUL-001",
                zone_name="Gangnam",
                service_priority=1,
                center=GeoPoint(lat=37.4979, lon=127.0276),
                coverage_bbox=BoundingBox(
                    min_lat=37.45,
                    min_lon=126.99,
                    max_lat=37.54,
                    max_lon=127.06,
                ),
            )
        ],
        road_segments=[
            RoadSegment(
                segment_id="S-100",
                zone_id="Z-01",
                road_name="Teheran-ro",
                lane_count=4,
                length_m=2400,
                speed_limit_kmh=60,
                slope_pct=4.5,
                is_bridge=False,
                geometry=[
                    GeoPoint(lat=37.5001, lon=127.0261),
                    GeoPoint(lat=37.5025, lon=127.0310),
                ],
            )
        ],
        grid_cells=[
            GridCell(
                cell_id="G-10-12",
                zone_id="Z-01",
                row=10,
                col=12,
                center=GeoPoint(lat=37.5010, lon=127.0270),
                segment_id="S-100",
            )
        ],
        urban_exposures=[
            UrbanExposure(
                segment_id="S-100",
                measured_at=now,
                shadow_index=0.78,
                sunlight_minutes_last_hour=8,
                geothermal_index=0.2,
                heat_island_index=0.35,
            )
        ],
        mobility_snapshots=[
            MobilitySnapshot(
                segment_id="S-100",
                observed_at=now,
                pedestrian_count=120,
                vehicle_count=340,
                bicycle_count=18,
                exposure_score=1.7,
            )
        ],
        operations=[
            DeicingOperation(
                operation_id="OP-1",
                zone_id="Z-01",
                segment_ids=["S-100"],
                vehicle_id="V-12",
                operator_id="U-77",
                planned_start_at=now,
                actual_start_at=now + timedelta(minutes=10),
                actual_end_at=now + timedelta(minutes=40),
                status=OperationStatus.COMPLETED,
                material_applications=[
                    MaterialApplication(
                        material=MaterialType.BRINE,
                        quantity_kg=500,
                        spread_rate_g_m2=18,
                    )
                ],
            )
        ],
        predictions=[
            RiskPrediction(
                segment_id="S-100",
                predicted_at=now,
                target_time=now + timedelta(hours=1),
                icing_probability=0.73,
                accident_probability=0.62,
                predicted_surface_state=SurfaceState.ICE,
                predicted_severity=SeverityLevel.HIGH,
                exposure_score=1.7,
                safety_gain_score=1.241,
                recommended_material=MaterialType.CALCIUM_CHLORIDE,
                recommended_spread_rate_g_m2=24,
            )
        ],
        accident_predictions=[
            AccidentPrediction(
                segment_id="S-100",
                predicted_at=now,
                target_time=now + timedelta(hours=1),
                accident_probability=0.62,
                icing_probability=0.73,
                exposure_score=1.7,
                safety_gain_score=1.241,
            )
        ],
    )

    assert model.organization.org_id == "ORG-SEOUL-001"
    assert model.zones[0].service_priority == 1
    assert model.grid_cells[0].segment_id == "S-100"
    assert model.urban_exposures[0].shadow_index == 0.78
    assert model.mobility_snapshots[0].exposure_score == 1.7
    assert model.operations[0].status == OperationStatus.COMPLETED
    assert model.predictions[0].predicted_surface_state == SurfaceState.ICE
    assert model.accident_predictions[0].accident_probability == 0.62
