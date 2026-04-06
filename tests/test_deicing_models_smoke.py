from datetime import datetime, timedelta, timezone

from src.deicing_models import (
    BoundingBox,
    DeicingOperation,
    DeicingZone,
    GeoPoint,
    MaterialApplication,
    MaterialType,
    OperationStatus,
    Organization,
    RiskPrediction,
    RoadSegment,
    SeverityLevel,
    SmartDeicingMapModel,
    SurfaceState,
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
                predicted_surface_state=SurfaceState.ICE,
                predicted_severity=SeverityLevel.HIGH,
                recommended_material=MaterialType.CALCIUM_CHLORIDE,
                recommended_spread_rate_g_m2=24,
            )
        ],
    )

    assert model.organization.org_id == "ORG-SEOUL-001"
    assert model.zones[0].service_priority == 1
    assert model.operations[0].status == OperationStatus.COMPLETED
    assert model.predictions[0].predicted_surface_state == SurfaceState.ICE
