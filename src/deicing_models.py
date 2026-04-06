"""Detailed domain models for Smart Deicing Map."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SurfaceState(StrEnum):
    DRY = "dry"
    DAMP = "damp"
    WET = "wet"
    SLUSH = "slush"
    ICE = "ice"
    HARDPACKED_SNOW = "hardpacked_snow"
    FRESH_SNOW = "fresh_snow"


class SeverityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrecipitationType(StrEnum):
    NONE = "none"
    RAIN = "rain"
    SNOW = "snow"
    SLEET = "sleet"
    FREEZING_RAIN = "freezing_rain"


class MaterialType(StrEnum):
    ROCK_SALT = "rock_salt"
    BRINE = "brine"
    CALCIUM_CHLORIDE = "calcium_chloride"
    MAGNESIUM_CHLORIDE = "magnesium_chloride"
    ABRASIVE = "abrasive"


class OperationStatus(StrEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AlertType(StrEnum):
    BLACK_ICE_RISK = "black_ice_risk"
    HEAVY_SNOW = "heavy_snow"
    SENSOR_FAULT = "sensor_fault"
    MATERIAL_SHORTAGE = "material_shortage"


class GeoPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class BoundingBox(BaseModel):
    min_lat: float = Field(ge=-90, le=90)
    min_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.min_lat > self.max_lat:
            raise ValueError("min_lat must be <= max_lat")
        if self.min_lon > self.max_lon:
            raise ValueError("min_lon must be <= max_lon")
        return self


class Organization(BaseModel):
    org_id: str = Field(min_length=1)
    org_name: str = Field(min_length=1)
    jurisdiction_code: str = Field(min_length=2, max_length=20)


class DeicingZone(BaseModel):
    zone_id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    zone_name: str = Field(min_length=1)
    service_priority: int = Field(ge=1, le=5, description="1 is highest priority")
    center: GeoPoint
    coverage_bbox: BoundingBox


class RoadSegment(BaseModel):
    segment_id: str = Field(min_length=1)
    zone_id: str = Field(min_length=1)
    road_name: str = Field(min_length=1)
    lane_count: int = Field(ge=1, le=16)
    length_m: float = Field(gt=0)
    speed_limit_kmh: int = Field(ge=10, le=140)
    slope_pct: float = Field(ge=0, le=30)
    is_bridge: bool = False
    geometry: list[GeoPoint] = Field(min_length=2)


class SensorStation(BaseModel):
    station_id: str = Field(min_length=1)
    zone_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    location: GeoPoint
    installed_at: datetime
    active: bool = True


class WeatherSnapshot(BaseModel):
    station_id: str = Field(min_length=1)
    observed_at: datetime
    air_temp_c: float = Field(ge=-60, le=60)
    road_temp_c: float = Field(ge=-60, le=80)
    dew_point_c: float = Field(ge=-80, le=50)
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_m_s: float = Field(ge=0, le=80)
    wind_gust_m_s: float = Field(ge=0, le=100)
    precipitation_type: PrecipitationType = PrecipitationType.NONE
    precipitation_mm_h: float = Field(ge=0, le=300)


class RoadCondition(BaseModel):
    segment_id: str = Field(min_length=1)
    observed_at: datetime
    surface_state: SurfaceState
    severity: SeverityLevel
    friction_coefficient: float = Field(ge=0, le=1)
    snow_depth_cm: float = Field(ge=0, le=200)
    ice_thickness_mm: float = Field(ge=0, le=50)
    confidence: float = Field(ge=0, le=1)


class MaterialApplication(BaseModel):
    material: MaterialType
    quantity_kg: float = Field(gt=0)
    spread_rate_g_m2: float = Field(gt=0)


class DeicingOperation(BaseModel):
    operation_id: str = Field(min_length=1)
    zone_id: str = Field(min_length=1)
    segment_ids: list[str] = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    operator_id: str = Field(min_length=1)
    planned_start_at: datetime
    actual_start_at: datetime | None = None
    actual_end_at: datetime | None = None
    status: OperationStatus = OperationStatus.PLANNED
    material_applications: list[MaterialApplication] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "DeicingOperation":
        if self.actual_start_at and self.actual_start_at < self.planned_start_at:
            raise ValueError("actual_start_at must be >= planned_start_at")
        if self.actual_end_at and self.actual_start_at and self.actual_end_at < self.actual_start_at:
            raise ValueError("actual_end_at must be >= actual_start_at")
        return self


class RiskPrediction(BaseModel):
    segment_id: str = Field(min_length=1)
    predicted_at: datetime
    target_time: datetime
    icing_probability: float = Field(ge=0, le=1)
    predicted_surface_state: SurfaceState
    predicted_severity: SeverityLevel
    recommended_material: MaterialType
    recommended_spread_rate_g_m2: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_target_time(self) -> "RiskPrediction":
        if self.target_time <= self.predicted_at:
            raise ValueError("target_time must be later than predicted_at")
        return self


class AlertEvent(BaseModel):
    alert_id: str = Field(min_length=1)
    alert_type: AlertType
    severity: SeverityLevel
    zone_id: str = Field(min_length=1)
    segment_id: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    message: str = Field(min_length=5, max_length=500)


class SmartDeicingMapModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    organization: Organization
    zones: list[DeicingZone] = Field(default_factory=list)
    road_segments: list[RoadSegment] = Field(default_factory=list)
    stations: list[SensorStation] = Field(default_factory=list)
    weather_snapshots: list[WeatherSnapshot] = Field(default_factory=list)
    road_conditions: list[RoadCondition] = Field(default_factory=list)
    operations: list[DeicingOperation] = Field(default_factory=list)
    predictions: list[RiskPrediction] = Field(default_factory=list)
    alerts: list[AlertEvent] = Field(default_factory=list)
