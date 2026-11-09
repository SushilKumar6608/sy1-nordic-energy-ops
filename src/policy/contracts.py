"""
Pydantic tool contracts for SY1-Gov.

Every tool has:
  - A typed input model
  - A typed output model
  - A typed error model

The ToolExecutor validates inputs and outputs against these models.
An unvalidated input never reaches the actual tool function.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class FaultType(str, Enum):
    PIPE_BURST = "pipe_burst"
    SUPPLY_FAILURE = "supply_failure"
    INSULATION_DEGRADATION = "insulation_degradation"
    STUCK_VALVE = "stuck_valve"
    SENSOR_DRIFT = "sensor_drift"
    HEAT_EXCHANGER_FOULING = "heat_exchanger_fouling"
    DATA_CENTRE_THERMAL_DUMP = "data_centre_thermal_dump"
    CONTROL_SYSTEM_MISFIRE = "control_system_misfire"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolError(BaseModel):
    tool_name: str
    error_code: str
    message: str
    retryable: bool
    idempotency_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


class GetDemandForecastInput(BaseModel):
    city: str = Field(..., description="City name, e.g. Stockholm")
    target: str = Field(..., description="heating_demand_mw or cooling_demand_mw")
    horizon_hours: int = Field(24, ge=1, le=168)


class GetDemandForecastOutput(BaseModel):
    city: str
    target: str
    forecast_mw: list[float]
    timestamps: list[str]
    model_version: str


class GetAnomalySummaryInput(BaseModel):
    city: str
    target: str
    lookback_hours: int = Field(24, ge=1, le=720)


class GetAnomalySummaryOutput(BaseModel):
    city: str
    target: str
    n_detected: int
    n_true_anom: int
    f1_score: float
    strategy: str


class GetNodeStatusInput(BaseModel):
    node_id: int = Field(..., ge=0, le=136, description="GNN node index 0-136")


class GetNodeStatusOutput(BaseModel):
    node_id: int
    node_type: str
    fault_probability: float
    is_flagged: bool
    last_updated: str


# ---------------------------------------------------------------------------
# Reversible write tools
# ---------------------------------------------------------------------------


class SetAlertThresholdInput(BaseModel):
    node_id: int = Field(..., ge=0, le=136)
    threshold_mw: float = Field(..., gt=0.0, lt=5000.0)
    idempotency_key: str = Field(..., description="UUID for safe retry")

    @field_validator("threshold_mw")
    @classmethod
    def threshold_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("threshold_mw must be positive")
        return v


class SetAlertThresholdOutput(BaseModel):
    node_id: int
    previous_threshold_mw: float
    new_threshold_mw: float
    applied_at: str
    idempotency_key: str


class AcknowledgeIncidentInput(BaseModel):
    incident_id: str = Field(..., description="Incident ID from agent report")
    operator_note: str = Field(..., min_length=10, max_length=500)
    idempotency_key: str


class AcknowledgeIncidentOutput(BaseModel):
    incident_id: str
    acknowledged_at: str
    operator_note: str
    idempotency_key: str


# ---------------------------------------------------------------------------
# Irreversible write tools
# ---------------------------------------------------------------------------


class DispatchMaintenanceCrewInput(BaseModel):
    node_id: int = Field(..., ge=0, le=136)
    fault_type: FaultType
    priority: Priority
    justification: str = Field(
        ...,
        min_length=20,
        max_length=1000,
        description="Why this dispatch is warranted",
    )
    approval_token: str = Field(..., description="Token from logged human approval")
    idempotency_key: str

    @field_validator("justification")
    @classmethod
    def justification_must_be_substantive(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("justification must be at least 20 characters")
        return v


class DispatchMaintenanceCrewOutput(BaseModel):
    dispatch_id: str
    node_id: int
    fault_type: str
    priority: str
    dispatched_at: str
    estimated_arrival_minutes: int
    approval_token: str
    idempotency_key: str


class IsolateGridSegmentInput(BaseModel):
    segment_id: str = Field(..., description="Segment identifier, e.g. SEG-023-024")
    reason: str = Field(..., min_length=20, max_length=1000)
    affected_customer_count: int = Field(
        ..., ge=0, description="Estimated customers affected"
    )
    approval_token: str
    idempotency_key: str


class IsolateGridSegmentOutput(BaseModel):
    isolation_id: str
    segment_id: str
    isolated_at: str
    affected_customer_count: int
    expected_restoration_minutes: Optional[int]
    approval_token: str
    idempotency_key: str


# ---------------------------------------------------------------------------
# Contract registry: maps tool name -> (InputModel, OutputModel)
# ---------------------------------------------------------------------------

TOOL_CONTRACTS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "get_demand_forecast": (GetDemandForecastInput, GetDemandForecastOutput),
    "get_anomaly_summary": (GetAnomalySummaryInput, GetAnomalySummaryOutput),
    "get_node_status": (GetNodeStatusInput, GetNodeStatusOutput),
    "set_alert_threshold": (SetAlertThresholdInput, SetAlertThresholdOutput),
    "acknowledge_incident": (AcknowledgeIncidentInput, AcknowledgeIncidentOutput),
    "dispatch_maintenance_crew": (
        DispatchMaintenanceCrewInput,
        DispatchMaintenanceCrewOutput,
    ),
    "isolate_grid_segment": (IsolateGridSegmentInput, IsolateGridSegmentOutput),
}
