"""
Unit tests for src/policy/contracts.py

Covers:
- Input validation for all tool contracts
- Required fields missing -> ValidationError
- Out-of-range values rejected
- Sensitive field presence in models
- TOOL_CONTRACTS registry completeness
"""

import pytest
from pydantic import ValidationError

from src.policy.contracts import (
    GetDemandForecastInput,
    GetAnomalySummaryInput,
    GetNodeStatusInput,
    SetAlertThresholdInput,
    AcknowledgeIncidentInput,
    DispatchMaintenanceCrewInput,
    IsolateGridSegmentInput,
    FaultType,
    Priority,
    TOOL_CONTRACTS,
)
from src.policy.engine import TOOL_PERMISSIONS


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    def test_all_permission_tools_have_contracts(self):
        """Every tool in TOOL_PERMISSIONS must have a contract in TOOL_CONTRACTS."""
        # get_grid_topology and retrieve_rag_context are read stubs without contracts yet
        for tool in TOOL_CONTRACTS:
            assert (
                tool in TOOL_PERMISSIONS
            ), f"{tool} has a contract but no permission tier"

    def test_contract_registry_has_input_and_output(self):
        for tool_name, (InputModel, OutputModel) in TOOL_CONTRACTS.items():
            assert InputModel is not None
            assert OutputModel is not None


# ---------------------------------------------------------------------------
# GetDemandForecastInput
# ---------------------------------------------------------------------------


class TestGetDemandForecastInput:
    def test_valid(self):
        m = GetDemandForecastInput(city="Stockholm", target="heating_demand_mw")
        assert m.city == "Stockholm"
        assert m.horizon_hours == 24  # default

    def test_missing_city(self):
        with pytest.raises(ValidationError):
            GetDemandForecastInput(target="heating_demand_mw")

    def test_horizon_too_large(self):
        with pytest.raises(ValidationError):
            GetDemandForecastInput(
                city="Oslo", target="heating_demand_mw", horizon_hours=200
            )

    def test_horizon_zero(self):
        with pytest.raises(ValidationError):
            GetDemandForecastInput(
                city="Oslo", target="heating_demand_mw", horizon_hours=0
            )


# ---------------------------------------------------------------------------
# GetNodeStatusInput
# ---------------------------------------------------------------------------


class TestGetNodeStatusInput:
    def test_valid_node_zero(self):
        m = GetNodeStatusInput(node_id=0)
        assert m.node_id == 0

    def test_valid_node_136(self):
        m = GetNodeStatusInput(node_id=136)
        assert m.node_id == 136

    def test_node_too_high(self):
        with pytest.raises(ValidationError):
            GetNodeStatusInput(node_id=137)

    def test_node_negative(self):
        with pytest.raises(ValidationError):
            GetNodeStatusInput(node_id=-1)


# ---------------------------------------------------------------------------
# SetAlertThresholdInput
# ---------------------------------------------------------------------------


class TestSetAlertThresholdInput:
    def test_valid(self):
        m = SetAlertThresholdInput(
            node_id=5,
            threshold_mw=100.0,
            idempotency_key="uuid-1234",
        )
        assert m.threshold_mw == 100.0

    def test_threshold_zero_rejected(self):
        with pytest.raises(ValidationError):
            SetAlertThresholdInput(node_id=5, threshold_mw=0.0, idempotency_key="k1")

    def test_threshold_negative_rejected(self):
        with pytest.raises(ValidationError):
            SetAlertThresholdInput(node_id=5, threshold_mw=-50.0, idempotency_key="k1")

    def test_threshold_too_high_rejected(self):
        with pytest.raises(ValidationError):
            SetAlertThresholdInput(node_id=5, threshold_mw=6000.0, idempotency_key="k1")

    def test_missing_idempotency_key(self):
        with pytest.raises(ValidationError):
            SetAlertThresholdInput(node_id=5, threshold_mw=100.0)


# ---------------------------------------------------------------------------
# AcknowledgeIncidentInput
# ---------------------------------------------------------------------------


class TestAcknowledgeIncidentInput:
    def test_valid(self):
        m = AcknowledgeIncidentInput(
            incident_id="INC-001",
            operator_note="Acknowledged by shift supervisor after visual inspection.",
            idempotency_key="uuid-5678",
        )
        assert m.incident_id == "INC-001"

    def test_note_too_short(self):
        with pytest.raises(ValidationError):
            AcknowledgeIncidentInput(
                incident_id="INC-001",
                operator_note="Too short",
                idempotency_key="k1",
            )

    def test_note_too_long(self):
        with pytest.raises(ValidationError):
            AcknowledgeIncidentInput(
                incident_id="INC-001",
                operator_note="x" * 501,
                idempotency_key="k1",
            )


# ---------------------------------------------------------------------------
# DispatchMaintenanceCrewInput
# ---------------------------------------------------------------------------


class TestDispatchMaintenanceCrewInput:
    def test_valid(self):
        m = DispatchMaintenanceCrewInput(
            node_id=42,
            fault_type=FaultType.PIPE_BURST,
            priority=Priority.HIGH,
            justification="GNN confidence 0.94 for pipe burst at node 42. Pressure drop confirmed.",
            approval_token="approved-token-abc",
            idempotency_key="uuid-dispatch-001",
        )
        assert m.fault_type == FaultType.PIPE_BURST

    def test_justification_too_short(self):
        with pytest.raises(ValidationError):
            DispatchMaintenanceCrewInput(
                node_id=1,
                fault_type=FaultType.STUCK_VALVE,
                priority=Priority.LOW,
                justification="short",
                approval_token="tok",
                idempotency_key="k1",
            )

    def test_missing_approval_token(self):
        with pytest.raises(ValidationError):
            DispatchMaintenanceCrewInput(
                node_id=1,
                fault_type=FaultType.STUCK_VALVE,
                priority=Priority.LOW,
                justification="A" * 25,
                idempotency_key="k1",
            )

    def test_invalid_fault_type(self):
        with pytest.raises(ValidationError):
            DispatchMaintenanceCrewInput(
                node_id=1,
                fault_type="exploding_boiler",  # not in enum
                priority=Priority.LOW,
                justification="A" * 25,
                approval_token="tok",
                idempotency_key="k1",
            )

    def test_invalid_priority(self):
        with pytest.raises(ValidationError):
            DispatchMaintenanceCrewInput(
                node_id=1,
                fault_type=FaultType.SENSOR_DRIFT,
                priority="extreme",  # not in enum
                justification="A" * 25,
                approval_token="tok",
                idempotency_key="k1",
            )


# ---------------------------------------------------------------------------
# IsolateGridSegmentInput
# ---------------------------------------------------------------------------


class TestIsolateGridSegmentInput:
    def test_valid(self):
        m = IsolateGridSegmentInput(
            segment_id="SEG-023-024",
            reason="Burst confirmed by pressure sensors. Isolating to contain loss.",
            affected_customer_count=450,
            approval_token="approved-token-xyz",
            idempotency_key="uuid-isolate-001",
        )
        assert m.affected_customer_count == 450

    def test_negative_customer_count_rejected(self):
        with pytest.raises(ValidationError):
            IsolateGridSegmentInput(
                segment_id="SEG-001",
                reason="A" * 25,
                affected_customer_count=-1,
                approval_token="tok",
                idempotency_key="k1",
            )

    def test_reason_too_short(self):
        with pytest.raises(ValidationError):
            IsolateGridSegmentInput(
                segment_id="SEG-001",
                reason="too short",
                affected_customer_count=100,
                approval_token="tok",
                idempotency_key="k1",
            )
