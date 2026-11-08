"""Tests for the model router - all offline, no API key needed."""

from src.policy.engine import PermissionTier
from src.routing.model_router import (
    MODEL_HAIKU,
    MODEL_SONNET,
    RoutingDecision,
    route,
)


def test_read_simple_routes_to_haiku():
    decision = route("get_demand_forecast", PermissionTier.READ, {})
    assert decision.model == MODEL_HAIKU


def test_read_complex_routes_to_sonnet():
    inputs = {"a": "x" * 300}
    decision = route("get_demand_forecast", PermissionTier.READ, inputs)
    assert decision.model == MODEL_SONNET


def test_reversible_always_routes_to_sonnet():
    decision = route("set_alert_threshold", PermissionTier.REVERSIBLE, {})
    assert decision.model == MODEL_SONNET


def test_irreversible_always_routes_to_sonnet():
    decision = route("dispatch_maintenance_crew", PermissionTier.IRREVERSIBLE, {})
    assert decision.model == MODEL_SONNET


def test_kill_switch_routes_to_haiku():
    decision = route(
        "dispatch_maintenance_crew",
        PermissionTier.IRREVERSIBLE,
        {},
        kill_switch_active=True,
    )
    assert decision.model == MODEL_HAIKU


def test_kill_switch_overrides_irreversible():
    decision = route(
        "isolate_grid_segment",
        PermissionTier.IRREVERSIBLE,
        {"segment_id": "S-02"},
        kill_switch_active=True,
    )
    assert decision.model == MODEL_HAIKU
    assert "kill switch" in decision.reason.lower()


def test_routing_decision_has_reason():
    decision = route("get_demand_forecast", PermissionTier.READ, {})
    assert isinstance(decision.reason, str)
    assert len(decision.reason) > 0


def test_routing_decision_has_complexity():
    decision = route("get_demand_forecast", PermissionTier.READ, {})
    assert decision.input_complexity in ("simple", "complex")


def test_many_keys_is_complex():
    inputs = {"a": 1, "b": 2, "c": 3, "d": 4}
    decision = route("get_node_status", PermissionTier.READ, inputs)
    assert decision.input_complexity == "complex"
    assert decision.model == MODEL_SONNET


def test_simple_inputs_is_simple():
    decision = route("get_node_status", PermissionTier.READ, {"node_id": 3})
    assert decision.input_complexity == "simple"


def test_returns_routing_decision_type():
    decision = route("get_demand_forecast", PermissionTier.READ, {})
    assert isinstance(decision, RoutingDecision)


def test_irreversible_reason_mentions_tool():
    decision = route("dispatch_maintenance_crew", PermissionTier.IRREVERSIBLE, {})
    assert "dispatch_maintenance_crew" in decision.reason
