"""
Unit tests for src/policy/engine.py

Covers:
- All three permission tiers
- Kill switch activation/deactivation
- Unknown tool (fail-closed)
- PolicyDecision.to_dict()
"""

import pytest

from src.policy.engine import (
    PolicyEngine,
    PermissionTier,
    TOOL_PERMISSIONS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return PolicyEngine()


@pytest.fixture
def engine_kill_switch():
    return PolicyEngine(kill_switch_active=True)


# ---------------------------------------------------------------------------
# READ tier
# ---------------------------------------------------------------------------


class TestReadTier:
    def test_read_tool_allowed(self, engine):
        decision = engine.evaluate("get_demand_forecast", {"city": "Stockholm"})
        assert decision.allowed is True
        assert decision.tier == PermissionTier.READ
        assert decision.requires_approval is False

    def test_read_tool_no_approval_needed(self, engine):
        decision = engine.evaluate("get_anomaly_summary", {})
        assert decision.requires_approval is False

    def test_all_read_tools_allowed(self, engine):
        read_tools = [
            t for t, tier in TOOL_PERMISSIONS.items() if tier == PermissionTier.READ
        ]
        for tool in read_tools:
            d = engine.evaluate(tool, {})
            assert d.allowed is True, f"{tool} should be allowed"


# ---------------------------------------------------------------------------
# REVERSIBLE tier
# ---------------------------------------------------------------------------


class TestReversibleTier:
    def test_reversible_tool_allowed(self, engine):
        decision = engine.evaluate("set_alert_threshold", {"node_id": 1})
        assert decision.allowed is True
        assert decision.tier == PermissionTier.REVERSIBLE
        assert decision.requires_approval is False

    def test_acknowledge_incident_allowed(self, engine):
        decision = engine.evaluate("acknowledge_incident", {})
        assert decision.allowed is True

    def test_all_reversible_tools_allowed(self, engine):
        reversible_tools = [
            t
            for t, tier in TOOL_PERMISSIONS.items()
            if tier == PermissionTier.REVERSIBLE
        ]
        for tool in reversible_tools:
            d = engine.evaluate(tool, {})
            assert d.allowed is True, f"{tool} should be allowed"


# ---------------------------------------------------------------------------
# IRREVERSIBLE tier
# ---------------------------------------------------------------------------


class TestIrreversibleTier:
    def test_irreversible_tool_blocked(self, engine):
        decision = engine.evaluate("dispatch_maintenance_crew", {})
        assert decision.allowed is False
        assert decision.tier == PermissionTier.IRREVERSIBLE
        assert decision.requires_approval is True

    def test_isolate_grid_segment_blocked(self, engine):
        decision = engine.evaluate("isolate_grid_segment", {})
        assert decision.allowed is False
        assert decision.requires_approval is True

    def test_irreversible_reason_mentions_approval(self, engine):
        decision = engine.evaluate("dispatch_maintenance_crew", {})
        assert "approval" in decision.reason.lower()

    def test_all_irreversible_tools_blocked(self, engine):
        irreversible_tools = [
            t
            for t, tier in TOOL_PERMISSIONS.items()
            if tier == PermissionTier.IRREVERSIBLE
        ]
        for tool in irreversible_tools:
            d = engine.evaluate(tool, {})
            assert d.allowed is False, f"{tool} should be blocked"
            assert d.requires_approval is True, f"{tool} should require approval"


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_kill_switch_blocks_all_writes(self, engine_kill_switch):
        for tool in TOOL_PERMISSIONS:
            d = engine_kill_switch.evaluate(tool, {})
            assert d.allowed is False, f"Kill switch should block {tool}"

    def test_kill_switch_message(self, engine_kill_switch):
        d = engine_kill_switch.evaluate("set_alert_threshold", {})
        assert "kill switch" in d.reason.lower()

    def test_deactivate_restores_normal_behavior(self):
        engine = PolicyEngine(kill_switch_active=True)
        engine.deactivate_kill_switch()
        d = engine.evaluate("get_demand_forecast", {})
        assert d.allowed is True

    def test_activate_then_deactivate_cycle(self):
        engine = PolicyEngine()
        d = engine.evaluate("get_demand_forecast", {})
        assert d.allowed is True

        engine.activate_kill_switch()
        d = engine.evaluate("get_demand_forecast", {})
        assert d.allowed is False

        engine.deactivate_kill_switch()
        d = engine.evaluate("get_demand_forecast", {})
        assert d.allowed is True


# ---------------------------------------------------------------------------
# Unknown tool - fail closed
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_raises(self, engine):
        with pytest.raises(ValueError, match="Unknown tool"):
            engine.evaluate("delete_all_data", {})

    def test_unknown_tool_does_not_default_to_allowed(self, engine):
        with pytest.raises(ValueError):
            engine.evaluate("nonexistent_tool", {})

    def test_kill_switch_still_blocks_before_unknown_check(self, engine_kill_switch):
        # Kill switch fires before the unknown-tool check
        d = engine_kill_switch.evaluate("get_demand_forecast", {})
        assert d.allowed is False


# ---------------------------------------------------------------------------
# PolicyDecision serialization
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    def test_to_dict_has_required_keys(self, engine):
        d = engine.evaluate("get_demand_forecast", {})
        result = d.to_dict()
        assert "tool_name" in result
        assert "tier" in result
        assert "allowed" in result
        assert "reason" in result
        assert "requires_approval" in result

    def test_to_dict_tier_is_string(self, engine):
        d = engine.evaluate("dispatch_maintenance_crew", {})
        result = d.to_dict()
        assert isinstance(result["tier"], str)
        assert result["tier"] == "irreversible"

    def test_to_dict_read_tier(self, engine):
        d = engine.evaluate("get_node_status", {"node_id": 0})
        result = d.to_dict()
        assert result["tier"] == "read"
        assert result["allowed"] is True
        assert result["requires_approval"] is False
