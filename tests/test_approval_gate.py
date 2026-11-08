from src.policy.engine import PolicyEngine, PermissionTier


def test_kill_switch_blocks_all_writes():
    engine = PolicyEngine(kill_switch_active=True)
    decision = engine.evaluate("dispatch_maintenance_crew", {})
    assert decision.allowed is False


def test_kill_switch_also_blocks_reversible():
    engine = PolicyEngine(kill_switch_active=True)
    decision = engine.evaluate("set_alert_threshold", {})
    assert decision.allowed is False


def test_kill_switch_deactivate_restores():
    engine = PolicyEngine(kill_switch_active=True)
    engine.deactivate_kill_switch()
    decision = engine.evaluate("get_demand_forecast", {})
    assert decision.allowed is True


def test_unknown_tool_raises_value_error():
    import pytest

    engine = PolicyEngine()
    with pytest.raises(ValueError):
        engine.evaluate("nonexistent_tool_xyz", {})
