from src.policy.engine import PolicyEngine, PermissionTier


def test_decision_reason_is_nonempty():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert len(decision.reason) > 0


def test_advisory_tool_exists():
    engine = PolicyEngine()
    decision = engine.evaluate("adjust_setpoint", {})
    assert decision.tier == PermissionTier.ADVISORY


def test_to_dict_contains_reason():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    d = decision.to_dict()
    assert "reason" in d
