from src.policy.engine import PolicyEngine, PermissionTier


def test_evaluate_returns_decision_object():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert decision is not None


def test_multiple_tools_evaluate_consistently():
    engine = PolicyEngine()
    d1 = engine.evaluate("get_sensor_reading", {})
    d2 = engine.evaluate("get_sensor_reading", {})
    assert d1.tier == d2.tier


def test_irreversible_decision_to_dict():
    engine = PolicyEngine()
    decision = engine.evaluate("dispatch_maintenance_crew", {})
    d = decision.to_dict()
    assert d["tier"] == PermissionTier.IRREVERSIBLE.value
