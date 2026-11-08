from src.policy.engine import PolicyEngine, PermissionTier


def test_policy_engine_is_deterministic():
    engine = PolicyEngine()
    d1 = engine.evaluate("get_sensor_reading", {})
    d2 = engine.evaluate("get_sensor_reading", {})
    assert d1.tier == d2.tier
    assert d1.reason == d2.reason


def test_to_dict_has_expected_keys():
    engine = PolicyEngine()
    d = engine.evaluate("get_sensor_reading", {}).to_dict()
    assert set(d.keys()) >= {"tier", "reason"}


def test_different_tools_can_have_different_tiers():
    engine = PolicyEngine()
    r = engine.evaluate("get_sensor_reading", {}).tier
    i = engine.evaluate("dispatch_maintenance_crew", {}).tier
    assert r != i
