from src.policy.engine import PolicyEngine


def test_policy_engine_is_deterministic():
    engine = PolicyEngine()
    d1 = engine.evaluate("get_demand_forecast", {})
    d2 = engine.evaluate("get_demand_forecast", {})
    assert d1.tier == d2.tier
    assert d1.reason == d2.reason


def test_to_dict_has_expected_keys():
    engine = PolicyEngine()
    d = engine.evaluate("get_demand_forecast", {}).to_dict()
    assert set(d.keys()) >= {"tier", "reason"}


def test_different_tools_have_different_tiers():
    engine = PolicyEngine()
    r = engine.evaluate("get_demand_forecast", {}).tier
    i = engine.evaluate("dispatch_maintenance_crew", {}).tier
    assert r != i
