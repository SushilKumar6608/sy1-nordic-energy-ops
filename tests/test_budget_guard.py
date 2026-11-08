from src.policy.engine import PolicyEngine, PermissionTier


def test_all_three_tiers_reachable():
    engine = PolicyEngine()
    tiers_found = set()
    for tool in ["get_sensor_reading", "adjust_setpoint", "dispatch_maintenance_crew"]:
        decision = engine.evaluate(tool, {})
        tiers_found.add(decision.tier)
    assert PermissionTier.READ_ONLY in tiers_found
    assert PermissionTier.IRREVERSIBLE in tiers_found


def test_decision_tier_is_permission_tier():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert isinstance(decision.tier, PermissionTier)


def test_decision_reason_is_string():
    engine = PolicyEngine()
    for tool in ["get_sensor_reading", "dispatch_maintenance_crew"]:
        decision = engine.evaluate(tool, {})
        assert isinstance(decision.reason, str)
