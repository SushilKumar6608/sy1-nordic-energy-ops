from src.policy.engine import PolicyEngine, PermissionTier


def test_permission_tier_has_three_values():
    tiers = list(PermissionTier)
    assert len(tiers) == 3


def test_read_only_is_lowest_tier():
    assert PermissionTier.READ_ONLY != PermissionTier.IRREVERSIBLE


def test_policy_engine_evaluates_all_tiers():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert decision.tier == PermissionTier.READ_ONLY


def test_irreversible_tool_tier():
    engine = PolicyEngine()
    decision = engine.evaluate("dispatch_maintenance_crew", {})
    assert decision.tier == PermissionTier.IRREVERSIBLE
