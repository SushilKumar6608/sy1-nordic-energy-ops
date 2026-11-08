from src.policy.engine import PolicyEngine, PermissionTier


def test_permission_tier_has_three_values():
    assert len(list(PermissionTier)) == 3


def test_tiers_are_distinct():
    assert PermissionTier.READ_ONLY != PermissionTier.ADVISORY
    assert PermissionTier.ADVISORY != PermissionTier.IRREVERSIBLE


def test_read_tool_is_read_only():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert decision.tier == PermissionTier.READ_ONLY


def test_irreversible_tool_tier():
    engine = PolicyEngine()
    decision = engine.evaluate("dispatch_maintenance_crew", {})
    assert decision.tier == PermissionTier.IRREVERSIBLE
