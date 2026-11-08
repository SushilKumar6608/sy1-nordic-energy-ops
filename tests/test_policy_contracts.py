from src.policy.engine import PolicyEngine, PermissionTier


def test_permission_tier_has_three_values():
    assert len(list(PermissionTier)) == 3


def test_tiers_are_distinct():
    assert PermissionTier.READ != PermissionTier.REVERSIBLE
    assert PermissionTier.REVERSIBLE != PermissionTier.IRREVERSIBLE


def test_read_tool_tier():
    engine = PolicyEngine()
    decision = engine.evaluate("get_demand_forecast", {})
    assert decision.tier == PermissionTier.READ


def test_irreversible_tool_tier():
    engine = PolicyEngine()
    decision = engine.evaluate("dispatch_maintenance_crew", {})
    assert decision.tier == PermissionTier.IRREVERSIBLE
