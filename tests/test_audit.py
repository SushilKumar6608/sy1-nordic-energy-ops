from src.policy.engine import PolicyEngine, PermissionTier


def test_reversible_tool_is_allowed():
    engine = PolicyEngine()
    decision = engine.evaluate("set_alert_threshold", {})
    assert decision.allowed is True
    assert decision.tier == PermissionTier.REVERSIBLE


def test_irreversible_tool_requires_approval():
    engine = PolicyEngine()
    decision = engine.evaluate("dispatch_maintenance_crew", {})
    assert decision.requires_approval is True
    assert decision.allowed is False


def test_to_dict_has_all_keys():
    engine = PolicyEngine()
    d = engine.evaluate("get_demand_forecast", {}).to_dict()
    assert set(d.keys()) >= {"tool_name", "tier", "allowed", "reason", "requires_approval"}
