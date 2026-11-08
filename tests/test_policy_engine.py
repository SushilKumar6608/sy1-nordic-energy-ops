from src.policy.engine import PolicyEngine, PermissionTier


def test_read_tier_exists():
    assert PermissionTier.READ.value == "read"


def test_reversible_tier_exists():
    assert PermissionTier.REVERSIBLE.value == "reversible"


def test_irreversible_tier_exists():
    assert PermissionTier.IRREVERSIBLE.value == "irreversible"


def test_policy_engine_instantiates():
    engine = PolicyEngine()
    assert engine is not None


def test_read_tool_is_allowed():
    engine = PolicyEngine()
    decision = engine.evaluate("get_demand_forecast", {})
    assert decision.allowed is True
    assert decision.tier == PermissionTier.READ


def test_decision_has_reason():
    engine = PolicyEngine()
    decision = engine.evaluate("get_demand_forecast", {})
    assert isinstance(decision.reason, str)
    assert len(decision.reason) > 0


def test_decision_to_dict():
    engine = PolicyEngine()
    decision = engine.evaluate("get_demand_forecast", {})
    d = decision.to_dict()
    assert isinstance(d, dict)
    assert "tier" in d
    assert "reason" in d
