from src.policy.engine import PolicyEngine


def test_activate_kill_switch():
    engine = PolicyEngine()
    engine.activate_kill_switch()
    decision = engine.evaluate("get_demand_forecast", {})
    assert decision.allowed is False


def test_kill_switch_reason_mentions_kill_switch():
    engine = PolicyEngine(kill_switch_active=True)
    decision = engine.evaluate("get_demand_forecast", {})
    assert "kill switch" in decision.reason.lower()


def test_decision_tool_name_in_dict():
    engine = PolicyEngine()
    decision = engine.evaluate("isolate_grid_segment", {})
    d = decision.to_dict()
    assert d["tool_name"] == "isolate_grid_segment"


def test_read_does_not_require_approval():
    engine = PolicyEngine()
    decision = engine.evaluate("retrieve_rag_context", {})
    assert decision.requires_approval is False
