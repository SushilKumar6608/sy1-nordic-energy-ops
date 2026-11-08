from src.policy.engine import PolicyEngine, PermissionTier


def test_all_read_tools_are_allowed():
    engine = PolicyEngine()
    for tool in ["get_demand_forecast", "get_anomaly_summary", "get_grid_topology",
                 "get_node_status", "query_fault_ontology", "retrieve_rag_context"]:
        decision = engine.evaluate(tool, {})
        assert decision.allowed is True, f"{tool} should be allowed"
        assert decision.tier == PermissionTier.READ


def test_all_irreversible_tools_require_approval():
    engine = PolicyEngine()
    for tool in ["dispatch_maintenance_crew", "isolate_grid_segment"]:
        decision = engine.evaluate(tool, {})
        assert decision.requires_approval is True
        assert decision.allowed is False


def test_reversible_tools_do_not_require_approval():
    engine = PolicyEngine()
    decision = engine.evaluate("set_alert_threshold", {})
    assert decision.requires_approval is False
    assert decision.allowed is True
