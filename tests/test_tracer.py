"""Tests for the Langfuse tracer - all offline, no API key needed."""

from src.policy.engine import PermissionTier, PolicyDecision
from src.routing.model_router import MODEL_HAIKU, MODEL_SONNET, RoutingDecision
from src.tracing.langfuse_tracer import LangfuseTracer, TraceSpan


def _make_policy(tool: str, tier: PermissionTier, allowed: bool) -> PolicyDecision:
    return PolicyDecision(
        tool_name=tool,
        tier=tier,
        allowed=allowed,
        reason="test reason",
        requires_approval=not allowed and tier == PermissionTier.IRREVERSIBLE,
    )


def _make_routing(model: str, tier: PermissionTier) -> RoutingDecision:
    return RoutingDecision(
        model=model,
        tier=tier,
        reason="test routing",
        input_complexity="simple",
    )


def test_tracer_records_span():
    tracer = LangfuseTracer()
    policy = _make_policy("get_demand_forecast", PermissionTier.READ, True)
    routing = _make_routing(MODEL_HAIKU, PermissionTier.READ)
    span = tracer.record(policy, routing, token_usage=100)
    assert isinstance(span, TraceSpan)
    assert len(tracer.get_spans()) == 1


def test_span_has_correct_tool_name():
    tracer = LangfuseTracer()
    policy = _make_policy("get_demand_forecast", PermissionTier.READ, True)
    routing = _make_routing(MODEL_HAIKU, PermissionTier.READ)
    span = tracer.record(policy, routing)
    assert span.tool_name == "get_demand_forecast"


def test_span_has_correct_model():
    tracer = LangfuseTracer()
    policy = _make_policy(
        "dispatch_maintenance_crew", PermissionTier.IRREVERSIBLE, False
    )
    routing = _make_routing(MODEL_SONNET, PermissionTier.IRREVERSIBLE)
    span = tracer.record(policy, routing)
    assert span.model == MODEL_SONNET


def test_total_tokens_sums_correctly():
    tracer = LangfuseTracer()
    for tokens in [100, 200, 300]:
        policy = _make_policy("get_demand_forecast", PermissionTier.READ, True)
        routing = _make_routing(MODEL_HAIKU, PermissionTier.READ)
        tracer.record(policy, routing, token_usage=tokens)
    assert tracer.total_tokens() == 600


def test_cost_by_model_groups_correctly():
    tracer = LangfuseTracer()
    policy_read = _make_policy("get_demand_forecast", PermissionTier.READ, True)
    policy_irr = _make_policy(
        "dispatch_maintenance_crew", PermissionTier.IRREVERSIBLE, False
    )
    tracer.record(
        policy_read, _make_routing(MODEL_HAIKU, PermissionTier.READ), token_usage=100
    )
    tracer.record(
        policy_irr,
        _make_routing(MODEL_SONNET, PermissionTier.IRREVERSIBLE),
        token_usage=500,
    )
    costs = tracer.cost_by_model()
    assert costs[MODEL_HAIKU] == 100
    assert costs[MODEL_SONNET] == 500


def test_clear_resets_spans():
    tracer = LangfuseTracer()
    policy = _make_policy("get_demand_forecast", PermissionTier.READ, True)
    routing = _make_routing(MODEL_HAIKU, PermissionTier.READ)
    tracer.record(policy, routing)
    tracer.clear()
    assert len(tracer.get_spans()) == 0
    assert tracer.total_tokens() == 0


def test_span_id_increments():
    tracer = LangfuseTracer(session_id="test")
    policy = _make_policy("get_demand_forecast", PermissionTier.READ, True)
    routing = _make_routing(MODEL_HAIKU, PermissionTier.READ)
    s1 = tracer.record(policy, routing)
    s2 = tracer.record(policy, routing)
    assert s1.span_id != s2.span_id


def test_multiple_sessions_independent():
    t1 = LangfuseTracer(session_id="s1")
    t2 = LangfuseTracer(session_id="s2")
    policy = _make_policy("get_demand_forecast", PermissionTier.READ, True)
    routing = _make_routing(MODEL_HAIKU, PermissionTier.READ)
    t1.record(policy, routing, token_usage=100)
    assert t2.total_tokens() == 0
