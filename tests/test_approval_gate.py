from src.tools.approval_gate import ApprovalGateNode, request_approval, PendingApproval


def test_approval_gate_no_pending_returns_unchanged():
    node = ApprovalGateNode()
    state = {"pending_approval": None, "approval_history": []}
    result = node(state)
    assert result["pending_approval"] is None


def test_request_approval_returns_pending_approval():
    pa = request_approval(
        tool_name="dispatch_maintenance_crew",
        inputs={"crew_id": "C1"},
        policy_reason="irreversible",
    )
    assert isinstance(pa, PendingApproval)
    assert pa.tool_name == "dispatch_maintenance_crew"
    assert pa.approved is False
    assert pa.denied is False
    assert pa.approval_token is None


def test_request_approval_with_trace_id():
    pa = request_approval(
        tool_name="dispatch_maintenance_crew",
        inputs={},
        policy_reason="irreversible",
        trace_id="trace-fixed-001",
    )
    assert pa.trace_id == "trace-fixed-001"


def test_request_approval_auto_generates_trace_id():
    pa = request_approval(
        tool_name="dispatch_maintenance_crew",
        inputs={},
        policy_reason="irreversible",
    )
    assert pa.trace_id is not None
    assert len(pa.trace_id) > 0
