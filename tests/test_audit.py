from src.tools.approval_gate import request_approval, PendingApproval


def test_pending_approval_is_not_approved_by_default():
    pa = request_approval(
        tool_name="dispatch_maintenance_crew",
        inputs={},
        policy_reason="irreversible",
    )
    assert pa.approved is False
    assert pa.denied is False


def test_pending_approval_token_is_none_by_default():
    pa = request_approval(
        tool_name="dispatch_maintenance_crew",
        inputs={},
        policy_reason="irreversible",
    )
    assert pa.approval_token is None


def test_pending_approval_stores_tool_name():
    pa = request_approval(
        tool_name="isolate_grid_segment",
        inputs={"segment": "S3"},
        policy_reason="irreversible",
    )
    assert pa.tool_name == "isolate_grid_segment"
    assert pa.inputs == {"segment": "S3"}
