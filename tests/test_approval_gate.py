"""
Unit tests for src/tools/approval_gate.py

Covers:
- No pending approval - passthrough
- First entry with irreversible tool - logs PENDING, returns state unchanged
- Resume with valid token - logs APPROVED, sets approved=True
- Resume with empty token - logs DENIED, sets denied=True
- Unknown tool - logs DENIED_UNKNOWN_TOOL
- Non-irreversible tool routed through gate - gate skips and approves
- request_approval() helper
- approval_history accumulates
"""

import pytest

import src.policy.audit as audit_module
from src.tools.approval_gate import ApprovalGateNode, request_approval
from src.policy.engine import PolicyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    log_path = tmp_path / "audit_logs" / "audit.jsonl"
    monkeypatch.setattr(audit_module, "AUDIT_LOG_PATH", log_path)
    yield log_path


@pytest.fixture
def gate():
    return ApprovalGateNode(PolicyEngine())


def _irreversible_pending(token=None, approved=False, denied=False):
    return {
        "tool_name": "dispatch_maintenance_crew",
        "inputs": {
            "node_id": 5,
            "fault_type": "pipe_burst",
            "priority": "high",
            "justification": "GNN confidence 0.94. Pressure drop confirmed at node 5.",
            "idempotency_key": "uuid-test-001",
        },
        "policy_reason": "Irreversible tool requires approval.",
        "trace_id": "trace-gate-test",
        "approval_token": token,
        "approved": approved,
        "denied": denied,
    }


# ---------------------------------------------------------------------------
# No pending approval
# ---------------------------------------------------------------------------


class TestNoPendingApproval:
    def test_passthrough_when_no_pending(self, gate):
        state = {"some_key": "some_value"}
        result = gate(state)
        assert result == state

    def test_none_pending_approval(self, gate):
        state = {"pending_approval": None}
        result = gate(state)
        assert result == state


# ---------------------------------------------------------------------------
# First entry - no token yet
# ---------------------------------------------------------------------------


class TestFirstEntry:
    def test_state_unchanged_on_first_entry(self, gate, isolated_audit_log):
        state = {"pending_approval": _irreversible_pending()}
        result = gate(state)
        # State is returned as-is; approved stays False
        assert result["pending_approval"]["approved"] is False
        assert result["pending_approval"]["denied"] is False

    def test_pending_audit_entry_written(self, gate, isolated_audit_log):
        state = {"pending_approval": _irreversible_pending()}
        gate(state)
        import json

        lines = isolated_audit_log.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["outcome"] == "PENDING_HUMAN_APPROVAL"
        assert entry["tool_name"] == "dispatch_maintenance_crew"


# ---------------------------------------------------------------------------
# Resume with valid token
# ---------------------------------------------------------------------------


class TestResumeWithValidToken:
    def test_approved_flag_set(self, gate):
        state = {"pending_approval": _irreversible_pending(token="operator-token-abc")}
        result = gate(state)
        assert result["pending_approval"]["approved"] is True
        assert result["pending_approval"]["denied"] is False

    def test_approval_token_preserved(self, gate):
        state = {"pending_approval": _irreversible_pending(token="operator-token-abc")}
        result = gate(state)
        assert result["pending_approval"]["approval_token"] == "operator-token-abc"

    def test_approved_audit_entry_written(self, gate, isolated_audit_log):
        state = {"pending_approval": _irreversible_pending(token="operator-token-abc")}
        gate(state)
        import json

        lines = isolated_audit_log.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["outcome"] == "APPROVED"

    def test_approval_history_updated(self, gate):
        state = {
            "pending_approval": _irreversible_pending(token="tok-xyz"),
            "approval_history": [],
        }
        result = gate(state)
        assert len(result["approval_history"]) == 1
        assert result["approval_history"][0]["tool_name"] == "dispatch_maintenance_crew"
        assert result["approval_history"][0]["outcome"] == "APPROVED"

    def test_approval_history_accumulates(self, gate):
        state = {
            "pending_approval": _irreversible_pending(token="tok-1"),
            "approval_history": [
                {"tool_name": "isolate_grid_segment", "outcome": "APPROVED"}
            ],
        }
        result = gate(state)
        assert len(result["approval_history"]) == 2


# ---------------------------------------------------------------------------
# Resume with empty token
# ---------------------------------------------------------------------------


class TestEmptyToken:
    def test_denied_on_empty_token(self, gate):
        state = {"pending_approval": _irreversible_pending(token="")}
        result = gate(state)
        assert result["pending_approval"]["denied"] is True
        assert result["pending_approval"]["approved"] is False

    def test_denied_on_whitespace_token(self, gate):
        state = {"pending_approval": _irreversible_pending(token="   ")}
        result = gate(state)
        assert result["pending_approval"]["denied"] is True

    def test_denied_audit_entry_written(self, gate, isolated_audit_log):
        state = {"pending_approval": _irreversible_pending(token="")}
        gate(state)
        import json

        lines = isolated_audit_log.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["outcome"] == "DENIED_EMPTY_TOKEN"


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_denied_for_unknown_tool(self, gate):
        state = {
            "pending_approval": {
                "tool_name": "hack_the_grid",
                "inputs": {},
                "policy_reason": "unknown",
                "trace_id": "trace-unknown",
                "approval_token": "some-token",
                "approved": False,
                "denied": False,
            }
        }
        result = gate(state)
        assert result["pending_approval"]["denied"] is True

    def test_unknown_tool_audit_outcome(self, gate, isolated_audit_log):
        state = {
            "pending_approval": {
                "tool_name": "hack_the_grid",
                "inputs": {},
                "policy_reason": "unknown",
                "trace_id": "trace-unknown",
                "approval_token": "some-token",
                "approved": False,
                "denied": False,
            }
        }
        gate(state)
        import json

        lines = isolated_audit_log.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["outcome"] == "DENIED_UNKNOWN_TOOL"


# ---------------------------------------------------------------------------
# Non-irreversible tool routed through gate
# ---------------------------------------------------------------------------


class TestNonIrreversibleTool:
    def test_gate_skips_for_read_tool(self, gate):
        state = {
            "pending_approval": {
                "tool_name": "get_demand_forecast",
                "inputs": {"city": "Oslo", "target": "heating_demand_mw"},
                "policy_reason": "Should not be here",
                "trace_id": "trace-read",
                "approval_token": None,
                "approved": False,
                "denied": False,
            }
        }
        result = gate(state)
        # Gate should approve it since it's not irreversible
        assert result["pending_approval"]["approved"] is True


# ---------------------------------------------------------------------------
# request_approval helper
# ---------------------------------------------------------------------------


class TestRequestApproval:
    def test_creates_pending_approval(self):
        pa = request_approval(
            tool_name="dispatch_maintenance_crew",
            inputs={"node_id": 1},
            policy_reason="Requires approval.",
        )
        assert pa.tool_name == "dispatch_maintenance_crew"
        assert pa.approved is False
        assert pa.denied is False
        assert pa.approval_token is None

    def test_trace_id_auto_generated(self):
        pa = request_approval("dispatch_maintenance_crew", {}, "reason")
        assert pa.trace_id is not None
        assert len(pa.trace_id) == 36  # UUID

    def test_custom_trace_id_preserved(self):
        pa = request_approval(
            "dispatch_maintenance_crew", {}, "reason", trace_id="custom-trace"
        )
        assert pa.trace_id == "custom-trace"
