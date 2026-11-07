"""
Integration tests for src/policy/audit.py

Covers:
- write_audit_entry creates a file and returns an audit_id
- Sensitive fields are redacted
- Chain integrity holds across multiple entries
- Tampering with an entry breaks chain verification
- get_entry_by_id retrieves the correct entry
- Empty/missing log handled gracefully
"""

import json
import pytest
from pathlib import Path

from src.policy.audit import (
    write_audit_entry,
    verify_chain_integrity,
    get_entry_by_id,
    AUDIT_LOG_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """
    Redirect AUDIT_LOG_PATH to a temp directory for each test.
    This ensures tests never write to the real audit_logs/ directory.
    """
    import src.policy.audit as audit_module
    log_path = tmp_path / "audit_logs" / "audit.jsonl"
    monkeypatch.setattr(audit_module, "AUDIT_LOG_PATH", log_path)
    yield log_path


def _make_entry(tool_name="get_demand_forecast", outcome="SUCCESS", **kwargs):
    """Helper: write one entry and return its audit_id."""
    return write_audit_entry(
        actor="test_agent",
        tool_name=tool_name,
        inputs=kwargs.get("inputs", {"city": "Stockholm"}),
        policy_decision={"tier": "read", "allowed": True},
        trace_id="trace-001",
        outcome=outcome,
        **{k: v for k, v in kwargs.items() if k != "inputs"},
    )


# ---------------------------------------------------------------------------
# Basic write
# ---------------------------------------------------------------------------

class TestWriteAuditEntry:
    def test_creates_log_file(self, isolated_audit_log):
        _make_entry()
        assert isolated_audit_log.exists()

    def test_returns_audit_id(self, isolated_audit_log):
        audit_id = _make_entry()
        assert isinstance(audit_id, str)
        assert len(audit_id) == 36  # UUID format

    def test_entry_is_valid_json(self, isolated_audit_log):
        _make_entry()
        lines = isolated_audit_log.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "audit_id" in entry

    def test_entry_has_required_fields(self, isolated_audit_log):
        _make_entry()
        entry = json.loads(isolated_audit_log.read_text().strip())
        required = {
            "audit_id", "timestamp", "actor", "tool_name",
            "inputs", "policy_decision", "trace_id",
            "outcome", "chain_hash", "prev_hash",
        }
        assert required.issubset(entry.keys())

    def test_multiple_entries_appended(self, isolated_audit_log):
        _make_entry()
        _make_entry()
        _make_entry()
        lines = isolated_audit_log.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_genesis_prev_hash_is_zeros(self, isolated_audit_log):
        _make_entry()
        entry = json.loads(isolated_audit_log.read_text().strip())
        assert entry["prev_hash"] == "0" * 64


# ---------------------------------------------------------------------------
# Sensitive field redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_approval_token_redacted(self, isolated_audit_log):
        _make_entry(
            tool_name="dispatch_maintenance_crew",
            inputs={"node_id": 1, "approval_token": "secret-token-123"},
            approval_token="secret-token-123",
        )
        entry = json.loads(isolated_audit_log.read_text().strip())
        assert entry["inputs"]["approval_token"] == "[REDACTED]"

    def test_non_sensitive_fields_not_redacted(self, isolated_audit_log):
        _make_entry(inputs={"city": "Stockholm", "node_id": 42})
        entry = json.loads(isolated_audit_log.read_text().strip())
        assert entry["inputs"]["city"] == "Stockholm"
        assert entry["inputs"]["node_id"] == 42

    def test_nested_sensitive_field_redacted(self, isolated_audit_log):
        _make_entry(inputs={"outer": {"api_key": "super-secret"}})
        entry = json.loads(isolated_audit_log.read_text().strip())
        assert entry["inputs"]["outer"]["api_key"] == "[REDACTED]"

    def test_approval_present_flag_set(self, isolated_audit_log):
        _make_entry(approval_token="some-token")
        entry = json.loads(isolated_audit_log.read_text().strip())
        assert entry["approval_present"] is True

    def test_approval_present_false_when_no_token(self, isolated_audit_log):
        _make_entry()
        entry = json.loads(isolated_audit_log.read_text().strip())
        assert entry["approval_present"] is False


# ---------------------------------------------------------------------------
# Chain integrity
# ---------------------------------------------------------------------------

class TestChainIntegrity:
    def test_empty_log_is_valid(self, isolated_audit_log):
        is_valid, errors = verify_chain_integrity()
        assert is_valid is True
        assert errors == []

    def test_single_entry_valid(self, isolated_audit_log):
        _make_entry()
        is_valid, errors = verify_chain_integrity()
        assert is_valid is True
        assert errors == []

    def test_multiple_entries_valid(self, isolated_audit_log):
        for _ in range(10):
            _make_entry()
        is_valid, errors = verify_chain_integrity()
        assert is_valid is True
        assert errors == []

    def test_tampered_entry_detected(self, isolated_audit_log):
        _make_entry()
        _make_entry()
        _make_entry()

        # Tamper with the first entry
        lines = isolated_audit_log.read_text().splitlines()
        first = json.loads(lines[0])
        first["outcome"] = "TAMPERED"
        lines[0] = json.dumps(first)
        isolated_audit_log.write_text("\n".join(lines) + "\n")

        is_valid, errors = verify_chain_integrity()
        assert is_valid is False
        assert len(errors) > 0

    def test_missing_entry_detected(self, isolated_audit_log):
        for _ in range(5):
            _make_entry()

        # Delete the second line (simulates a deleted entry)
        lines = isolated_audit_log.read_text().splitlines()
        del lines[1]
        isolated_audit_log.write_text("\n".join(lines) + "\n")

        is_valid, errors = verify_chain_integrity()
        assert is_valid is False

    def test_prev_hash_links_entries(self, isolated_audit_log):
        _make_entry()
        _make_entry()
        lines = isolated_audit_log.read_text().strip().splitlines()
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert second["prev_hash"] == first["chain_hash"]


# ---------------------------------------------------------------------------
# get_entry_by_id
# ---------------------------------------------------------------------------

class TestGetEntryById:
    def test_retrieves_correct_entry(self, isolated_audit_log):
        audit_id = _make_entry(tool_name="get_node_status")
        entry = get_entry_by_id(audit_id)
        assert entry is not None
        assert entry["audit_id"] == audit_id
        assert entry["tool_name"] == "get_node_status"

    def test_returns_none_for_missing_id(self, isolated_audit_log):
        _make_entry()
        result = get_entry_by_id("nonexistent-id")
        assert result is None

    def test_returns_none_when_no_log_exists(self, isolated_audit_log):
        # Log file doesn't exist yet
        result = get_entry_by_id("any-id")
        assert result is None

    def test_retrieves_from_multiple_entries(self, isolated_audit_log):
        ids = [_make_entry() for _ in range(5)]
        # Retrieve the third entry specifically
        target_id = ids[2]
        entry = get_entry_by_id(target_id)
        assert entry["audit_id"] == target_id
