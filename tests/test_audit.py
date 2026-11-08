import json
import pytest
from unittest.mock import patch
from src.policy.audit import write_audit_entry


def test_write_audit_entry_creates_file(tmp_path):
    log_file = str(tmp_path / "audit.jsonl")
    with patch("src.policy.audit.AUDIT_LOG_PATH", log_file):
        write_audit_entry(
            actor="test",
            tool_name="get_sensor_reading",
            inputs={},
            policy_decision={"tier": "READ_ONLY"},
            trace_id="t-001",
            outcome="ALLOWED",
        )
    with open(log_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["outcome"] == "ALLOWED"
    assert entry["tool_name"] == "get_sensor_reading"


def test_write_audit_entry_has_timestamp(tmp_path):
    log_file = str(tmp_path / "audit.jsonl")
    with patch("src.policy.audit.AUDIT_LOG_PATH", log_file):
        write_audit_entry(
            actor="test",
            tool_name="get_sensor_reading",
            inputs={},
            policy_decision={},
            trace_id="t-002",
            outcome="ALLOWED",
        )
    with open(log_file) as f:
        entry = json.loads(f.readline())
    assert "timestamp" in entry


def test_write_audit_entry_appends(tmp_path):
    log_file = str(tmp_path / "audit.jsonl")
    with patch("src.policy.audit.AUDIT_LOG_PATH", log_file):
        for i in range(3):
            write_audit_entry(
                actor="test",
                tool_name=f"tool_{i}",
                inputs={},
                policy_decision={},
                trace_id=f"t-{i:03d}",
                outcome="ALLOWED",
            )
    with open(log_file) as f:
        lines = f.readlines()
    assert len(lines) == 3
