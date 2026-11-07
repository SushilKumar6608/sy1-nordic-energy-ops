"""
Append-only audit log for SY1-Gov.

Each entry records: actor, tool, inputs (redacted), policy decision,
trace ID, outcome, and a SHA-256 chain hash linking it to the previous entry.

The hash chain means any tampering with a past entry breaks all subsequent hashes.
A single entry is sufficient to reconstruct the full action from request to outcome.

Log file: audit_logs/audit.jsonl  (gitignored except the directory)
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Fields that must be redacted before logging or sending to the LLM
_SENSITIVE_FIELDS = {
    "approval_token",
    "password",
    "api_key",
    "secret",
    "token",
    "credential",
}

AUDIT_LOG_PATH = Path("audit_logs/audit.jsonl")


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive fields."""
    out = {}
    for k, v in data.items():
        if k.lower() in _SENSITIVE_FIELDS:
            out[k] = "[REDACTED]"
        elif isinstance(v, dict):
            out[k] = _redact(v)
        else:
            out[k] = v
    return out


def _read_last_hash() -> str:
    """Return the chain hash of the last log entry, or genesis hash."""
    if not AUDIT_LOG_PATH.exists():
        return "0" * 64  # genesis
    with open(AUDIT_LOG_PATH, "rb") as f:
        lines = f.read().splitlines()
    if not lines:
        return "0" * 64
    last = json.loads(lines[-1])
    return last.get("chain_hash", "0" * 64)


def _compute_chain_hash(entry: dict[str, Any], prev_hash: str) -> str:
    content = json.dumps(entry, sort_keys=True, default=str) + prev_hash
    return hashlib.sha256(content.encode()).hexdigest()


def write_audit_entry(
    actor: str,
    tool_name: str,
    inputs: dict[str, Any],
    policy_decision: dict[str, Any],
    trace_id: str,
    outcome: str,
    approval_token: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    """
    Append one entry to the audit log.
    Returns the entry's audit_id.
    """
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit_id = str(uuid.uuid4())
    prev_hash = _read_last_hash()

    entry: dict[str, Any] = {
        "audit_id": audit_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "tool_name": tool_name,
        "inputs": _redact(inputs),
        "policy_decision": policy_decision,
        "trace_id": trace_id,
        "outcome": outcome,
        "idempotency_key": idempotency_key,
        "error": error,
        "approval_present": approval_token is not None,
    }

    entry["chain_hash"] = _compute_chain_hash(entry, prev_hash)
    entry["prev_hash"] = prev_hash

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return audit_id


def verify_chain_integrity() -> tuple[bool, list[str]]:
    """
    Walk the entire audit log and verify every hash chain link.
    Returns (is_valid, list_of_errors).
    """
    if not AUDIT_LOG_PATH.exists():
        return True, []

    errors = []
    prev_hash = "0" * 64

    with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            entry = json.loads(line.strip())
            stored_hash = entry.pop("chain_hash")
            claimed_prev = entry.pop("prev_hash")

            if claimed_prev != prev_hash:
                errors.append(
                    f"Line {line_no}: prev_hash mismatch. "
                    f"Expected {prev_hash[:8]}... got {claimed_prev[:8]}..."
                )

            recomputed = _compute_chain_hash(entry, claimed_prev)
            if recomputed != stored_hash:
                errors.append(
                    f"Line {line_no}: chain_hash mismatch on audit_id={entry.get('audit_id')}. "
                    "Entry may have been tampered with."
                )

            prev_hash = stored_hash

    return len(errors) == 0, errors


def get_entry_by_id(audit_id: str) -> Optional[dict[str, Any]]:
    """Retrieve a single audit entry by its audit_id."""
    if not AUDIT_LOG_PATH.exists():
        return None
    with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("audit_id") == audit_id:
                return entry
    return None
