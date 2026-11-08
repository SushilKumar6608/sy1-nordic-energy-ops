"""Audit log for policy decisions on write-tier tool calls."""

from datetime import datetime, timezone
from typing import Any

from src.policy.engine import PermissionTier, PolicyDecision

_audit_log: list[dict[str, Any]] = []


def write_audit_entry(decision: PolicyDecision, inputs: dict[str, Any]) -> None:
    """Append an audit entry for REVERSIBLE and IRREVERSIBLE decisions."""
    if decision.tier == PermissionTier.READ:
        return
    _audit_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": decision.tool_name,
            "tier": decision.tier.value,
            "allowed": decision.allowed,
            "requires_approval": decision.requires_approval,
            "reason": decision.reason,
            "inputs_snapshot": inputs,
        }
    )


def get_audit_log() -> list[dict[str, Any]]:
    """Return a copy of the in-memory audit log."""
    return list(_audit_log)


def clear_audit_log() -> None:
    """Clear the audit log (test helper)."""
    _audit_log.clear()
