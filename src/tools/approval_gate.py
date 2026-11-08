"""
Approval gate for IRREVERSIBLE tool calls.
In production this would call an external approval service.
In this governed stub it tracks pending approvals in memory.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.policy.engine import PolicyDecision


@dataclass
class PendingApproval:
    approval_id: str
    tool_name: str
    inputs: dict[str, Any]
    decision: PolicyDecision
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


_pending: dict[str, PendingApproval] = {}


def request_approval(
    decision: PolicyDecision, inputs: dict[str, Any]
) -> PendingApproval:
    """Create a pending approval record for an IRREVERSIBLE tool call."""
    approval_id = str(uuid.uuid4())
    record = PendingApproval(
        approval_id=approval_id,
        tool_name=decision.tool_name,
        inputs=inputs,
        decision=decision,
    )
    _pending[approval_id] = record
    return record


def approve(approval_id: str) -> PendingApproval:
    """Mark a pending approval as approved. Raises KeyError if not found."""
    record = _pending[approval_id]
    record.approved = True
    return record


def get_pending() -> list[PendingApproval]:
    """Return all pending (not yet approved) approvals."""
    return [r for r in _pending.values() if not r.approved]


def clear_approvals() -> None:
    """Clear all pending approvals (test helper)."""
    _pending.clear()
