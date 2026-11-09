"""
LangGraph approval gate for SY1-Gov.

This module wires the PolicyEngine's IRREVERSIBLE tier into the LangGraph
graph as an interrupt node. When the supervisor agent requests an irreversible
tool call, the graph pauses here, waits for a human operator to supply an
approval token, then resumes execution with that token passed to the
ToolExecutor.

Design:
- The gate never executes the tool itself. It only validates and logs the approval.
- The approval token is a UUID generated and stored in the audit log.
- After the human approves, the graph resumes and the ToolExecutor re-checks
  the token before executing.

Usage in a LangGraph graph:
    from src.tools.approval_gate import ApprovalGateNode, ApprovalGateState

    builder = StateGraph(ApprovalGateState)
    builder.add_node("approval_gate", ApprovalGateNode())
    builder.add_edge("supervisor", "approval_gate")
    builder.add_edge("approval_gate", "tool_executor_node")

    # Wire the interrupt:
    graph = builder.compile(interrupt_before=["approval_gate"])

    # Resume with human input:
    graph.invoke(
        Command(resume={"approval_token": "human-supplied-token"}),
        config={"configurable": {"thread_id": "..."}},
    )
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel

from src.policy.engine import PolicyEngine, PermissionTier
from src.policy.audit import write_audit_entry

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class PendingApproval(BaseModel):
    """Describes one pending irreversible tool call awaiting human approval."""

    tool_name: str
    inputs: dict[str, Any]
    policy_reason: str
    trace_id: str
    approval_token: Optional[str] = None  # filled in after human approves
    approved: bool = False
    denied: bool = False


class ApprovalGateState(BaseModel):
    """
    Minimal LangGraph state slice owned by the approval gate.
    Merge this into your full graph state with TypedDict or model inheritance.
    """

    pending_approval: Optional[PendingApproval] = None
    approval_history: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Approval gate node
# ---------------------------------------------------------------------------


class ApprovalGateNode:
    """
    LangGraph node that blocks irreversible tool calls for human approval.

    Instantiate once and add to the graph:
        builder.add_node("approval_gate", ApprovalGateNode(policy_engine))

    The graph must be compiled with:
        graph.compile(interrupt_before=["approval_gate"])

    so that LangGraph pauses execution at this node and waits for
    graph.invoke(Command(resume={...})) to supply the approval token.
    """

    def __init__(self, policy_engine: Optional[PolicyEngine] = None):
        self._policy = policy_engine or PolicyEngine()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Called by LangGraph when the graph reaches this node.

        On first entry (no approval token yet):
        - Validates the pending tool call against the policy engine
        - Logs a PENDING_APPROVAL audit entry
        - Returns state unchanged (LangGraph will interrupt here)

        On resume (approval token supplied via Command(resume=...)):
        - Validates the token is present
        - Logs an APPROVED or DENIED audit entry
        - Sets approved=True so the downstream executor node can proceed
        """
        pending: Optional[dict] = state.get("pending_approval")

        if pending is None:
            # No pending approval - nothing to do
            return state

        tool_name = pending.get("tool_name", "")
        inputs = pending.get("inputs", {})
        trace_id = pending.get("trace_id", str(uuid.uuid4()))
        approval_token = pending.get("approval_token")

        # Verify this tool is actually irreversible
        try:
            decision = self._policy.evaluate(tool_name, inputs)
        except ValueError as e:
            # Unknown tool - deny immediately
            write_audit_entry(
                actor="approval_gate",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision={"error": str(e)},
                trace_id=trace_id,
                outcome="DENIED_UNKNOWN_TOOL",
            )
            return {
                **state,
                "pending_approval": {
                    **pending,
                    "denied": True,
                    "policy_reason": str(e),
                },
            }

        if decision.tier != PermissionTier.IRREVERSIBLE:
            # Not irreversible - should not have come through the gate
            write_audit_entry(
                actor="approval_gate",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="GATE_SKIPPED_NOT_IRREVERSIBLE",
            )
            return {
                **state,
                "pending_approval": {
                    **pending,
                    "approved": True,
                    "policy_reason": "Not irreversible; gate not applicable.",
                },
            }

        if approval_token is None:
            # First entry: log pending, let LangGraph interrupt
            write_audit_entry(
                actor="approval_gate",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="PENDING_HUMAN_APPROVAL",
            )
            # State is returned as-is; LangGraph will pause here because
            # this node is listed in interrupt_before=["approval_gate"]
            return state

        # Resumed with token: validate it is non-empty
        if not approval_token.strip():
            write_audit_entry(
                actor="approval_gate",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="DENIED_EMPTY_TOKEN",
                approval_token=approval_token,
            )
            return {
                **state,
                "pending_approval": {
                    **pending,
                    "denied": True,
                    "policy_reason": "Approval token was empty or whitespace.",
                },
            }

        # Valid token supplied - log approval and allow execution
        write_audit_entry(
            actor="approval_gate",
            tool_name=tool_name,
            inputs=inputs,
            policy_decision=decision.to_dict(),
            trace_id=trace_id,
            outcome="APPROVED",
            approval_token=approval_token,
        )

        history_entry = {
            "tool_name": tool_name,
            "trace_id": trace_id,
            "approval_token": approval_token,
            "outcome": "APPROVED",
        }

        return {
            **state,
            "pending_approval": {
                **pending,
                "approved": True,
                "approval_token": approval_token,
            },
            "approval_history": state.get("approval_history", []) + [history_entry],
        }


# ---------------------------------------------------------------------------
# Helper: build a pending approval from a tool call
# ---------------------------------------------------------------------------


def request_approval(
    tool_name: str,
    inputs: dict[str, Any],
    policy_reason: str,
    trace_id: Optional[str] = None,
) -> PendingApproval:
    """
    Create a PendingApproval record to place in graph state before
    routing to the approval_gate node.

    Called by the supervisor agent when it wants to invoke an
    irreversible tool.

    Example:
        state["pending_approval"] = request_approval(
            tool_name="dispatch_maintenance_crew",
            inputs={...},
            policy_reason=decision.reason,
            trace_id=trace_id,
        ).model_dump()
        # Then route to "approval_gate" node
    """
    return PendingApproval(
        tool_name=tool_name,
        inputs=inputs,
        policy_reason=policy_reason,
        trace_id=trace_id or str(uuid.uuid4()),
    )
