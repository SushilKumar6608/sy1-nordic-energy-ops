"""
Policy engine for SY1-Gov.

This module is intentionally separate from all prompts and LLM calls.
The LLM cannot talk around this layer. Every tool call passes through here.

Permission tiers:
  READ       - no approval needed, no audit entry (just a trace)
  REVERSIBLE - logged, no human approval needed, can be retried safely
  IRREVERSIBLE - logged, requires human approval before execution
"""

from enum import Enum
from typing import Any


class PermissionTier(str, Enum):
    READ = "read"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


# Tool registry: maps tool name -> permission tier
TOOL_PERMISSIONS: dict[str, PermissionTier] = {
    # Read tools (Phase 1 stubs - expanded as tools are added)
    "get_demand_forecast":          PermissionTier.READ,
    "get_anomaly_summary":          PermissionTier.READ,
    "get_grid_topology":            PermissionTier.READ,
    "get_node_status":              PermissionTier.READ,
    "query_fault_ontology":         PermissionTier.READ,
    "retrieve_rag_context":         PermissionTier.READ,
    # Reversible write tools
    "set_alert_threshold":          PermissionTier.REVERSIBLE,
    "acknowledge_incident":         PermissionTier.REVERSIBLE,
    # Irreversible write tools
    "dispatch_maintenance_crew":    PermissionTier.IRREVERSIBLE,
    "isolate_grid_segment":         PermissionTier.IRREVERSIBLE,
}


class PolicyDecision:
    def __init__(
        self,
        tool_name: str,
        tier: PermissionTier,
        allowed: bool,
        reason: str,
        requires_approval: bool = False,
    ):
        self.tool_name = tool_name
        self.tier = tier
        self.allowed = allowed
        self.reason = reason
        self.requires_approval = requires_approval

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tier": self.tier.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
        }


class PolicyEngine:
    """
    Evaluates whether a tool call is permitted.
    Called by the ToolExecutor before every tool call.
    Never instantiated inside an agent prompt.
    """

    def __init__(self, kill_switch_active: bool = False):
        self._kill_switch = kill_switch_active

    def evaluate(self, tool_name: str, inputs: dict[str, Any]) -> PolicyDecision:
        """
        Returns a PolicyDecision for the requested tool call.
        Raises ValueError for unknown tools (fail closed).
        """
        if self._kill_switch:
            return PolicyDecision(
                tool_name=tool_name,
                tier=PermissionTier.READ,
                allowed=False,
                reason="Global kill switch is active. All write operations suspended.",
                requires_approval=False,
            )

        if tool_name not in TOOL_PERMISSIONS:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                "All tools must be registered in TOOL_PERMISSIONS before use."
            )

        tier = TOOL_PERMISSIONS[tool_name]

        if tier == PermissionTier.READ:
            return PolicyDecision(
                tool_name=tool_name,
                tier=tier,
                allowed=True,
                reason="Read-only tool. No approval required.",
                requires_approval=False,
            )

        if tier == PermissionTier.REVERSIBLE:
            return PolicyDecision(
                tool_name=tool_name,
                tier=tier,
                allowed=True,
                reason="Reversible write. Logged but no human approval required.",
                requires_approval=False,
            )

        # IRREVERSIBLE
        return PolicyDecision(
            tool_name=tool_name,
            tier=tier,
            allowed=False,  # blocked until human approval is logged
            reason=(
                f"Irreversible tool '{tool_name}' requires explicit human approval "
                "before execution. Routing to approval gate."
            ),
            requires_approval=True,
        )

    def activate_kill_switch(self) -> None:
        self._kill_switch = True

    def deactivate_kill_switch(self) -> None:
        self._kill_switch = False
