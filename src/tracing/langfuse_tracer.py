"""
Langfuse tracer stub for SY1-Gov.
Wraps policy decisions as trace spans without requiring a live Langfuse connection.
"""

from typing import Any

from src.policy.engine import PolicyDecision


class LangfuseTracer:
    """
    Lightweight trace collector.
    In production, spans are flushed to the Langfuse API.
    In tests or offline mode, spans accumulate in memory.
    """

    def __init__(self, flush_live: bool = False) -> None:
        self._flush_live = flush_live
        self._spans: list[dict[str, Any]] = []

    def trace_decision(self, decision: PolicyDecision, inputs: dict[str, Any]) -> None:
        """Record a policy decision as a trace span."""
        span = {
            "tool_name": decision.tool_name,
            "tier": decision.tier.value,
            "allowed": decision.allowed,
            "requires_approval": decision.requires_approval,
            "reason": decision.reason,
            "inputs": inputs,
        }
        self._spans.append(span)
        if self._flush_live:
            self._flush(span)

    def _flush(self, span: dict[str, Any]) -> None:
        """No-op flush stub - override in integration tests."""

    def get_spans(self) -> list[dict[str, Any]]:
        """Return all recorded spans."""
        return list(self._spans)

    def clear(self) -> None:
        """Clear recorded spans (test helper)."""
        self._spans.clear()
