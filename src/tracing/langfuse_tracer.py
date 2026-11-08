"""
Langfuse tracer for SY1-Gov.

Records policy decisions, routing decisions, and tool outcomes
as structured trace spans. Works offline (spans stored in memory)
and optionally flushes to Langfuse when keys are present.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from src.policy.engine import PolicyDecision
from src.routing.model_router import RoutingDecision


def _langfuse_enabled() -> bool:
    """Return True if Langfuse keys are set in the environment."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )


@dataclass
class TraceSpan:
    span_id: str
    tool_name: str
    model: str
    tier: str
    allowed: bool
    requires_approval: bool
    input_complexity: str
    policy_reason: str
    routing_reason: str
    latency_ms: float = 0.0
    token_usage: int = 0
    outcome: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


class LangfuseTracer:
    """
    Collects trace spans per session.
    Flushes to Langfuse API when keys are present; otherwise in-memory only.
    """

    def __init__(self, session_id: str = "default") -> None:
        self._session_id = session_id
        self._spans: list[TraceSpan] = []
        self._span_counter = 0

    def _next_span_id(self) -> str:
        self._span_counter += 1
        return f"{self._session_id}-span-{self._span_counter:04d}"

    def record(
        self,
        policy: PolicyDecision,
        routing: RoutingDecision,
        token_usage: int = 0,
        latency_ms: float = 0.0,
        outcome: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Record a policy + routing decision as a trace span."""
        span = TraceSpan(
            span_id=self._next_span_id(),
            tool_name=policy.tool_name,
            model=routing.model,
            tier=policy.tier.value,
            allowed=policy.allowed,
            requires_approval=policy.requires_approval,
            input_complexity=routing.input_complexity,
            policy_reason=policy.reason,
            routing_reason=routing.reason,
            latency_ms=latency_ms,
            token_usage=token_usage,
            outcome=outcome,
            metadata=metadata or {},
        )
        self._spans.append(span)
        if _langfuse_enabled():
            self._flush_to_langfuse(span)
        return span

    def _flush_to_langfuse(self, span: TraceSpan) -> None:
        """No-op stub - real implementation calls Langfuse SDK."""

    def get_spans(self) -> list[TraceSpan]:
        """Return all recorded spans for this session."""
        return list(self._spans)

    def total_tokens(self) -> int:
        """Return total token usage across all spans."""
        return sum(s.token_usage for s in self._spans)

    def cost_by_model(self) -> dict[str, int]:
        """Return token usage grouped by model."""
        result: dict[str, int] = {}
        for span in self._spans:
            result[span.model] = result.get(span.model, 0) + span.token_usage
        return result

    def clear(self) -> None:
        """Clear all spans (test helper)."""
        self._spans.clear()
        self._span_counter = 0
