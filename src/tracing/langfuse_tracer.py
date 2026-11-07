"""
Optional Langfuse tracing integration for SY1-Gov.

Wraps the ToolExecutor's 8-step tool-call lifecycle with Langfuse spans so
every policy evaluation, approval gate check, and tool execution is traced
end-to-end.

Design goals:
- Completely optional: when LANGFUSE_SECRET_KEY is not set, all calls are
  no-ops. CI passes without any Langfuse credentials.
- Zero side-effects on import: the Langfuse SDK is imported lazily only when
  _tracing_enabled() returns True.
- Thread-safe: each call to get_tracer() returns the same singleton tracer
  for the process lifetime, initialised once.

Environment variables (all optional):
  LANGFUSE_SECRET_KEY  - Required to enable tracing
  LANGFUSE_PUBLIC_KEY  - Required to enable tracing
  LANGFUSE_HOST        - Defaults to https://cloud.langfuse.com

Usage:
    from src.tracing.langfuse_tracer import get_tracer

    tracer = get_tracer()

    with tracer.policy_span("get_demand_forecast", inputs) as span:
        decision = engine.evaluate(tool_name, inputs)
        span.set_outcome(decision)

    # No-op when Langfuse is not configured:
    with tracer.policy_span("get_demand_forecast", inputs):
        pass
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracing_enabled() -> bool:
    """Return True only when both required Langfuse keys are set."""
    return bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
        and os.environ.get("LANGFUSE_PUBLIC_KEY")
    )


# ---------------------------------------------------------------------------
# Span context objects
# ---------------------------------------------------------------------------

@dataclass
class PolicySpanContext:
    """
    Thin wrapper around a Langfuse span (or a no-op stub).

    Callers set the outcome after they have it; the context manager
    flushes/closes the underlying span on exit.
    """
    tool_name: str
    inputs: dict[str, Any]
    _span: Any = field(default=None, repr=False)   # Langfuse span or None
    _enabled: bool = field(default=False, repr=False)

    def set_outcome(self, decision: Any) -> None:
        """Record the PolicyDecision on the span (no-op when disabled)."""
        if not self._enabled or self._span is None:
            return
        try:
            self._span.update(
                output={
                    "allowed": decision.allowed,
                    "tier": decision.tier.value if hasattr(decision.tier, "value") else str(decision.tier),
                    "requires_approval": decision.requires_approval,
                    "reason": decision.reason,
                }
            )
        except Exception:
            pass  # Never let tracing break the main path

    def set_error(self, exc: Exception) -> None:
        """Record an exception on the span (no-op when disabled)."""
        if not self._enabled or self._span is None:
            return
        try:
            self._span.update(
                level="ERROR",
                status_message=str(exc),
            )
        except Exception:
            pass

    def _close(self) -> None:
        if not self._enabled or self._span is None:
            return
        try:
            self._span.end()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class SY1Tracer:
    """
    Wraps Langfuse tracing for the SY1-Gov governance layer.

    Instantiate via get_tracer() - do not construct directly.
    """

    def __init__(self, enabled: bool, langfuse_client: Any = None):
        self._enabled = enabled
        self._lf = langfuse_client   # Langfuse client instance or None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextmanager
    def policy_span(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Generator[PolicySpanContext, None, None]:
        """
        Context manager that wraps a single policy evaluation with a span.

        Example:
            with tracer.policy_span("set_alert_threshold", inputs) as span:
                decision = engine.evaluate(tool_name, inputs)
                span.set_outcome(decision)
        """
        ctx = PolicySpanContext(
            tool_name=tool_name,
            inputs=inputs,
            _enabled=self._enabled,
        )

        if self._enabled and self._lf is not None:
            try:
                generation_kwargs: dict[str, Any] = {
                    "name": f"policy_eval:{tool_name}",
                    "input": inputs,
                    "metadata": {"tool_name": tool_name},
                }
                if trace_id:
                    generation_kwargs["trace_id"] = trace_id
                ctx._span = self._lf.span(**generation_kwargs)
            except Exception:
                ctx._span = None  # Degrade gracefully

        try:
            yield ctx
        except Exception as exc:
            ctx.set_error(exc)
            raise
        finally:
            ctx._close()

    @contextmanager
    def trace(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager for a top-level Langfuse trace (e.g. one full
        agent turn).

        Yields the Langfuse trace object (or None when disabled).
        """
        if not self._enabled or self._lf is None:
            yield None
            return

        lf_trace = None
        try:
            lf_trace = self._lf.trace(
                name=name,
                metadata=metadata or {},
            )
        except Exception:
            lf_trace = None

        try:
            yield lf_trace
        finally:
            if lf_trace is not None:
                try:
                    lf_trace.update(metadata=metadata or {})
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush buffered events to Langfuse (no-op when disabled)."""
        if not self._enabled or self._lf is None:
            return
        try:
            self._lf.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_tracer_instance: Optional[SY1Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer() -> SY1Tracer:
    """
    Return the process-level SY1Tracer singleton.

    - When LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set, initialises
      a real Langfuse client and returns a tracing-enabled tracer.
    - Otherwise returns a no-op tracer; all spans and traces are silent.
    """
    global _tracer_instance

    if _tracer_instance is not None:
        return _tracer_instance

    with _tracer_lock:
        if _tracer_instance is not None:
            return _tracer_instance

        if not _tracing_enabled():
            _tracer_instance = SY1Tracer(enabled=False)
            return _tracer_instance

        try:
            from langfuse import Langfuse  # type: ignore

            lf = Langfuse(
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            _tracer_instance = SY1Tracer(enabled=True, langfuse_client=lf)
        except ImportError:
            # langfuse package not installed - silently degrade
            _tracer_instance = SY1Tracer(enabled=False)
        except Exception:
            # Any other init failure - silently degrade
            _tracer_instance = SY1Tracer(enabled=False)

        return _tracer_instance


def reset_tracer() -> None:
    """
    Reset the singleton. Used in tests to isolate tracer state between
    test cases without restarting the interpreter.
    """
    global _tracer_instance
    with _tracer_lock:
        _tracer_instance = None
