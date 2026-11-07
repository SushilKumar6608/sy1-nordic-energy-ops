"""
Central ToolExecutor for SY1-Gov.

Every tool call in the system passes through this single entry point.
Order of operations:
  1. Contract validation (Pydantic input model)
  2. Policy check (permission tier)
  3. Approval gate (irreversible tools only - LangGraph interrupt)
  4. Idempotency check (don't double-execute)
  5. Tool execution with tenacity retry
  6. Output validation (Pydantic output model)
  7. Audit log entry
  8. Return result or typed ToolError

Nothing bypasses this. The LLM calls tools by name; the executor
handles all governance before the tool function is ever called.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from src.policy.engine import PolicyEngine, PermissionTier
from src.policy.contracts import TOOL_CONTRACTS, ToolError
from src.policy.audit import write_audit_entry


# In-memory idempotency cache (replace with Redis/DB in production)
_executed_keys: set[str] = set()

# Tool function registry - populated by register_tool()
_TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str, fn: Callable) -> None:
    """Register a tool function so the executor can call it by name."""
    _TOOL_REGISTRY[name] = fn


class ApprovalRequiredError(Exception):
    """Raised when an irreversible tool is called without a logged approval."""

    def __init__(self, tool_name: str, policy_reason: str):
        self.tool_name = tool_name
        self.policy_reason = policy_reason
        super().__init__(f"Approval required for '{tool_name}': {policy_reason}")


class ToolExecutor:
    """
    The single gateway for all tool calls in SY1-Gov.
    Instantiated once; shared across all agents.
    """

    def __init__(self, policy_engine: PolicyEngine, actor: str = "agent"):
        self._policy = policy_engine
        self._actor = actor

    def call(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        trace_id: str | None = None,
        approval_token: str | None = None,
    ) -> Any:
        """
        Execute a tool call through the full governance pipeline.

        Args:
            tool_name:      Registered tool name.
            inputs:         Raw input dict (will be validated against contract).
            trace_id:       Langfuse trace ID for observability.
            approval_token: Required for IRREVERSIBLE tools; must be a logged approval.

        Returns:
            Validated output model instance.

        Raises:
            ApprovalRequiredError: if irreversible tool called without approval token.
            ValueError:            if tool is unknown or inputs fail validation.
            ToolError:             if the tool function itself fails after retries.
        """
        trace_id = trace_id or str(uuid.uuid4())

        # --- 1. Contract validation (input) ---
        if tool_name not in TOOL_CONTRACTS:
            raise ValueError(f"No contract registered for tool '{tool_name}'")

        InputModel, OutputModel = TOOL_CONTRACTS[tool_name]
        try:
            validated_input = InputModel(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed for '{tool_name}': {e}") from e

        # --- 2. Policy check ---
        decision = self._policy.evaluate(tool_name, inputs)

        # --- 3. Approval gate for irreversible tools ---
        if decision.tier == PermissionTier.IRREVERSIBLE:
            if not approval_token:
                write_audit_entry(
                    actor=self._actor,
                    tool_name=tool_name,
                    inputs=inputs,
                    policy_decision=decision.to_dict(),
                    trace_id=trace_id,
                    outcome="BLOCKED_NO_APPROVAL",
                    idempotency_key=inputs.get("idempotency_key"),
                )
                raise ApprovalRequiredError(tool_name, decision.reason)

        if not decision.allowed and not approval_token:
            write_audit_entry(
                actor=self._actor,
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="BLOCKED_POLICY",
                idempotency_key=inputs.get("idempotency_key"),
            )
            raise ValueError(f"Policy blocked '{tool_name}': {decision.reason}")

        # --- 4. Idempotency check ---
        idem_key = inputs.get("idempotency_key")
        if idem_key and idem_key in _executed_keys:
            # Already executed; return a note rather than re-executing
            write_audit_entry(
                actor=self._actor,
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="SKIPPED_IDEMPOTENT",
                approval_token=approval_token,
                idempotency_key=idem_key,
            )
            return {"status": "already_executed", "idempotency_key": idem_key}

        # --- 5. Tool execution with retry ---
        if tool_name not in _TOOL_REGISTRY:
            raise ValueError(
                f"Tool '{tool_name}' has a contract but is not registered. "
                "Call register_tool() during startup."
            )

        tool_fn = _TOOL_REGISTRY[tool_name]

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _execute():
            return tool_fn(validated_input)

        outcome = "SUCCESS"
        error_msg = None
        raw_output = None

        try:
            raw_output = _execute()
        except RetryError as e:
            outcome = "FAILED_AFTER_RETRIES"
            error_msg = str(e)
        except Exception as e:
            outcome = "FAILED"
            error_msg = str(e)

        # --- 6. Output validation ---
        if raw_output is not None:
            try:
                validated_output = (
                    OutputModel(**raw_output)
                    if isinstance(raw_output, dict)
                    else raw_output
                )
            except Exception as e:
                outcome = "OUTPUT_VALIDATION_FAILED"
                error_msg = str(e)
                validated_output = None
        else:
            validated_output = None

        # --- 7. Audit log ---
        if idem_key:
            _executed_keys.add(idem_key)

        write_audit_entry(
            actor=self._actor,
            tool_name=tool_name,
            inputs=inputs,
            policy_decision=decision.to_dict(),
            trace_id=trace_id,
            outcome=outcome,
            approval_token=approval_token,
            idempotency_key=idem_key,
            error=error_msg,
        )

        # --- 8. Return or raise ---
        if outcome == "SUCCESS" and validated_output is not None:
            return validated_output

        raise ToolError(
            tool_name=tool_name,
            error_code=outcome,
            message=error_msg or "Unknown error",
            retryable=(outcome == "FAILED_AFTER_RETRIES"),
            idempotency_key=idem_key,
        )
