"""
MCP server for SY1-Gov governed tools.

Exposes the district heating grid tools as an MCP (Model Context Protocol)
server so external clients (e.g. Claude Desktop, other agents) can call
them via the governed PolicyEngine layer.

Every tool call is:
1. Evaluated by PolicyEngine before execution
2. Blocked if tier is IRREVERSIBLE (requires human approval outside MCP)
3. Logged to the append-only audit log
4. Traced via Langfuse (when configured)

Design goals:
- No LangGraph dependency: the MCP server wires PolicyEngine directly
- stdio transport: works with any MCP client out of the box
- Graceful degradation: tracing is optional

Starting the server:
    python -m src.mcp.server

Or from another module:
    from src.mcp.server import create_server, run_server
    server = create_server()
    run_server(server)

Environment variables:
  LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY  - Optional Langfuse tracing
  SY1_KILL_SWITCH                            - Set to "1" to activate kill switch
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any

# Add project root to sys.path so src.* imports work when run as __main__
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.policy.engine import PolicyEngine, PermissionTier
from src.policy.audit import write_audit_entry
from src.tracing.langfuse_tracer import get_tracer


# ---------------------------------------------------------------------------
# Tool registry - mirrors the tools the supervisor agent uses
# Each tool has a name, description, and input JSON schema.
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_demand_forecast",
        "description": (
            "Retrieve the district heating demand forecast for a city. "
            "READ tier - always allowed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Stockholm'",
                },
                "target": {
                    "type": "string",
                    "description": "Forecast target, e.g. 'heating_demand_mw'",
                },
            },
            "required": ["city", "target"],
        },
    },
    {
        "name": "get_node_status",
        "description": (
            "Get the current operational status of a grid node. "
            "READ tier - always allowed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer", "description": "Grid node ID"},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_demand_timeseries",
        "description": (
            "Retrieve historical demand timeseries for a city and date range. "
            "READ tier - always allowed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "start_date": {
                    "type": "string",
                    "description": "ISO date, e.g. '2024-01-01'",
                },
                "end_date": {
                    "type": "string",
                    "description": "ISO date, e.g. '2024-01-31'",
                },
            },
            "required": ["city", "start_date", "end_date"],
        },
    },
    {
        "name": "set_alert_threshold",
        "description": (
            "Update the alert threshold for a grid node. "
            "REVERSIBLE tier - allowed; can be undone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer"},
                "threshold_mw": {"type": "number"},
                "idempotency_key": {
                    "type": "string",
                    "description": "UUID for deduplication",
                },
            },
            "required": ["node_id", "threshold_mw", "idempotency_key"],
        },
    },
    {
        "name": "update_valve_position",
        "description": (
            "Adjust valve position on a grid segment. "
            "REVERSIBLE tier - allowed; can be reset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_id": {"type": "integer"},
                "position_pct": {"type": "number", "description": "0-100 percent open"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["segment_id", "position_pct", "idempotency_key"],
        },
    },
    {
        "name": "dispatch_maintenance_crew",
        "description": (
            "Dispatch a maintenance crew to a node. "
            "IRREVERSIBLE tier - requires human approval; will be blocked via MCP."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer"},
                "fault_type": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "justification": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": [
                "node_id",
                "fault_type",
                "priority",
                "justification",
                "idempotency_key",
            ],
        },
    },
    {
        "name": "trigger_emergency_shutdown",
        "description": (
            "Trigger emergency shutdown of a grid segment. "
            "IRREVERSIBLE tier - requires human approval; will be blocked via MCP."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_id": {"type": "integer"},
                "reason": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["segment_id", "reason", "idempotency_key"],
        },
    },
    {
        "name": "issue_grid_alert",
        "description": (
            "Issue a public grid alert. "
            "IRREVERSIBLE tier - requires human approval; will be blocked via MCP."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                "message": {"type": "string"},
                "affected_cities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "idempotency_key": {"type": "string"},
            },
            "required": ["severity", "message", "affected_cities", "idempotency_key"],
        },
    },
]


# ---------------------------------------------------------------------------
# Policy-enforced tool call handler
# ---------------------------------------------------------------------------


def _handle_tool_call(
    tool_name: str,
    inputs: dict[str, Any],
    engine: PolicyEngine,
    tracer: Any,
    trace_id: str,
) -> dict[str, Any]:
    """
    Evaluate and (if permitted) execute one tool call through the
    PolicyEngine.

    Returns a dict with:
      - allowed (bool)
      - tier (str)
      - requires_approval (bool)
      - reason (str)
      - result (any, only when allowed and not irreversible)
    """
    with tracer.policy_span(tool_name, inputs, trace_id=trace_id) as span:
        try:
            decision = engine.evaluate(tool_name, inputs)
        except ValueError as exc:
            write_audit_entry(
                actor="mcp_server",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision={"error": str(exc)},
                trace_id=trace_id,
                outcome="DENIED_UNKNOWN_TOOL",
            )
            return {
                "allowed": False,
                "tier": "unknown",
                "requires_approval": False,
                "reason": str(exc),
            }

        span.set_outcome(decision)

        tier = decision.tier

        # Irreversible tools are never executed via MCP - they require
        # out-of-band human approval through the LangGraph approval gate.
        if tier == PermissionTier.IRREVERSIBLE:
            write_audit_entry(
                actor="mcp_server",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="BLOCKED_IRREVERSIBLE_REQUIRES_APPROVAL",
            )
            return {
                "allowed": False,
                "tier": tier.value,
                "requires_approval": True,
                "reason": (
                    f"Tool '{tool_name}' is IRREVERSIBLE and requires human approval. "
                    "Submit this action through the approval gate."
                ),
            }

        # Kill switch check is handled inside PolicyEngine.evaluate()
        if not decision.allowed:
            write_audit_entry(
                actor="mcp_server",
                tool_name=tool_name,
                inputs=inputs,
                policy_decision=decision.to_dict(),
                trace_id=trace_id,
                outcome="BLOCKED_POLICY",
            )
            return {
                "allowed": False,
                "tier": tier.value,
                "requires_approval": decision.requires_approval,
                "reason": decision.reason,
            }

        # Tool is allowed - return a stub result for READ/REVERSIBLE tiers.
        # In a full deployment this would delegate to the actual tool
        # implementation; here we return the policy decision so the caller
        # knows the action was permitted.
        write_audit_entry(
            actor="mcp_server",
            tool_name=tool_name,
            inputs=inputs,
            policy_decision=decision.to_dict(),
            trace_id=trace_id,
            outcome="ALLOWED",
        )
        return {
            "allowed": True,
            "tier": tier.value,
            "requires_approval": False,
            "reason": decision.reason,
            "result": {
                "status": "ok",
                "tool_name": tool_name,
                "message": f"Tool '{tool_name}' authorised by PolicyEngine.",
            },
        }


# ---------------------------------------------------------------------------
# MCP JSON-RPC message handlers
# ---------------------------------------------------------------------------


def _build_server_info() -> dict[str, Any]:
    return {
        "name": "sy1-gov",
        "version": "0.1.0",
        "description": (
            "SY1 Nordic Energy Ops - governed tool server. "
            "All tool calls are evaluated by the PolicyEngine before execution."
        ),
    }


def _handle_initialize(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": _build_server_info(),
    }


def _handle_tools_list() -> dict[str, Any]:
    return {"tools": TOOL_DEFINITIONS}


def _handle_tools_call(
    params: dict[str, Any],
    engine: PolicyEngine,
    tracer: Any,
) -> dict[str, Any]:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    trace_id = str(uuid.uuid4())

    outcome = _handle_tool_call(tool_name, arguments, engine, tracer, trace_id)

    if outcome.get("allowed"):
        result_text = json.dumps(outcome.get("result", {}))
        return {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
        }
    else:
        error_text = json.dumps(
            {
                "blocked": True,
                "tier": outcome.get("tier"),
                "requires_approval": outcome.get("requires_approval", False),
                "reason": outcome.get("reason"),
            }
        )
        return {
            "content": [{"type": "text", "text": error_text}],
            "isError": True,
        }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class GovernedMCPServer:
    """
    Minimal MCP server that serves the governed SY1 tools over stdio.

    Processes one JSON-RPC message at a time, writing responses to stdout.
    Errors go to stderr so they do not corrupt the stdio protocol stream.
    """

    def __init__(
        self,
        kill_switch_active: bool = False,
        policy_engine: PolicyEngine | None = None,
    ):
        ks = kill_switch_active or os.environ.get("SY1_KILL_SWITCH", "") == "1"
        self._engine = policy_engine or PolicyEngine(kill_switch_active=ks)
        self._tracer = get_tracer()
        self._next_id = 0

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """
        Process one JSON-RPC 2.0 message and return the response dict,
        or None for notifications (no response needed).
        """
        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})

        try:
            if method == "initialize":
                result = _handle_initialize(params)
            elif method == "notifications/initialized":
                return None  # Notification - no response
            elif method == "tools/list":
                result = _handle_tools_list()
            elif method == "tools/call":
                result = _handle_tools_call(params, self._engine, self._tracer)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(
                    msg_id, -32601, f"Method not found: {method}"
                )

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }

        except Exception as exc:
            print(f"[sy1-gov] Error handling {method}: {exc}", file=sys.stderr)
            return self._error_response(msg_id, -32603, str(exc))

    @staticmethod
    def _error_response(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

    def run_stdio(self) -> None:
        """
        Run the server in stdio mode, reading JSON-RPC messages from stdin
        line-by-line and writing responses to stdout.
        """
        print("[sy1-gov] MCP server starting (stdio transport)", file=sys.stderr)
        print(
            f"[sy1-gov] Kill switch: {self._engine.kill_switch_active}", file=sys.stderr
        )
        print(f"[sy1-gov] Langfuse tracing: {self._tracer.enabled}", file=sys.stderr)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                error = self._error_response(None, -32700, f"Parse error: {exc}")
                print(json.dumps(error), flush=True)
                continue

            response = self.handle_message(message)
            if response is not None:
                print(json.dumps(response), flush=True)

        self._tracer.flush()
        print("[sy1-gov] MCP server stopped", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public factory + entry point
# ---------------------------------------------------------------------------


def create_server(
    kill_switch_active: bool = False,
    policy_engine: PolicyEngine | None = None,
) -> GovernedMCPServer:
    """Create and return a GovernedMCPServer instance."""
    return GovernedMCPServer(
        kill_switch_active=kill_switch_active,
        policy_engine=policy_engine,
    )


def run_server(server: GovernedMCPServer | None = None) -> None:
    """Run the MCP server until stdin closes."""
    if server is None:
        server = create_server()
    server.run_stdio()


if __name__ == "__main__":
    run_server()
