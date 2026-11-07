# SY1-Gov: Architecture Overview

## Purpose

SY1-Gov is a production governance layer wrapped around the SY1 district heating operations agent. The base SY1 system is a six-agent LangGraph pipeline that monitors grid demand, localises faults with a GNN, and dispatches crews. SY1-Gov adds a policy enforcement shell so that no tool call can bypass human oversight controls, regardless of what the LLM decides.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  External callers                                               │
│  (Claude Desktop / other agents / operator dashboard)          │
│                        │  MCP (stdio)                          │
│                        ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  src/mcp/server.py  (GovernedMCPServer)                 │   │
│  │  - Exposes 8 governed tools via JSON-RPC over stdio     │   │
│  │  - Calls PolicyEngine before returning any result       │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  src/policy/engine.py  (PolicyEngine)                   │   │
│  │                                                         │   │
│  │  TOOL_PERMISSIONS registry (Python dict, no LLM)        │   │
│  │    READ        -> allowed, no audit                     │   │
│  │    REVERSIBLE  -> allowed, audit logged                 │   │
│  │    IRREVERSIBLE -> blocked, requires approval token     │   │
│  │                                                         │   │
│  │  kill_switch_active flag                                │   │
│  │    -> blocks ALL writes when True                       │   │
│  └────────────┬──────────────────────────┬─────────────────┘   │
│               │                          │                      │
│               ▼                          ▼                      │
│  ┌────────────────────┐    ┌─────────────────────────────────┐  │
│  │ src/policy/audit.py│    │ src/tools/approval_gate.py      │  │
│  │ Append-only JSONL  │    │ LangGraph interrupt node        │  │
│  │ SHA-256 hash chain │    │ Pauses graph execution on       │  │
│  │ Sensitive redaction│    │ IRREVERSIBLE; waits for human   │  │
│  │                    │    │ operator to supply token        │  │
│  └────────────────────┘    └─────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  src/tracing/langfuse_tracer.py  (SY1Tracer)            │   │
│  │  Optional: wraps every policy_span with Langfuse        │   │
│  │  No-op when LANGFUSE_SECRET_KEY is absent               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  src/evals/budget_guard.py  (BudgetGuard)               │   │
│  │  Per-session token + USD cap                            │   │
│  │  check_budget() / record_usage() called by supervisor   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### PolicyEngine (`src/policy/engine.py`)

The single source of truth for what the LLM is allowed to do. It is pure Python with no LLM calls and no network dependencies. It evaluates every tool call against a static registry (`TOOL_PERMISSIONS`) and returns a `PolicyDecision` with:

- `allowed` - whether the call may proceed
- `tier` - READ / REVERSIBLE / IRREVERSIBLE
- `requires_approval` - True only for IRREVERSIBLE tools
- `reason` - human-readable explanation

The kill switch (`kill_switch_active`) short-circuits all tier evaluation and blocks every write operation. It is activated by setting `kill_switch_active=True` on the engine instance, or by passing `SY1_KILL_SWITCH=1` as an environment variable to the MCP server.

**Fail-closed**: any tool name not in the registry raises `ValueError`. The caller is responsible for catching this and treating unknown tools as blocked.

### Audit Log (`src/policy/audit.py`)

An append-only JSONL file at `logs/audit.jsonl`. Each entry contains:

- Timestamp (ISO 8601 UTC)
- Actor, tool name, inputs
- Policy decision
- Outcome string
- SHA-256 hash of this entry
- `prev_hash` linking to the previous entry (genesis entry uses `"0" * 64`)

Sensitive fields (`approval_token`, `password`, `secret`, `api_key`) are redacted before writing. The chain can be verified offline with `verify_chain(log_path)`.

### Approval Gate (`src/tools/approval_gate.py`)

A LangGraph interrupt node wired into the graph with `interrupt_before=["approval_gate"]`. When the supervisor wants to invoke an IRREVERSIBLE tool:

1. It places a `PendingApproval` record in graph state and routes to `approval_gate`.
2. LangGraph pauses. The human operator sees the pending action and supplies a token via `Command(resume={"approval_token": "..."})`.
3. The gate validates the token (non-empty), logs `APPROVED`, and lets execution continue.

The gate never executes the tool itself - it only validates and logs.

### MCP Server (`src/mcp/server.py`)

Exposes the governed tools via the Model Context Protocol (JSON-RPC 2.0 over stdio). Supported methods: `initialize`, `tools/list`, `tools/call`, `ping`. Every `tools/call` goes through PolicyEngine. IRREVERSIBLE tools are always blocked at the MCP layer with a clear message directing the caller to the LangGraph approval gate.

Start with:
```
python -m src.mcp.server
```

Configure in Claude Desktop's `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "sy1-gov": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "D:\\Egna projekt\\sy1-nordic-energy-ops"
    }
  }
}
```

### Langfuse Tracer (`src/tracing/langfuse_tracer.py`)

An optional wrapper around every `engine.evaluate()` call. When `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are set, each policy evaluation becomes a named Langfuse span with inputs and outcome recorded. When credentials are absent, all calls are silent no-ops - no code changes needed. The package itself is not required; a missing `langfuse` install also degrades silently.

### Budget Guard (`src/evals/budget_guard.py`)

A per-session token and USD cap. The supervisor calls `guard.check_budget()` before each LLM call and `guard.record_usage(input_tokens, output_tokens, model)` after. Raises `BudgetExceeded` when either cap is reached. Defaults: 100 000 tokens / $1.00. Override via constructor args or `SY1_MAX_TOKENS` / `SY1_MAX_COST_USD` environment variables.

---

## Evaluation Layer

### Trajectory Evaluator (`src/evals/trajectory_eval.py`)

Offline harness that runs scripted tool-call sequences through PolicyEngine and scores pass/fail per step. No LLM or network needed. Used in CI to verify that policy decisions are stable across code changes.

### LLM-as-Judge (`src/evals/judge.py`)

Scores agent free-text responses 0-4 on safety and helpfulness using a calibrated rubric. Live tests are auto-skipped in CI when `ANTHROPIC_API_KEY` is absent (marked `@pytest.mark.live_api`).

### Promptfoo Injection Suite (`evals/promptfoo/injection_suite.yaml`)

25 adversarial YAML cases across 7 categories: tier bypass attempts, kill switch circumvention, unknown tool probing, SQL injection in tool names, oversized inputs. The Promptfoo provider (`evals/promptfoo/policy_provider.py`) calls PolicyEngine directly - fully offline.

---

## Data Flow: Normal Tool Call

```
Supervisor LLM
    |
    | tool_call("set_alert_threshold", {node_id: 5, threshold_mw: 150, ...})
    v
PolicyEngine.evaluate("set_alert_threshold", inputs)
    |-- tier = REVERSIBLE
    |-- allowed = True (not kill-switched)
    |-- requires_approval = False
    v
write_audit_entry(outcome="ALLOWED")
    v
Langfuse span closed (if enabled)
    v
ToolExecutor executes the tool
    v
Result returned to supervisor
```

## Data Flow: Irreversible Tool Call

```
Supervisor LLM
    |
    | tool_call("dispatch_maintenance_crew", {...})
    v
PolicyEngine.evaluate(...)
    |-- tier = IRREVERSIBLE
    |-- allowed = False
    |-- requires_approval = True
    v
write_audit_entry(outcome="PENDING_HUMAN_APPROVAL")
    v
LangGraph routes to approval_gate node
    v
graph.compile(interrupt_before=["approval_gate"])
    |-- PAUSED - waiting for human
    v
Human operator: graph.invoke(Command(resume={"approval_token": "tok-abc123"}))
    v
ApprovalGateNode validates token
    v
write_audit_entry(outcome="APPROVED")
    v
ToolExecutor executes the tool
```

---

## Test Coverage

| Test file                     | Tests | Offline |
|-------------------------------|-------|---------|
| test_policy_engine.py         | 20    | Yes     |
| test_policy_contracts.py      | 26    | Yes     |
| test_audit.py                 | 21    | Yes     |
| test_approval_gate.py         | 18    | Yes     |
| test_evals.py                 | 29+3  | 29/3 skipped |
| test_budget_guard.py          | 37    | Yes     |
| test_tracing.py               | 20    | Yes     |
| **Total**                     | **174** | **169 pass, 3 skip** |

All 169 offline tests pass in CI without any API keys or Langfuse credentials.
