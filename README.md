# SY1-Gov: Governed District Heating Operations Agent

A production governance layer for the SY1 district heating operations agent. SY1-Gov wraps a six-agent LangGraph pipeline with deterministic policy enforcement, human-in-the-loop approval gates, a tamper-evident audit log, and a full evaluation suite - all offline-testable without API keys.

---

## What it does

District heating grids need automated monitoring and fault response. The base SY1 system uses an LLM supervisor to coordinate demand forecasting, GNN-based fault localisation, and crew dispatch. SY1-Gov adds a governance shell so the LLM can never execute irreversible actions (crew dispatch, grid isolation, public alerts) without a human operator supplying an explicit approval token.

**Core guarantees:**

- READ tools execute freely, with no audit overhead
- REVERSIBLE tools (threshold updates, alert configs) execute with full audit logging
- IRREVERSIBLE tools (crew dispatch, grid isolation, public alerts) are blocked until a human operator approves via LangGraph interrupt
- Kill switch blocks all writes instantly, with a single environment variable
- Policy enforcement is pure Python - no LLM calls, no network, no prompt to bypass

---

## Architecture

```
External callers (MCP / operator dashboard)
    |  JSON-RPC 2.0 over stdio
    v
GovernedMCPServer  (src/mcp/server.py)
    |
    v
PolicyEngine  (src/policy/engine.py)
    |-- READ        -> allowed, no audit
    |-- REVERSIBLE  -> allowed + audit entry written
    |-- IRREVERSIBLE -> blocked, routed to approval gate
    |
    +---> AuditLog  (src/policy/audit.py)
    |     SHA-256 hash-chained JSONL, sensitive fields redacted
    |
    +---> ApprovalGateNode  (src/tools/approval_gate.py)
          LangGraph interrupt_before node
          Waits for: graph.invoke(Command(resume={"approval_token": "..."}))
```

See [docs/architecture.md](docs/architecture.md) for the full component diagram and data-flow walkthrough.

---

## Project structure

```
sy1-nordic-energy-ops/
|-- src/
|   |-- policy/
|   |   |-- engine.py          # PolicyEngine: tier registry + kill switch
|   |   |-- contracts.py       # Pydantic input validators
|   |   `-- audit.py           # Append-only hash-chained audit log
|   |-- tools/
|   |   `-- approval_gate.py   # LangGraph interrupt approval gate
|   |-- evals/
|   |   |-- trajectory_eval.py # Offline trajectory scoring harness
|   |   |-- judge.py           # LLM-as-judge (skipped in CI without key)
|   |   `-- budget_guard.py    # Per-session token/USD cap
|   |-- tracing/
|   |   `-- langfuse_tracer.py # Optional Langfuse wrapper (no-op if absent)
|   `-- mcp/
|       `-- server.py          # JSON-RPC MCP server over stdio
|-- evals/
|   `-- promptfoo/
|       |-- injection_suite.yaml  # 25 adversarial test cases
|       `-- policy_provider.py    # Offline promptfoo provider
|-- tests/                     # 169 offline tests, 3 live-API tests
|-- docs/
|   |-- architecture.md
|   |-- threat_model.md
|   `-- adr/                   # Architecture Decision Records
|-- logs/
|   `-- audit.jsonl            # Runtime audit log (gitignored)
`-- pyproject.toml
```

---

## Setup

**Requirements:** Python 3.11+, Anaconda (Windows) or venv (Linux/macOS)

```
conda create -p D:\conda-envs\sy1-gov python=3.11
conda activate D:\conda-envs\sy1-gov
pip install -e ".[dev]" --break-system-packages
```

**Environment variables (all optional):**

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Enables live LLM-as-judge tests | (live tests skipped) |
| `LANGFUSE_SECRET_KEY` | Enables Langfuse tracing | (tracing disabled) |
| `LANGFUSE_PUBLIC_KEY` | Enables Langfuse tracing | (tracing disabled) |
| `SY1_KILL_SWITCH` | Set to `1` to block all writes | `0` |
| `SY1_MAX_TOKENS` | Per-session token cap | `100000` |
| `SY1_MAX_COST_USD` | Per-session cost cap | `1.00` |

---

## Running the tests

All 169 offline tests pass without any API keys:

```
pytest tests/ -v
```

Expected output:

```
tests/test_policy_engine.py        20 passed
tests/test_policy_contracts.py     26 passed
tests/test_audit.py                21 passed
tests/test_approval_gate.py        18 passed
tests/test_evals.py                29 passed, 3 skipped
tests/test_budget_guard.py         37 passed
tests/test_tracing.py              20 passed

169 passed, 3 skipped in X.XXs
```

The 3 skipped tests require `ANTHROPIC_API_KEY` and are marked `@pytest.mark.live_api`. They score agent responses using LLM-as-judge and are excluded from CI.

---

## Running the MCP server

```
python -m src.mcp.server
```

The server reads JSON-RPC 2.0 from stdin and writes responses to stdout. Add it to Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sy1-gov": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "D:\\Egna projekt\\sy1-nordic-energy-ops",
      "env": {
        "SY1_KILL_SWITCH": "0"
      }
    }
  }
}
```

To activate the kill switch without restarting the server, set `SY1_KILL_SWITCH=1` and restart the process. All REVERSIBLE and IRREVERSIBLE tool calls will return an error until it is cleared.

---

## Demo: policy decisions in action

**READ tool - passes through immediately:**

```
$ echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_demand_forecast","arguments":{"node_id":3,"horizon_hours":24}}}' | python -m src.mcp.server

{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Tool get_demand_forecast executed successfully"}], "isError": false}}
```

**IRREVERSIBLE tool - blocked, approval required:**

```
$ echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"dispatch_maintenance_crew","arguments":{"crew_id":"C-04","location":"node_7","justification":"GNN flagged 3-sigma anomaly"}}}' | python -m src.mcp.server

{"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "IRREVERSIBLE action requires human approval. Use the LangGraph approval gate to supply an approval token."}], "isError": true}}
```

**Kill switch active - all writes blocked:**

```
$ SY1_KILL_SWITCH=1 echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"set_alert_threshold","arguments":{"node_id":5,"threshold_mw":150,"alert_type":"overload"}}}' | python -m src.mcp.server

{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "Kill switch is active. All write operations are blocked."}], "isError": true}}
```

---

## Approval gate walkthrough

When an IRREVERSIBLE action is detected, the LangGraph supervisor routes to `approval_gate`. The graph pauses. The operator resumes it with a token:

```python
from langgraph.types import Command
from src.tools.approval_gate import build_approval_graph

graph = build_approval_graph()
config = {"configurable": {"thread_id": "session-001"}}

# Supervisor has requested dispatch_maintenance_crew - graph is now paused
# Operator reviews the pending action from the checkpoint store, then:

result = graph.invoke(
    Command(resume={"approval_token": "tok-operator-abc123"}),
    config=config
)
# Audit entry: outcome="APPROVED", actor="operator", approval_token_present=True
```

---

## Evaluation suite

**Trajectory evaluator** - offline, no API key needed:

```
python -m src.evals.trajectory_eval
```

Runs scripted tool-call sequences through PolicyEngine and reports pass/fail per step.

**Promptfoo injection suite** - offline, 25 adversarial cases:

```
promptfoo eval --config evals/promptfoo/injection_suite.yaml
```

**LLM-as-judge** - requires `ANTHROPIC_API_KEY`:

```
pytest tests/test_evals.py -m live_api -v
```

---

## Security

See [docs/threat_model.md](docs/threat_model.md) for the full threat catalogue. Key properties:

- Prompt injection has no effect on tier decisions (pure Python dict lookup)
- Kill switch cannot be deactivated via tool input fields
- Unknown tools fail closed with `ValueError`
- Approval tokens are redacted in the audit log
- The audit log is SHA-256 hash-chained; tampering is detectable offline

**Known gap:** approval tokens are not currently bound to a specific `trace_id`. Token binding is on the roadmap (see threat model T5).

---

## Architecture decisions

| ADR | Decision |
|-----|---------|
| [ADR-001](docs/adr/001-python-only-policy-engine.md) | Policy enforcement in pure Python, never in a prompt |
| [ADR-002](docs/adr/002-langgraph-interrupt-approval-gate.md) | LangGraph interrupt node as the approval gate |
| [ADR-003](docs/adr/003-append-only-hash-chained-audit-log.md) | Append-only SHA-256 hash-chained audit log |
