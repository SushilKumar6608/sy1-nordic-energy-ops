# SY1-Gov: Governed District Heating Operations Agent

A governance layer for the SY1 district heating operations system (a six-agent LangGraph pipeline). It provides a deterministic policy engine, an approval gate for irreversible actions, a hash-chained audit log, an MCP server that exposes policy decisions, and an offline evaluation suite. Everything in the test suite runs without API keys.

**Status:** the governance layer is a working, tested library. It is reachable through the MCP server and the LangGraph approval gate node. It is not yet called from the agent loop itself (`src/agents/energy_ops_agent.py`); see [Known limitations](#known-limitations).

---

## What it does

District heating grids need automated monitoring and fault response. The base SY1 system uses an LLM supervisor to coordinate demand forecasting, GNN-based fault localisation and crew dispatch. SY1-Gov decides, in plain Python, whether a requested tool call may run.

**Policy tiers** (`src/policy/engine.py`, ten registered tools):

- READ tools (demand forecast, anomaly summary, grid topology, node status, fault ontology, RAG retrieval) are allowed.
- REVERSIBLE tools (`set_alert_threshold`, `acknowledge_incident`) are allowed without human approval.
- IRREVERSIBLE tools (`dispatch_maintenance_crew`, `isolate_grid_segment`) are blocked until an approval token is supplied through the approval gate.
- Unknown tools raise `ValueError` (fail closed).
- The kill switch (`SY1_KILL_SWITCH=1`) blocks every tool call, reads included.
- Policy decisions are a dictionary lookup in pure Python: no LLM call, no network, no prompt to bypass.

---

## Architecture

```
External callers (MCP clients)
    |  JSON-RPC 2.0 over stdio
    v
GovernedMCPServer  (src/mcp/server.py)       authorises tool calls, does not run them
    |
    v
PolicyEngine  (src/policy/engine.py)
    |-- READ         -> allowed
    |-- REVERSIBLE   -> allowed
    |-- IRREVERSIBLE -> blocked, approval required
    |-- kill switch  -> everything blocked

LangGraph graph (supervisor -> approval_gate -> tool executor)
    |
    +---> ApprovalGateNode  (src/tools/approval_gate.py)
    |       compiled with interrupt_before=["approval_gate"]
    |       checks the policy tier, validates the token, writes audit entries
    |
    +---> AuditLog  (src/policy/audit.py)
    |       SHA-256 hash-chained JSONL, sensitive fields redacted
    |
    +---> ToolExecutor  (src/tools/executor.py)
            input validation, retries, idempotency keys, typed errors
```

See [docs/architecture.md](docs/architecture.md) for the component diagram and data flows.

---

## Project structure

```
src/
|-- policy/      engine.py (tiers, kill switch), contracts.py (Pydantic models), audit.py
|-- tools/       approval_gate.py (LangGraph node), executor.py
|-- mcp/         server.py (JSON-RPC MCP server over stdio)
|-- evals/       trajectory_eval.py, judge.py (LLM-as-judge), budget_guard.py (token and USD caps)
|-- tracing/     langfuse_tracer.py (no-op when Langfuse keys are absent)
|-- routing/     model_router.py (model tiers and cost estimates)
|-- knowledge/   ontology_query.py, fault_ontology.ttl
|-- agents/      energy_ops_agent.py (the SY1 supervisor pipeline)
|-- anomaly/  forecasting/  gnn/  ingestion/  rag/  api/  dashboard/   (SY1 pipeline components)
evals/promptfoo/ injection_suite.yaml, policy_provider.py
tests/           offline test suite
docs/            architecture.md, threat_model.md, adr/
```

---

## Setup

Python 3.11.

```
conda create -n sy1-gov python=3.11
conda activate sy1-gov
pip install -r requirements-ci.txt
```

**Environment variables (all optional):**

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Enables the live LLM-as-judge tests | live tests skipped |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Enables Langfuse tracing | tracing disabled |
| `SY1_KILL_SWITCH` | Set to `1` to block every tool call | `0` |
| `SY1_MAX_COST_USD` | Per-session cost cap in USD | `1.00` |

---

## Running the tests

```
pytest tests/ -m "not live_api"
```

177 offline tests pass without any API key. Three more tests call the Anthropic API (LLM-as-judge) and are marked `live_api`; CI deselects them. Coverage of the governance packages (`policy`, `tools`, `evals`, `mcp`, `tracing`) is about 74%, and CI enforces a 70% floor on that scope. The weak spots are `src/mcp/server.py` (not covered by unit tests; the demos below were run by hand) and `src/tools/executor.py` (about 28%; the retry and idempotency paths are untested).

The approval gate is tested end to end inside a real LangGraph graph (`tests/test_approval_gate_e2e.py`): the graph pauses before the gate, resuming without a token does not approve, resuming with a token does, and the token is redacted in the audit log.

---

## Running the MCP server

```
python -m src.mcp.server
```

The server reads JSON-RPC 2.0 from stdin and writes responses to stdout (banner lines go to stderr). Example Claude Desktop configuration:

```json
{
  "mcpServers": {
    "sy1-gov": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/sy1-nordic-energy-ops",
      "env": { "SY1_KILL_SWITCH": "0" }
    }
  }
}
```

The kill switch is read when the server starts. To activate it, set `SY1_KILL_SWITCH=1` and restart the process.

---

## Demo: policy decisions

Real output (stderr banners omitted). Linux and macOS:

**READ tool, allowed:**

```
$ echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_demand_forecast","arguments":{"node_id":3,"horizon_hours":24}}}' | python -m src.mcp.server
{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "{\"status\": \"ok\", \"tool_name\": \"get_demand_forecast\", \"message\": \"Tool 'get_demand_forecast' authorised by PolicyEngine.\"}"}], "isError": false}}
```

**IRREVERSIBLE tool, blocked pending approval:**

```
$ echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"dispatch_maintenance_crew","arguments":{"crew_id":"C-04","location":"node_7","justification":"GNN flagged 3-sigma anomaly"}}}' | python -m src.mcp.server
{"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "{\"blocked\": true, \"tier\": \"irreversible\", \"requires_approval\": true, \"reason\": \"Tool 'dispatch_maintenance_crew' is IRREVERSIBLE and requires human approval. Submit this action through the approval gate.\"}"}], "isError": true}}
```

**Kill switch active, everything blocked:**

```
$ echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"set_alert_threshold","arguments":{"node_id":5,"threshold_mw":150,"alert_type":"overload"}}}' | SY1_KILL_SWITCH=1 python -m src.mcp.server
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "{\"blocked\": true, \"tier\": \"reversible\", \"requires_approval\": false, \"reason\": \"Global kill switch is active. All write operations suspended.\"}"}], "isError": true}}
```

On Windows `cmd`, set the variable on its own first: `set "SY1_KILL_SWITCH=1"`.

---

## Approval gate walkthrough

`interrupt_before` pauses the graph before the gate node runs. `Command(resume=...)` only feeds `interrupt()` calls, so it does not reach this node. The operator supplies the token by writing it into the graph state and resuming with `None`:

```python
config = {"configurable": {"thread_id": "session-001"}}
graph.invoke({}, config)              # supervisor requests dispatch_maintenance_crew; graph pauses

pending = dict(graph.get_state(config).values["pending_approval"])
graph.update_state(
    config, {"pending_approval": {**pending, "approval_token": "tok-operator-abc123"}}
)
graph.invoke(None, config)            # gate validates the token, writes an APPROVED audit entry
```

A runnable version, with the graph wiring, is in `tests/test_approval_gate_e2e.py`. Without a token the gate leaves `approved` false and the downstream executor does not act.

---

## Evaluation suite

These are policy-level evaluations. They test the decisions of `PolicyEngine`, not the behaviour of an LLM.

- **Trajectory scenarios** (`src/evals/trajectory_eval.py`): scripted sequences of tool calls scored against the expected policy outcome, including kill-switch scenarios. Run them with `pytest tests/test_evals.py -v`.
- **Promptfoo injection suite** (`evals/promptfoo/injection_suite.yaml`): adversarial strings in tool names and arguments, run through `policy_provider.py` against `PolicyEngine`:
  ```
  promptfoo eval --config evals/promptfoo/injection_suite.yaml
  ```
  A pass rate has not been recorded in this repository yet.
- **LLM-as-judge** (`src/evals/judge.py`): scores agent responses on safety and helpfulness. Needs `ANTHROPIC_API_KEY`:
  ```
  pytest tests/test_evals.py -m live_api -v
  ```

---

## Security

See [docs/threat_model.md](docs/threat_model.md) for the threat catalogue. Properties that the code and tests support:

- Tier decisions are a dictionary lookup, so prompt injection in tool arguments cannot change them.
- No tool input can deactivate the kill switch.
- Unknown tools fail closed with `ValueError`.
- Approval tokens are redacted in the audit log.
- The audit log is SHA-256 hash-chained, and `verify_chain_integrity()` detects tampering offline.

---

## Known limitations

- **Not wired into the agent loop.** `energy_ops_agent.py` does not call the policy engine. Today the governance layer is exercised through the MCP server and the approval gate node, not by the SY1 supervisor itself.
- **The MCP server authorises, it does not execute.** It returns the policy decision for a tool call; running the tool is the job of `ToolExecutor`.
- **The gate does not authenticate the approver.** It accepts any non-empty token and does not bind the token to a `trace_id` (threat model T5). Identity and token verification are not implemented.
- **Evaluations are policy-level.** Trajectories are scripted tool-call sequences and the injection cases are adversarial inputs to a lookup, not recorded agent runs or an LLM under attack.
- **No promptfoo pass rate, Ragas or Langfuse results recorded yet.**
- **Executor paths untested.** Retries, idempotency and typed errors exist in `executor.py` but are only partly covered by tests.
- **The audit log is a local file** (`audit_logs/audit.jsonl`), not a durable or shared store.
- **Lint and type checks run on the governance packages** (`policy`, `tools`, `evals`, `mcp`, `tracing`); the SY1 pipeline code has known lint debt.

---

## Architecture decisions

| ADR | Decision |
|-----|---------|
| [ADR-001](docs/adr/001-python-only-policy-engine.md) | Policy enforcement in pure Python, never in a prompt |
| [ADR-002](docs/adr/002-langgraph-interrupt-approval-gate.md) | LangGraph `interrupt_before` node as the approval gate |
| [ADR-003](docs/adr/003-append-only-hash-chained-audit-log.md) | Append-only SHA-256 hash-chained audit log |
