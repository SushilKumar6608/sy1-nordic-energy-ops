# ADR-002: LangGraph interrupt node as the approval gate for irreversible actions

**Status:** Accepted  
**Date:** 2026-11-07  
**Deciders:** SY1-Gov project

---

## Context

IRREVERSIBLE tool calls (crew dispatch, grid isolation, public alerts) require a human operator to confirm before execution. The confirmation mechanism must:

1. Pause the agent mid-graph without losing state.
2. Give the operator full context about what is about to happen.
3. Resume execution only after an explicit human signal - not a timeout, not a default.
4. Leave an auditable record of who approved what, and when.

The system uses LangGraph as the orchestration framework. LangGraph has a built-in `interrupt_before` mechanism that pauses graph execution at a named node and waits for `Command(resume={...})` to continue.

---

## Decision

IRREVERSIBLE tool calls are handled by `src/tools/approval_gate.py`, a LangGraph node compiled with `interrupt_before=["approval_gate"]`. When the supervisor routes to this node, the graph pauses. Execution resumes only when the operator calls `graph.invoke(Command(resume={"approval_token": "..."}))` with a non-empty token. The gate validates the token and writes an `APPROVED` or `DENIED` audit entry before allowing the downstream executor to proceed.

The gate never executes the tool itself. Separation of approval validation from tool execution means the gate can be tested in isolation without side effects.

---

## Consequences

**Positive:**
- LangGraph's checkpoint mechanism preserves the full graph state across the pause; no manual serialisation needed.
- The operator can inspect the full pending action (tool name, inputs, policy reason) from the checkpoint store before deciding.
- The pattern is familiar to LangGraph users and requires no custom async machinery.
- Audit entries for `PENDING_HUMAN_APPROVAL` and `APPROVED`/`DENIED` are written automatically, covering the full approval lifecycle.

**Negative:**
- Requires LangGraph's persistence backend (SQLite or Postgres checkpointer) to be configured for state to survive process restarts during the approval window.
- Multi-operator workflows (e.g. two-person authorisation) require additional state fields - not supported out of the box.

**Alternatives rejected:**
- *Blocking HTTP endpoint* - requires a web server and manual state persistence; adds infrastructure complexity.
- *Out-of-band email/Slack approval* - adds latency and an external dependency; approval tracking becomes harder to audit.
- *LLM second-opinion gate* - ruled out for the same reason as ADR-001: a second LLM call is a second point of failure, not a safety guarantee.
