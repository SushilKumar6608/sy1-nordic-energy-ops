# ADR-001: Policy enforcement in pure Python, never in a prompt

**Status:** Accepted  
**Date:** 2026-11-07  
**Deciders:** SY1-Gov project

---

## Context

The SY1 system uses an LLM as the supervisor agent. A naive implementation would instruct the LLM via a system prompt to "only call irreversible tools after getting approval." This is insufficient for a production governance layer: prompt instructions can be overridden by adversarial user input, by prompt injection embedded in tool outputs, or simply by LLM hallucination under load.

The system manages real-world physical infrastructure (district heating grids). An uncontrolled `dispatch_maintenance_crew` or `isolate_grid_segment` call has safety and cost consequences that cannot be undone in the same way a database row can be rolled back.

---

## Decision

All permission enforcement lives in `src/policy/engine.py`, a pure Python module with no LLM calls and no network dependencies. The `TOOL_PERMISSIONS` registry is a plain Python dict mapping tool name to `PermissionTier`. The `PolicyEngine.evaluate()` method is called on every tool invocation before execution, and its result is the authoritative decision.

The LLM cannot see, modify, or talk around this layer. It can only observe the blocked outcome and the reason string.

---

## Consequences

**Positive:**
- The enforcement path has zero LLM latency or cost.
- It is fully testable without API keys (169 offline tests pass in CI).
- Adding a new tool requires editing one dict entry - there is no prompt to update.
- Fail-closed by default: any tool not in the registry raises `ValueError` and is treated as blocked by callers.

**Negative:**
- The registry is static; tier changes require a code deploy, not a configuration update.
- Complex context-dependent rules (e.g. "allow dispatch only if GNN confidence > 0.9") would require extending `PolicyEngine.evaluate()` with additional logic rather than prompting the LLM.

**Alternatives rejected:**
- *System prompt instructions only* - ruled out because prompts can be bypassed.
- *LLM self-evaluation gate* - ruled out because it adds latency, cost, and a second failure mode.
- *External OPA/Rego policy* - considered; rejected as over-engineering for the current scope; can be adopted later if the policy registry grows large.
