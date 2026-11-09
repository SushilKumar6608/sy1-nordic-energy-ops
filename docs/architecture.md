# SY1-Gov - Architecture

## Overview

SY1-Gov is the governed production layer wrapping the SY1 six-agent LangGraph pipeline for
district heating operations. It enforces policy decisions, model routing, budget limits, human
approval gates, and full observability before any tool call reaches the underlying infrastructure.

```
Incoming tool call
       |
       v
  PolicyEngine          - evaluates tier, kill switch, approval requirement
       |
       +---> BudgetGuard        - blocks if token budget exceeded
       |
       +---> ModelRouter        - selects Haiku or Sonnet based on tier + complexity
       |
       +---> ApprovalGate       - holds IRREVERSIBLE calls for human review
       |
       v
  LangfuseTracer        - records span with policy + routing metadata
       |
       v
  Underlying tool execution (SY1 pipeline)
```

## Components

### Policy Layer (`src/policy/`)

| Module | Responsibility |
|---|---|
| `engine.py` | Core policy evaluation - `PermissionTier` enum, `TOOL_PERMISSIONS` registry, `PolicyEngine.evaluate()` |
| `audit.py` | In-memory append-only audit log of every `PolicyDecision` |
| `contracts.py` | Read-only accessors over `TOOL_PERMISSIONS` - tier lookup, tool listing |

**Permission tiers:**

- `READ` - safe read-only queries; always allowed
- `REVERSIBLE` - state-changing but undoable; allowed unless kill switch active
- `IRREVERSIBLE` - permanent actions (crew dispatch, segment isolation); blocked until human approves

**Kill switch:** when active, all non-READ tools are blocked and routing falls back to Haiku.

### Tools Layer (`src/tools/`)

| Module | Responsibility |
|---|---|
| `approval_gate.py` | Holds `PendingApproval` items; `approve()` / `get_pending()` / `clear_approvals()` |

The Streamlit review queue (`app.py`) is the operator UI over this layer.

### Routing Layer (`src/routing/`)

| Module | Responsibility |
|---|---|
| `model_router.py` | Rule-based Haiku vs Sonnet selection |

**Routing rules:**

| Tier | Complexity | Kill switch | Model |
|---|---|---|---|
| READ | simple | off | Haiku |
| READ | complex | off | Sonnet |
| REVERSIBLE | any | off | Sonnet |
| IRREVERSIBLE | any | off | Sonnet |
| any | any | on | Haiku |

Complexity is assessed by input key count (threshold: 3) and total string length (threshold: 200 chars).

### Evaluation Layer (`src/evals/`)

| Module | Responsibility |
|---|---|
| `budget_guard.py` | Token budget enforcement; raises on `is_over_budget()` |
| `trajectory_eval.py` | Offline scripted tool-call sequences asserting expected policy outcomes |
| `judge.py` | LLM-as-judge wrapper; requires live API key; skipped in CI |

### Knowledge Layer (`src/knowledge/` + `knowledge/`)

| File | Responsibility |
|---|---|
| `knowledge/fault_ontology.ttl` | OWL ontology - 10 fault classes with severity, recommended tool, immediate action flag |
| `src/knowledge/ontology_query.py` | SPARQL query wrapper over the ontology; gracefully degrades if rdflib not installed |

### Tracing Layer (`src/tracing/`)

| Module | Responsibility |
|---|---|
| `langfuse_tracer.py` | Records `TraceSpan` with policy + routing metadata; works offline (in-memory) when Langfuse keys absent |

### Evaluation Suite (`evals/promptfoo/`)

| File | Responsibility |
|---|---|
| `injection_suite.yaml` | 25 adversarial test cases across 6 categories |
| `policy_provider.py` | Promptfoo `call_api()` entry point - runs `PolicyEngine` offline |

## Data flows

**Normal READ call:**
```
evaluate("get_demand_forecast", {...})
  -> PolicyDecision(allowed=True, tier=READ)
  -> route() -> Haiku (simple) or Sonnet (complex)
  -> record() -> TraceSpan
```

**IRREVERSIBLE call:**
```
evaluate("dispatch_maintenance_crew", {...})
  -> PolicyDecision(allowed=False, requires_approval=True)
  -> request_approval() -> PendingApproval queued
  -> Operator reviews in app.py -> approve()
  -> Execution proceeds
```

**Kill switch active:**
```
evaluate("set_alert_threshold", {...})
  -> PolicyDecision(allowed=False, reason="Kill switch active")
  -> route() -> Haiku (kill switch override)
  -> No execution
```

## CI/CD

GitHub Actions (`ci.yml`) runs on every push to `main`:

1. Black - code formatting
2. flake8 - linting
3. mypy - type checking (`--explicit-package-bases`)
4. pytest - unit + trajectory tests with coverage

All tests are offline - no API keys required in CI.
