# ADR-001 - Fail-closed policy engine with explicit tool registry

**Date:** 2026-11-09
**Status:** Accepted

## Context

The SY1 pipeline calls tools that range from safe read queries to irreversible physical
actions (crew dispatch, grid segment isolation). We need a policy layer that decides whether
each tool call is allowed before execution.

The key design question was: what happens when a tool is not in the registry?

Two options were considered:

- **Fail-open:** unknown tools are allowed by default (permissive)
- **Fail-closed:** unknown tools raise an error and are blocked (restrictive)

## Decision

The engine is fail-closed. `PolicyEngine.evaluate()` raises `ValueError` for any tool name
not present in `TOOL_PERMISSIONS`.

`TOOL_PERMISSIONS` is a hardcoded dict of exactly 10 tools. There is no dynamic registration
path at runtime.

## Consequences

**Positive:**
- No silent permission escalation if a new tool is added to the pipeline without being
  registered in the governance layer
- Security posture is predictable - the full permission surface is visible in one place
- Tests can assert exact registry contents

**Negative:**
- Adding a new tool to SY1 requires a corresponding entry in `TOOL_PERMISSIONS` or the
  pipeline will fail at runtime
- The registry must be kept in sync with the pipeline manually

## Alternatives considered

Fail-open with logging was rejected because it would allow new tools to bypass governance
silently, which is exactly the class of risk SY1-Gov is designed to prevent.
