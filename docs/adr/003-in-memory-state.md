# ADR-003 - In-memory state for audit log and approval gate

**Date:** 2026-11-09
**Status:** Accepted (prototype scope)

## Context

The audit log (`audit.py`) and approval gate (`approval_gate.py`) both hold state that in a
production system would need to be durable, queryable, and potentially shared across
processes or services.

Options considered:

1. **In-memory lists** - simple, zero dependencies, lost on process restart
2. **SQLite file** - lightweight persistent store, no server required
3. **PostgreSQL / cloud database** - production-grade, requires infrastructure

## Decision

In-memory lists for both. The audit log is a plain Python list of dicts. The approval gate
is a plain Python list of `PendingApproval` dataclass instances.

Both provide `clear_*()` helpers used by tests to reset state between runs.

## Consequences

**Positive:**
- Zero external dependencies - CI and local runs need no database
- Tests are fully isolated and fast
- No connection management, migration, or schema concerns at prototype stage

**Negative:**
- All state is lost on process restart
- Not shareable across processes (e.g. if the Streamlit app and the pipeline run separately)
- Audit log cannot be queried or retained for compliance purposes

## Migration path

Both modules expose a narrow interface (`write_audit_entry`, `get_audit_log`,
`request_approval`, `approve`, `get_pending`). Swapping the backing store to SQLite or
PostgreSQL requires only changing the implementation of these functions, not the callers.

The in-memory implementation is explicitly accepted for the current prototype scope. A
persistent store is a production prerequisite before this system handles real infrastructure.
