# ADR-003: Append-only SHA-256 hash-chained audit log

**Status:** Accepted  
**Date:** 2026-11-07  
**Deciders:** SY1-Gov project

---

## Context

Every policy decision and approval event must be auditable. The audit record must be:

1. Tamper-evident: a modified or deleted entry should be detectable offline.
2. Redaction-aware: approval tokens and secrets must not appear in plaintext.
3. Portable: the record must be readable without a database or external service.
4. Fast to write: audit writes are on the hot path of every tool call.

---

## Decision

The audit log is an append-only JSONL file at `logs/audit.jsonl`. Each entry is a JSON object written as a single line. It contains:

- `audit_id` - UUID for this entry
- `timestamp` - ISO 8601 UTC
- `actor`, `tool_name`, `inputs`, `policy_decision`, `outcome`
- `entry_hash` - SHA-256 of the JSON body of this entry
- `prev_hash` - `entry_hash` of the immediately preceding entry (genesis: `"0" * 64`)

Before writing, sensitive fields (`approval_token`, `password`, `secret`, `api_key`, and any nested equivalents) are replaced with `"[REDACTED]"`. A boolean `approval_token_present` is set to record whether a token was supplied, without revealing its value.

The chain can be verified with `verify_chain(log_path)` which replays all entries and checks that each `prev_hash` matches the previous `entry_hash`.

---

## Consequences

**Positive:**
- No database dependency: the log is a plain file that can be archived, copied, or reviewed with `grep`/`jq`.
- Tamper detection works offline: a single deleted or modified entry breaks the chain.
- Write performance is a single `open(append)` + `json.dumps` + `sha256` per entry - microseconds.
- Sensitive fields are never stored; there is no rotation or expiry problem.

**Negative:**
- The file grows without bound; a log rotation strategy (daily rotation, S3 archival) is out of scope and must be added before production deployment.
- The hash chain does not prove who wrote an entry (no signing key); it only proves entries have not been modified after writing.
- Concurrent writes from multiple processes require an external lock or a database-backed log for correctness; the current implementation uses Python's file append mode which is safe for single-process use.

**Alternatives rejected:**
- *Relational database (Postgres/SQLite)* - adds a dependency for what is essentially a write-once record; JSONL is simpler to operate and audit.
- *No tamper detection* - rejected because audit logs that can be silently edited provide no actual assurance.
- *Merkle tree / blockchain* - over-engineering for the threat model; a simple linear hash chain is sufficient and far easier to verify.
