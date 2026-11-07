# SY1-Gov: Phase 0 Findings

*Living document. Updated as each phase completes. All decisions logged here.*

---

## Repo Audit: Exists vs Missing

### EXISTS (verified in SY1 repo)

| Component | File | Notes |
|---|---|---|
| 6-agent pipeline | `src/agents/energy_ops_agent.py` | Hand-rolled Anthropic SDK direct; NOT LangGraph despite CV wording |
| XGBoost forecasting | `src/forecasting/` | 7 cities, 16 features, real SMHI weather data |
| Conformal anomaly detection | `src/anomaly/conformal_detector.py` | Updated version with n_detected/n_true_anom columns |
| GraphSAGE GNN | `src/gnn/graph_sage.py` | 137-node synthetic grid, fault localisation |
| RAG pipeline | `src/rag/` | HyDE, FAISS IndexFlatIP, intent router, 128 chunks, dim=384 |
| FastAPI backend | `src/api/main.py` | 10 endpoints, Pydantic models - has open tz bug in /grid/demand/{city} |
| Streamlit dashboard | `src/dashboard/` | 3 tabs, Plotly conformal uncertainty bands |
| README | `README.md` | Comprehensive, pushed to GitHub |
| .gitignore | `.gitignore` | .env excluded, models/saved/ excluded, data/ excluded |

### MISSING (to be built in SY1-Gov)

| Component | Target path | Phase |
|---|---|---|
| Policy engine | `src/policy/engine.py` | 1 |
| Tool contracts (Pydantic) | `src/policy/contracts.py` | 1 |
| Approval gate (LangGraph interrupt) | `src/policy/approval_gate.py` | 1 |
| Audit log (append-only, hash-chained) | `src/policy/audit.py` | 1 |
| Central tool executor | `src/tools/executor.py` | 1 |
| Write tools (dispatch crew, isolate segment) | `src/tools/write_tools.py` | 1 |
| Read tools (refactored) | `src/tools/read_tools.py` | 1 |
| Injection attack suite | `src/evals/injection/attack_suite.yaml` | 2 |
| Trajectory evals | `src/evals/trajectories/test_trajectories.py` | 2 |
| Golden trajectory set | `src/evals/trajectories/golden_set.jsonl` | 2 |
| LLM-as-judge config | `src/evals/judge_config.py` | 2 |
| Model router | `src/routing/model_router.py` | 2.5 |
| OWL fault ontology | `src/knowledge/fault_ontology.ttl` | 3 |
| SPARQL query wrapper | `src/knowledge/ontology_query.py` | 3 |
| Langfuse tracing | `src/observability/tracer.py` | 3 |
| Budget guard | `src/observability/budget_guard.py` | 3 |
| Streamlit review queue | `src/dashboard/review_queue.py` | 3 |
| MCP server | `src/mcp_server/` | 3 |
| CI - smoke (PR) | `.github/workflows/ci.yml` | 0 (skeleton now) |
| CI - full suite (nightly) | `.github/workflows/nightly.yml` | 0 (skeleton now) |
| Architecture doc | `docs/architecture.md` | 4 |
| Threat model | `docs/threat_model.md` | 4 |
| ADR: policy-not-in-prompts | `docs/adr/001-policy-not-in-prompts.md` | 4 |
| ADR: model-routing | `docs/adr/002-model-routing.md` | 4 |
| ADR: ontology-vs-flat-rag | `docs/adr/003-ontology-vs-flat-rag.md` | 4 |
| Runbook | `docs/runbook.md` | 4 |

---

## Open Bugs in Existing SY1 (fix before or during Phase 1)

1. **`src/api/main.py` - `get_demand_timeseries`**: timezone mismatch.
   CSV timestamps are UTC-aware; date comparison uses naive `pd.to_datetime`.
   Fix: strip tz with `dt.tz_localize(None)` before comparison.
   Status: fix written, NOT yet saved to disk.

2. **`src/anomaly/conformal_detector.py`**: updated version with n_detected/n_true_anom
   may not be on disk if not saved from last session.
   Status: re-deliver in Phase 1 if needed.

---

## Dependency Decisions

### New packages added in SY1-Gov

| Package | Version | Why |
|---|---|---|
| `rdflib` | 7.0.0 | OWL ontology authoring and SPARQL queries |
| `owlready2` | 0.46 | OWL reasoning (optional, used for subclass inference) |
| `ragas` | 0.1.21 | Outcome quality + LLM-as-judge (already in ASSA ABLOY CV) |
| `deepeval` | 0.21.73 | Trajectory eval assertions (pytest-style) |
| `langfuse` | 2.52.0 | Tracing, cost per agent, dataset management (already piloted at ASSA ABLOY) |
| `opentelemetry-sdk` | 1.26.0 | Structured spans for Langfuse integration |
| `tenacity` | 8.5.0 | Retries with exponential backoff for write tools |
| `structlog` | 24.4.0 | Structured logging for audit trail |
| `cryptography` | 43.0.1 | SHA-256 hash-chaining for audit log tamper evidence |
| `pytest-asyncio` | 0.23.8 | Async test support for FastAPI endpoints |
| `pytest-cov` | 5.0.0 | Coverage reports in CI |
| `promptfoo` | (npm) | Injection test suite - Node.js CLI, installed via npm separately |

### Packages NOT added (and why)

| Package | Decision |
|---|---|
| Fine-tuning libraries (PEFT, trl) | Not in scope - honest gap, discuss conceptually in interview |
| LangSmith | Not added - Langfuse already covers tracing; claiming both would be thin |
| DeepEval vs deepeval | Same package - `deepeval` is the pip name |
| Weaviate | Not needed - FAISS sufficient for this scale |

### Conda environment name: `sy1-gov`
Stored at `D:\conda-envs\sy1-gov` per standing project conventions.

---

## Architecture Decisions (ADR previews - full docs in Phase 4)

**ADR-001: Policy layer is code, not a prompt**
Prompts can be jailbroken. The policy engine is a separate Python module called by the
tool executor before every tool call. The LLM cannot talk around it.

**ADR-002: Model routing by query complexity**
Simple reads and status checks go to claude-haiku-5-5 (fast, cheap).
Diagnosis and recommendation agents use claude-sonnet-4-6 (capable, more expensive).
Router is rule-based in Phase 2.5, upgrading to a classifier if time permits.

**ADR-003: OWL ontology over flat RAG for fault diagnosis**
Flat RAG matches text; it cannot reason about subclass relationships or
transitivity. An OWL ontology with SPARQL narrows fault candidates structurally
before the LLM explains and ranks them. More auditable, more injection-resistant.

---

## Phase Status

- [x] Phase 0: folder structure, CI skeleton, findings.md, environment.yml
- [ ] Phase 1: A, B, D - policy, approval gates, audit log
- [ ] Phase 2: E, F - injection suite, trajectory evals, CI gates
- [ ] Phase 2.5: H - model router
- [ ] Phase 3: C, G, H remainder - handoff UI, ontology, MCP, tracing
- [ ] Phase 4: K - documentation, ADRs, threat model, demo
