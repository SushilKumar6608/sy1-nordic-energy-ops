import os
import json
import sys
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gnn"))

from retriever import RAGRetriever
from graph_sage import infer_fault_nodes
import torch

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

_retriever = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def call_claude(
    system_prompt: str, user_message: str, max_tokens: int = MAX_TOKENS
) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


# =============================================================================
# AGENT STATE
# =============================================================================


class AgentState(dict):
    pass


def make_state(
    city,
    target,
    anomaly_score,
    actual_mw,
    forecast_mw,
    deviation_mw,
    temperature_c,
    timestamp,
    return_temp_c=None,
    dual_grid_event=False,
    spatial_cluster=False,
    phase_inverted=False,
) -> AgentState:
    return AgentState(
        {
            "city": city,
            "target": target,
            "anomaly_score": anomaly_score,
            "actual_mw": actual_mw,
            "forecast_mw": forecast_mw,
            "deviation_mw": deviation_mw,
            "temperature_c": temperature_c,
            "timestamp": timestamp,
            "return_temp_c": return_temp_c,
            "dual_grid_event": dual_grid_event,
            "spatial_cluster": spatial_cluster,
            "phase_inverted": phase_inverted,
            "demand_summary": "",
            "intent": "",  # from intent router
            "rag_context": "",
            "rag_tier": "",  # HIGH / PARTIAL / MISS
            "hyde_paragraph": "",  # HyDE generated paragraph (audit trail)
            "fault_nodes": "[]",
            "diagnosis": "",
            "recommendation": "",
            "final_report": "",
        }
    )


# =============================================================================
# INTENT ROUTER AGENT
# =============================================================================


def intent_router_agent(state: AgentState, user_query: str = None) -> tuple:
    """
    Intent Router: classifies the query BEFORE hitting the RAG database.

    Three intent categories:
    - SPECIFIC_OPERATIONAL: hyper-specific fault, node, procedure, or equipment
      query that requires documented grid knowledge. Route to HyDE + RAG.
    - GENERAL_ENGINEERING: broad thermodynamic, hydraulic, or engineering
      principle question. Route directly to parametric Claude — no RAG needed.
    - HYBRID: partially specific, partially general. Route to both RAG and
      parametric Claude, then synthesise.

    Returns (intent: str, rationale: str)
    """
    if user_query is None:
        # For pipeline-driven anomaly events, always use SPECIFIC_OPERATIONAL
        return (
            "SPECIFIC_OPERATIONAL",
            "Anomaly event — requires documented fault procedures",
        )

    system_prompt = """You are an intent classification router for a Nordic
district heating and cooling grid operations AI system.

Classify the user query into exactly one of these three categories:

SPECIFIC_OPERATIONAL: The query asks about a specific fault, node, substation,
valve, pipe segment, operating procedure, threshold value, equipment specification,
or documented fault pattern. Requires looking up documented grid knowledge.
Examples: "stuck valve Kiruna node 7", "what is the return temp limit for PHE",
"bolt torque for substation bypass valve", "spring thaw inspection protocol".

GENERAL_ENGINEERING: The query asks about universal thermodynamic, hydraulic,
or engineering principles that apply broadly across all district heating systems.
No specific grid knowledge is needed — Claude's training covers this.
Examples: "why does high return temperature reduce efficiency",
"what causes water hammer in pressurised pipes",
"thermodynamic consequences of PHE fouling".

HYBRID: The query is partly specific (needs grid docs) and partly general
(needs engineering principles). Requires both RAG and parametric reasoning.
Examples: "why is my Stockholm node 14 showing high return temp and what
are the efficiency consequences", "how does spring thaw cause pipe stress
and what should I inspect first".

Respond with ONLY a JSON object:
{"intent": "SPECIFIC_OPERATIONAL" | "GENERAL_ENGINEERING" | "HYBRID",
 "rationale": "one sentence explanation"}"""

    response = call_claude(system_prompt, f"Query: {user_query}", max_tokens=150)

    try:
        parsed = json.loads(response)
        intent = parsed.get("intent", "HYBRID")
        rationale = parsed.get("rationale", "")
    except json.JSONDecodeError:
        intent = "HYBRID"
        rationale = "Router parse error — defaulting to HYBRID"

    return intent, rationale


# =============================================================================
# PARAMETRIC FALLBACK
# =============================================================================


def parametric_engineering_response(query: str, anomaly_context: str = "") -> str:
    """
    Parametric fallback: answer using Claude's engineering training knowledge.
    Strict instruction to NOT invent specific grid topology facts.
    Used when RAG confidence is MISS or intent is GENERAL_ENGINEERING.
    """
    system_prompt = """You are a senior district heating and cooling grid
engineer with 20 years of experience in Nordic energy infrastructure.

You are answering a query where the specific documentation is not available
in the local knowledge base. You MUST follow these strict rules:

1. Reason ONLY from universal district heating and cooling engineering principles.
2. Do NOT invent or assume specific grid topology facts such as:
   - Specific node IDs, pipe lengths, or installation dates
   - Specific pressure values or flow rates for named substations
   - Specific equipment serial numbers or maintenance records
3. When referencing standards, cite the standard name (e.g. EN 253,
   Svensk Fjärrvärme SR 2021:04) but do not invent clause numbers.
4. Be explicit that your answer is based on general engineering principles,
   not documented procedures for this specific grid.
5. If the question requires specific grid data you do not have, say so clearly
   and explain what data the operator should look up in their local SCADA system.

Be precise, professional, and genuinely helpful."""

    user_message = f"""Engineering query:
{query}

{"Additional context from anomaly detection system:" + anomaly_context if anomaly_context else ""}

Provide a technical response based on general district heating engineering
principles. State clearly that this is based on general engineering knowledge,
not documented procedures for this specific grid."""

    return call_claude(system_prompt, user_message)


# =============================================================================
# DEMAND AGENTS
# =============================================================================


def heating_agent(state: AgentState) -> AgentState:
    system_prompt = """You are a district heating grid specialist at a Nordic
energy operations centre. Summarise heating demand anomalies precisely and
briefly. Note: negative deviations during cold snaps may indicate PHE fouling
not supply failure. Dual-grid events may indicate data centre thermal dumps.
Spatial clusters suggest spring thaw. Inverse diurnal patterns suggest sensor
drift. Do not include recommendations."""

    pct = (state["deviation_mw"] / max(state["forecast_mw"], 1.0)) * 100
    ret_info = (
        f"\nReturn line temperature: {state['return_temp_c']:.1f}°C"
        if state.get("return_temp_c")
        else ""
    )
    dual_info = (
        "\n*** DUAL-GRID EVENT: Simultaneous cooling anomaly ***"
        if state.get("dual_grid_event")
        else ""
    )
    clus_info = (
        "\n*** SPATIAL CLUSTER: Multiple adjacent nodes affected ***"
        if state.get("spatial_cluster")
        else ""
    )
    phase_inf = (
        "\n*** PHASE INVERSION: Demand pattern inverted vs diurnal profile ***"
        if state.get("phase_inverted")
        else ""
    )

    msg = f"""Summarise this heating demand anomaly:
City: {state["city"]} | Timestamp: {state["timestamp"]}
Outdoor temperature: {state["temperature_c"]:.1f}°C
Actual: {state["actual_mw"]:.1f} MW | Forecast: {state["forecast_mw"]:.1f} MW
Deviation: {state["deviation_mw"]:+.1f} MW ({pct:+.1f}%)
Severity: {state["anomaly_score"]:.2f}{ret_info}{dual_info}{clus_info}{phase_inf}

2-3 sentence technical summary only."""

    state["demand_summary"] = call_claude(system_prompt, msg)
    return state


def cooling_agent(state: AgentState) -> AgentState:
    system_prompt = """You are a district cooling grid specialist at a Nordic
energy operations centre. Summarise cooling demand anomalies precisely.
Pay special attention to winter shutdown violations, data centre thermal dumps,
and anti-cyclical demand pulses. Be precise, factual, brief."""

    pct = (state["deviation_mw"] / max(state["forecast_mw"], 0.1)) * 100
    winter = state["temperature_c"] < 12.0
    dual_info = (
        "\n*** DUAL-GRID EVENT: Simultaneous heating anomaly — possible thermal dump ***"
        if state.get("dual_grid_event")
        else ""
    )
    ret_info = (
        f"\nReturn line temperature: {state['return_temp_c']:.1f}°C"
        if state.get("return_temp_c")
        else ""
    )

    msg = f"""Summarise this cooling demand anomaly:
City: {state["city"]} | Timestamp: {state["timestamp"]}
Outdoor temperature: {state["temperature_c"]:.1f}°C
{"*** WINTER ANOMALY: Cooling demand detected below 12°C ***" if winter else ""}
Actual: {state["actual_mw"]:.1f} MW | Forecast: {state["forecast_mw"]:.1f} MW
Deviation: {state["deviation_mw"]:+.1f} MW ({pct:+.1f}%){ret_info}{dual_info}

2-3 sentence technical summary only."""

    state["demand_summary"] = call_claude(system_prompt, msg)
    return state


# =============================================================================
# RAG RETRIEVAL AGENT (with Intent Router + HyDE + Confidence Tiers)
# =============================================================================


def rag_retrieval_agent(state: AgentState, user_query: str = None) -> AgentState:
    """
    Multi-tier RAG retrieval agent.

    Flow:
    1. Intent Router classifies the query
    2. SPECIFIC_OPERATIONAL → HyDE + FAISS search
    3. GENERAL_ENGINEERING  → parametric fallback directly (no FAISS)
    4. HYBRID               → HyDE + FAISS + parametric, merged
    5. Confidence tier gates: HIGH / PARTIAL / MISS
    6. MISS → parametric fallback with explicit uncertainty flag
    """
    retriever = get_retriever()

    # Step 1: intent routing
    intent, rationale = intent_router_agent(state, user_query)
    state["intent"] = intent
    print(f"  Intent: {intent} — {rationale}")

    target_type = "heating" if "heating" in state["target"] else "cooling"
    deviation = state["deviation_mw"]
    temp = state["temperature_c"]

    # Build domain query from anomaly context
    if user_query:
        domain_query = user_query
    elif state.get("dual_grid_event"):
        domain_query = "data centre thermal dump return line temperature district heating cooling simultaneous"
    elif state.get("phase_inverted"):
        domain_query = (
            "inverse diurnal sensor drift calibration phase inversion meter fault"
        )
    elif state.get("spatial_cluster"):
        domain_query = (
            "spring thaw pipe stress freeze thaw cycling adjacent nodes spatial cluster"
        )
    elif target_type == "cooling" and temp < 12.0:
        domain_query = f"stuck valve cooling winter shutdown {state['city']} temperature below zero"
    elif target_type == "heating" and deviation < 0 and temp < -2.0:
        domain_query = "heat exchanger fouling PHE scaling return temperature secondary loop degradation"
    elif deviation > 0 and abs(deviation) / max(state["forecast_mw"], 1.0) > 0.4:
        domain_query = f"{target_type} demand spike pipe burst fault {state['city']}"
    elif deviation < 0 and abs(deviation) / max(state["forecast_mw"], 1.0) > 0.3:
        domain_query = f"{target_type} supply failure demand drop control system misfire {state['city']}"
    else:
        domain_query = (
            f"{target_type} demand deviation insulation sensor drift {state['city']}"
        )

    rag_context = ""
    rag_tier = "MISS"
    hyde_para = ""

    # Step 2-4: route by intent
    if intent == "GENERAL_ENGINEERING":
        # Skip RAG entirely — go straight to parametric
        rag_tier = "PARAMETRIC"
        rag_context = (
            "NOTE: Query classified as general engineering question. "
            "Responding from engineering principles only — "
            "no specific grid documentation consulted.\n\n"
        )
        anomaly_ctx = (
            f"City: {state['city']}, "
            f"Target: {state['target']}, "
            f"Deviation: {state['deviation_mw']:+.1f} MW, "
            f"Temperature: {state['temperature_c']:.1f}°C"
        )
        rag_context += parametric_engineering_response(domain_query, anomaly_ctx)

    elif intent in ("SPECIFIC_OPERATIONAL", "HYBRID"):
        # HyDE + FAISS retrieval
        results, hyde_para, rag_tier = retriever.query_with_hyde(
            domain_query, top_k=2, category_filter="fault_types"
        )

        proc_results = retriever.query(
            f"operator response procedure {target_type} {state['city']}",
            top_k=1,
            category_filter="procedures",
        )
        city_results = retriever.query(
            f"city operations {state['city']} grid vulnerabilities",
            top_k=1,
            category_filter="city_operations",
        )

        if state.get("return_temp_c") and state["return_temp_c"] > 55.0:
            equip_results = retriever.query(
                "return line temperature management PHE fouling thermal dump thresholds",
                top_k=1,
                category_filter="equipment",
            )
            all_results = results + proc_results + city_results + equip_results
        else:
            all_results = results + proc_results + city_results

        rag_context = retriever.format_context(all_results, rag_tier)

        # If MISS and HYBRID — supplement with parametric
        if rag_tier == "MISS":
            anomaly_ctx = (
                f"City: {state['city']}, "
                f"Deviation: {state['deviation_mw']:+.1f} MW, "
                f"Temperature: {state['temperature_c']:.1f}°C"
            )
            parametric = parametric_engineering_response(domain_query, anomaly_ctx)
            rag_context += (
                f"\n\n--- PARAMETRIC ENGINEERING KNOWLEDGE ---\n"
                f"(Knowledge base miss — reasoning from engineering "
                f"principles)\n\n{parametric}"
            )

    state["rag_context"] = rag_context
    state["rag_tier"] = rag_tier
    state["hyde_paragraph"] = hyde_para
    return state


# =============================================================================
# GNN LOCALISATION AGENT
# =============================================================================


def gnn_localisation_agent(state: AgentState) -> AgentState:
    try:
        fault_df = infer_fault_nodes(
            city=state["city"],
            anomaly_score=state["anomaly_score"],
            device=_device,
            top_k=3,
        )
        nodes_list = []
        for _, row in fault_df.iterrows():
            nodes_list.append(
                {
                    "node_id": int(row["node_id"]),
                    "type": str(row["type"]),
                    "capacity_mw": round(float(row["capacity_mw"]), 3),
                    "fault_probability": round(float(row["fault_probability"]), 4),
                }
            )
        state["fault_nodes"] = json.dumps(nodes_list)
    except Exception as e:
        state["fault_nodes"] = json.dumps([{"error": f"GNN failed: {str(e)}"}])
    return state


# =============================================================================
# DIAGNOSIS AGENT
# =============================================================================


def diagnosis_agent(state: AgentState) -> AgentState:
    system_prompt = """You are a senior grid diagnostician at a Nordic energy
operations centre. Synthesise anomaly data, grid documentation, and GNN
localisation to diagnose root cause. Recognise:
- Heat exchanger fouling: demand DROP during cold snap + high return temperature
- Data centre thermal dumps: simultaneous heating AND cooling demand collapse
- Inverse diurnal sensor drift: demand peaks at 02:00-05:00
- Spring thaw pipe stress: spatially clustered adjacent node anomalies April-May
- Control system misfires: synchronised simultaneous drops across non-adjacent nodes
Note the RAG confidence tier when citing documentation.
State confidence: low/medium/high. 3-4 sentences."""

    fault_nodes = json.loads(state["fault_nodes"])
    nodes_text = (
        "\n".join(
            [
                f"  Node {n.get('node_id','?')} ({n.get('type','?')}): "
                f"prob={n.get('fault_probability',0):.4f}, "
                f"cap={n.get('capacity_mw',0):.2f} MW"
                for n in fault_nodes
                if "error" not in n
            ]
        )
        or "  GNN localisation unavailable"
    )

    extra = ""
    if state.get("return_temp_c"):
        extra += f"\nReturn line temperature: {state['return_temp_c']:.1f}°C"
    if state.get("dual_grid_event"):
        extra += "\nDUAL-GRID EVENT: simultaneous anomaly in both heating and cooling"
    if state.get("spatial_cluster"):
        extra += "\nSPATIAL CLUSTER: multiple adjacent nodes affected simultaneously"
    if state.get("phase_inverted"):
        extra += "\nPHASE INVERSION: demand pattern is temporally inverted"

    rag_note = (
        f"\nRAG confidence tier: {state['rag_tier']} " f"(Intent: {state['intent']})"
    )

    msg = f"""Diagnose this grid anomaly:

ANOMALY SUMMARY:
{state["demand_summary"]}{extra}{rag_note}

GNN FAULT LOCALISATION:
{nodes_text}

KNOWLEDGE BASE / ENGINEERING CONTEXT:
{state["rag_context"]}

Most likely root cause, supporting evidence, and confidence level?"""

    state["diagnosis"] = call_claude(system_prompt, msg)
    return state


# =============================================================================
# RECOMMENDATION AGENT
# =============================================================================


def recommendation_agent(state: AgentState) -> AgentState:
    system_prompt = """You are a grid operations coordinator at a Nordic energy
operations centre. Generate clear prioritised operator action plans (max 5 steps).
Time-sensitive actions first.
- Data centre thermal dumps: include return line temperature monitoring steps.
- PHE fouling: include return temperature cross-check before dispatching crew.
- Inverse diurnal drift: recommend remote recalibration before physical dispatch.
- Spring thaw: cover the full spatial node cluster, not just the top GNN node.
- Control system misfires: check SCADA event log and issue manual override first.
- General engineering queries (PARAMETRIC tier): recommend operator consult
  local SCADA data before taking action, since answer is based on general
  principles not documented grid-specific procedures."""

    fault_nodes = json.loads(state["fault_nodes"])
    top_node = fault_nodes[0] if fault_nodes and "error" not in fault_nodes[0] else None
    node_hint = (
        f"GNN: node {top_node['node_id']} ({top_node['type']}) "
        f"p={top_node['fault_probability']:.3f}"
        if top_node
        else "GNN unavailable"
    )

    flags = []
    if state.get("dual_grid_event"):
        flags.append("DUAL-GRID EVENT — possible data centre thermal dump")
    if state.get("return_temp_c") and state["return_temp_c"] > 60:
        flags.append(f"RETURN TEMP CRITICAL: {state['return_temp_c']:.1f}°C")
    if state.get("spatial_cluster"):
        flags.append("SPATIAL CLUSTER — inspect all adjacent flagged nodes")
    if state.get("phase_inverted"):
        flags.append("PHASE INVERSION — attempt remote recalibration first")
    if state["rag_tier"] in ("MISS", "PARAMETRIC"):
        flags.append(
            "KNOWLEDGE BASE MISS — actions based on general engineering "
            "principles, verify against local SCADA before executing"
        )

    msg = f"""Action plan for:
City: {state["city"]} | {"Heating" if "heating" in state["target"] else "Cooling"}
Time: {state["timestamp"]} | Temp: {state["temperature_c"]:.1f}°C

DIAGNOSIS:
{state["diagnosis"]}

GNN: {node_hint}
RAG tier: {state["rag_tier"]}
Flags: {chr(10).join(flags) if flags else "None"}

Numbered action plan, max 5 steps."""

    state["recommendation"] = call_claude(system_prompt, msg)
    return state


# =============================================================================
# SUPERVISOR AGENT
# =============================================================================


def supervisor_agent(state: AgentState) -> AgentState:
    system_prompt = """You are the AI supervisor of a Nordic energy grid
operations system. Compile structured incident reports from sub-agent analyses.
Include the RAG confidence tier and intent classification in the report so
operators know whether the diagnosis is based on documented procedures or
general engineering principles. Be concise and professional."""

    target_label = "Heating" if "heating" in state["target"] else "Cooling"

    msg = f"""Compile structured incident report:

INCIDENT:
- City: {state["city"]} | Grid: {target_label}
- Time: {state["timestamp"]} | Temperature: {state["temperature_c"]:.1f}°C
- Actual: {state["actual_mw"]:.1f} MW | Forecast: {state["forecast_mw"]:.1f} MW
- Deviation: {state["deviation_mw"]:+.1f} MW | Severity: {state["anomaly_score"]:.2f}/1.0
- RAG tier: {state["rag_tier"]} | Intent: {state["intent"]}

DEMAND ANALYSIS:
{state["demand_summary"]}

DIAGNOSIS:
{state["diagnosis"]}

RECOMMENDED ACTIONS:
{state["recommendation"]}

Format: INCIDENT SUMMARY | ROOT CAUSE ASSESSMENT | ACTION PLAN | SEVERITY LEVEL
Include a DATA SOURCE note stating whether diagnosis used documented procedures
(HIGH/PARTIAL RAG) or general engineering principles (MISS/PARAMETRIC)."""

    state["final_report"] = call_claude(system_prompt, msg)
    return state


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================


def run_pipeline(
    city,
    target,
    anomaly_score,
    actual_mw,
    forecast_mw,
    deviation_mw,
    temperature_c,
    timestamp,
    return_temp_c=None,
    dual_grid_event=False,
    spatial_cluster=False,
    phase_inverted=False,
    user_query=None,
) -> AgentState:
    """
    Full multi-agent pipeline with Intent Router + HyDE + Confidence Tiers.

    Optional user_query: if provided, the intent router classifies it and
    routes accordingly. If None, pipeline runs in anomaly-event mode
    (always SPECIFIC_OPERATIONAL).
    """
    print(f"\n{'='*60}")
    print(f"SY1 Nordic Energy Ops Agent -- {city} {target}")
    if user_query:
        print(f"User query: {user_query}")
    print(f"{'='*60}")

    state = make_state(
        city,
        target,
        anomaly_score,
        actual_mw,
        forecast_mw,
        deviation_mw,
        temperature_c,
        timestamp,
        return_temp_c,
        dual_grid_event,
        spatial_cluster,
        phase_inverted,
    )

    print("\n[1/6] Running demand agent...")
    state = heating_agent(state) if "heating" in target else cooling_agent(state)
    print("  Done.")

    print("[2/6] Running RAG retrieval agent (Intent Router + HyDE)...")
    state = rag_retrieval_agent(state, user_query)
    print(f"  RAG tier: {state['rag_tier']} | Intent: {state['intent']}")

    print("[3/6] Running GNN localisation agent...")
    state = gnn_localisation_agent(state)
    print("  Done.")

    print("[4/6] Running diagnosis agent...")
    state = diagnosis_agent(state)
    print("  Done.")

    print("[5/6] Running recommendation agent...")
    state = recommendation_agent(state)
    print("  Done.")

    print("[6/6] Running supervisor agent...")
    state = supervisor_agent(state)
    print("  Done.")

    print(f"\n{'='*60}")
    print("FINAL INCIDENT REPORT:")
    print("=" * 60)
    print(state["final_report"])

    return state


# =============================================================================
# TEST CASES
# =============================================================================

if __name__ == "__main__":

    # Test 1: Pipe burst (SPECIFIC_OPERATIONAL — HIGH RAG confidence expected)
    print("\n>>> TEST 1: Pipe burst — Gothenburg")
    run_pipeline(
        city="Gothenburg",
        target="heating_demand_mw",
        anomaly_score=0.82,
        actual_mw=312.4,
        forecast_mw=198.6,
        deviation_mw=113.8,
        temperature_c=-8.3,
        timestamp="2024-01-15T03:00:00+00:00",
    )

    # Test 2: Stuck valve — winter (SPECIFIC_OPERATIONAL)
    print("\n\n>>> TEST 2: Stuck valve — Kiruna")
    run_pipeline(
        city="Kiruna",
        target="cooling_demand_mw",
        anomaly_score=0.91,
        actual_mw=0.09,
        forecast_mw=0.0,
        deviation_mw=0.09,
        temperature_c=-24.5,
        timestamp="2024-02-03T14:00:00+00:00",
    )

    # Test 3: PHE fouling (SPECIFIC_OPERATIONAL — negative deviation + return temp)
    print("\n\n>>> TEST 3: Heat exchanger fouling — Uppsala")
    run_pipeline(
        city="Uppsala",
        target="heating_demand_mw",
        anomaly_score=0.74,
        actual_mw=68.3,
        forecast_mw=98.7,
        deviation_mw=-30.4,
        temperature_c=-11.2,
        timestamp="2024-01-22T08:00:00+00:00",
        return_temp_c=63.5,
    )

    # Test 4: Data centre thermal dump (dual-grid, Level 4)
    print("\n\n>>> TEST 4: Data centre thermal dump — Stockholm")
    run_pipeline(
        city="Stockholm",
        target="heating_demand_mw",
        anomaly_score=0.95,
        actual_mw=124.3,
        forecast_mw=487.2,
        deviation_mw=-362.9,
        temperature_c=-6.1,
        timestamp="2024-03-08T11:30:00+00:00",
        return_temp_c=67.8,
        dual_grid_event=True,
    )

    # Test 5: Inverse diurnal sensor drift
    print("\n\n>>> TEST 5: Inverse diurnal sensor drift — Malmö")
    run_pipeline(
        city="Malmö",
        target="heating_demand_mw",
        anomaly_score=0.61,
        actual_mw=187.4,
        forecast_mw=142.1,
        deviation_mw=45.3,
        temperature_c=2.4,
        timestamp="2024-04-03T02:30:00+00:00",
        phase_inverted=True,
    )

    # Test 6: Spring thaw pipe stress (spatial cluster)
    print("\n\n>>> TEST 6: Spring thaw — Umeå")
    run_pipeline(
        city="Umeå",
        target="heating_demand_mw",
        anomaly_score=0.77,
        actual_mw=89.4,
        forecast_mw=71.2,
        deviation_mw=18.2,
        temperature_c=1.8,
        timestamp="2024-04-18T10:00:00+00:00",
        spatial_cluster=True,
    )

    # Test 7: Control system misfire (synchronised demand drop)
    print("\n\n>>> TEST 7: Control system misfire — Stockholm")
    run_pipeline(
        city="Stockholm",
        target="heating_demand_mw",
        anomaly_score=0.88,
        actual_mw=198.3,
        forecast_mw=521.7,
        deviation_mw=-323.4,
        temperature_c=-9.4,
        timestamp="2024-01-30T18:00:00+00:00",
    )

    # Test 8: GENERAL_ENGINEERING query — Intent Router should bypass RAG
    print("\n\n>>> TEST 8: General engineering query (no specific fault)")
    run_pipeline(
        city="Stockholm",
        target="heating_demand_mw",
        anomaly_score=0.45,
        actual_mw=310.2,
        forecast_mw=290.1,
        deviation_mw=20.1,
        temperature_c=-3.2,
        timestamp="2024-02-10T09:00:00+00:00",
        user_query="What are the thermodynamic consequences of high return line temperatures on overall grid efficiency?",
    )

    # Test 9: Out-of-knowledge-base query — HyDE should trigger MISS + parametric
    print("\n\n>>> TEST 9: Out-of-KB query (HyDE + parametric fallback)")
    run_pipeline(
        city="Gothenburg",
        target="heating_demand_mw",
        anomaly_score=0.55,
        actual_mw=215.3,
        forecast_mw=198.1,
        deviation_mw=17.2,
        temperature_c=-1.4,
        timestamp="2024-03-01T07:00:00+00:00",
        user_query="What is the recommended bolt torque specification for the bypass valve actuator flange on a DN200 substation connection?",
    )
