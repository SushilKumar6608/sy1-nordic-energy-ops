import os
import sys
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gnn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anomaly"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forecasting"))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAG_DOCS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "rag_docs")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")

# =============================================================================
# APP INITIALISATION
# =============================================================================

app = FastAPI(
    title       = "SY1 Nordic Energy Ops Agent API",
    description = """
REST API for the SY1 Nordic Energy Ops Agent system.

Provides endpoints for:
- Grid state: current demand, forecasts, and anomaly status per city
- GNN fault localisation: identify likely fault nodes for a detected anomaly
- Agent pipeline: run the full multi-agent diagnosis and recommendation pipeline
- RAG query: query the knowledge base directly
- RAG upload: ingest new documents into the live knowledge base
- Health check: system status

Designed for Nordic district heating and cooling grid operators.
    """,
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

_retriever = None
_demand_df = None
_detect_df = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from retriever import RAGRetriever
        _retriever = RAGRetriever()
    return _retriever


def get_demand_data() -> pd.DataFrame:
    global _demand_df
    if _demand_df is None:
        path       = os.path.join(PROCESSED_DIR, "energy_demand.csv")
        _demand_df = pd.read_csv(path, parse_dates=["timestamp"])
    return _demand_df


def get_detection_results() -> pd.DataFrame:
    global _detect_df
    if _detect_df is None:
        path       = os.path.join(PROCESSED_DIR, "conformal_detection_results.csv")
        _detect_df = pd.read_csv(path)
    return _detect_df


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class AnomalyEventRequest(BaseModel):
    city:            str             = Field(..., description="City name e.g. Stockholm")
    target:          str             = Field(..., description="heating_demand_mw or cooling_demand_mw")
    anomaly_score:   float           = Field(..., ge=0.0, le=1.0)
    actual_mw:       float           = Field(...)
    forecast_mw:     float           = Field(...)
    deviation_mw:    float           = Field(...)
    temperature_c:   float           = Field(...)
    timestamp:       str             = Field(...)
    return_temp_c:   Optional[float] = Field(None)
    dual_grid_event: bool            = Field(False)
    spatial_cluster: bool            = Field(False)
    phase_inverted:  bool            = Field(False)
    user_query:      Optional[str]   = Field(None)


class RAGQueryRequest(BaseModel):
    query:           str             = Field(..., description="Natural language query")
    top_k:           int             = Field(3, ge=1, le=10)
    category_filter: Optional[str]   = Field(None,
        description="fault_types | city_operations | procedures | equipment")
    use_hyde:        bool            = Field(True,
        description="Use HyDE expansion for better retrieval on informal queries")


class FaultLocalisationRequest(BaseModel):
    city:          str   = Field(..., description="City name e.g. Gothenburg")
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    top_k:         int   = Field(3, ge=1, le=10)


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """System health check — confirms all data files and models are accessible."""
    status = {}

    for fname in ["energy_demand.csv", "conformal_detection_results.csv",
                  "xgboost_results.csv"]:
        path          = os.path.join(PROCESSED_DIR, fname)
        status[fname] = "ok" if os.path.exists(path) else "missing"

    for fname in ["rag_faiss.index", "rag_embedder_name.pkl"]:
        path          = os.path.join(MODELS_DIR, fname)
        status[fname] = "ok" if os.path.exists(path) else "missing"

    cities = ["Stockholm", "Gothenburg", "Malmo", "Uppsala",
              "Umea", "Kiruna", "Are_Ostersund"]
    for city in cities:
        fname         = f"gnn_{city}.pt"
        path          = os.path.join(MODELS_DIR, fname)
        status[fname] = "ok" if os.path.exists(path) else "missing"

    all_ok = all(v == "ok" for v in status.values())

    return {
        "status":    "healthy" if all_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "checks":    status,
    }


# =============================================================================
# GRID STATE ENDPOINTS
# =============================================================================

@app.get("/grid/cities", tags=["Grid State"])
def list_cities():
    """List all monitored cities in the SY1 network."""
    return {
        "cities": [
            "Stockholm", "Gothenburg", "Malmö",
            "Uppsala", "Umeå", "Kiruna", "Åre-Östersund"
        ],
        "count": 7,
    }


@app.get("/grid/state/{city}", tags=["Grid State"])
def get_city_state(city: str, year: int = 2024, month: int = 1):
    """Get grid state summary for a city in a given month."""
    df      = get_demand_data()
    city_df = df[df["city"] == city]
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")

    month_df = city_df[
        (city_df["timestamp"].dt.year  == year) &
        (city_df["timestamp"].dt.month == month)
    ]
    if month_df.empty:
        raise HTTPException(status_code=404,
                            detail=f"No data for {city} in {year}-{month:02d}")

    detect_df   = get_detection_results()
    city_detect = detect_df[detect_df["city"] == city]
    heat_detect = city_detect[city_detect["target"] == "heating_demand_mw"]
    cool_detect = city_detect[city_detect["target"] == "cooling_demand_mw"]

    return {
        "city":   city,
        "period": f"{year}-{month:02d}",
        "heating": {
            "mean_demand_mw":      round(float(month_df["heating_demand_mw"].mean()), 2),
            "max_demand_mw":       round(float(month_df["heating_demand_mw"].max()),  2),
            "anomaly_hours":       int(month_df["anomaly_heating"].sum()),
            "anomaly_rate_pct":    round(float(month_df["anomaly_heating"].mean()) * 100, 2),
            "detection_f1":        float(heat_detect["f1"].values[0])
                                   if not heat_detect.empty else None,
            "detection_precision": float(heat_detect["precision"].values[0])
                                   if not heat_detect.empty else None,
            "detection_recall":    float(heat_detect["recall"].values[0])
                                   if not heat_detect.empty else None,
        },
        "cooling": {
            "mean_demand_mw":      round(float(month_df["cooling_demand_mw"].mean()), 4),
            "max_demand_mw":       round(float(month_df["cooling_demand_mw"].max()),  4),
            "anomaly_hours":       int(month_df["anomaly_cooling"].sum()),
            "anomaly_rate_pct":    round(float(month_df["anomaly_cooling"].mean()) * 100, 2),
            "detection_f1":        float(cool_detect["f1"].values[0])
                                   if not cool_detect.empty else None,
            "detection_precision": float(cool_detect["precision"].values[0])
                                   if not cool_detect.empty else None,
            "detection_recall":    float(cool_detect["recall"].values[0])
                                   if not cool_detect.empty else None,
        },
    }


@app.get("/grid/demand/{city}", tags=["Grid State"])
def get_demand_timeseries(city: str, target: str = "heating_demand_mw",
                           start_date: str = "2024-01-01",
                           end_date:   str = "2024-01-07"):
    """Get hourly demand time series for a city and target. Max 30 days."""
    valid_targets = ["heating_demand_mw", "cooling_demand_mw"]
    if target not in valid_targets:
        raise HTTPException(status_code=400,
                            detail=f"target must be one of {valid_targets}")

    df      = get_demand_data()
    city_df = df[df["city"] == city]
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")

    try:
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if (end - start).days > 30:
        raise HTTPException(status_code=400, detail="Date range cannot exceed 30 days.")

    # Strip timezone for comparison — CSV timestamps have UTC tz info
    city_df = city_df.copy()
    city_df["ts_naive"] = city_df["timestamp"].dt.tz_localize(None) \
        if city_df["timestamp"].dt.tz is not None \
        else city_df["timestamp"]

    result = city_df[
        (city_df["ts_naive"] >= start) &
        (city_df["ts_naive"] <= end)
    ].drop(columns=["ts_naive"]).reset_index(drop=True)

    if result.empty:
        raise HTTPException(status_code=404,
                            detail=f"No data for {city} between {start_date} and {end_date}")

    anomaly_col = "anomaly_heating" if "heating" in target else "anomaly_cooling"
    records = []
    for _, row in result.iterrows():
        records.append({
            "timestamp":     row["timestamp"].isoformat(),
            "demand_mw":     round(float(row[target]), 3),
            "temperature_c": round(float(row["temperature_c"]), 1),
            "anomaly":       int(row[anomaly_col]),
        })

    return {
        "city":       city,
        "target":     target,
        "start_date": start_date,
        "end_date":   end_date,
        "records":    len(records),
        "data":       records,
    }


@app.get("/grid/anomaly-summary", tags=["Grid State"])
def get_anomaly_summary():
    """Get anomaly detection performance summary across all cities and targets."""
    detect_df = get_detection_results()
    summary   = []
    for _, row in detect_df.iterrows():
        summary.append({
            "city":        row["city"],
            "target":      row["target"],
            "strategy":    row.get("strategy", "conformal") if "strategy" in row else "conformal",
            "alpha":       row.get("alpha", None) if "alpha" in row else None,
            "precision":   row["precision"],
            "recall":      row["recall"],
            "f1":          row["f1"],
            "tp":          int(row["tp"]),
            "fp":          int(row["fp"]),
            "fn":          int(row["fn"]),
            "tn":          int(row["tn"]),
            "n_detected":  int(row["n_detected"]),
            "n_true_anom": int(row["n_true_anom"]),
        })

    return {
        "summary":      summary,
        "total_cities": len(detect_df["city"].unique()),
        "total_rows":   len(summary),
    }


# =============================================================================
# GNN FAULT LOCALISATION
# =============================================================================

@app.post("/gnn/localise", tags=["GNN Fault Localisation"])
def localise_fault(request: FaultLocalisationRequest):
    """Run GraphSAGE GNN fault localisation. Returns top-k fault candidate nodes."""
    import torch
    from graph_sage import infer_fault_nodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        fault_df = infer_fault_nodes(
            city=request.city, anomaly_score=request.anomaly_score,
            device=device, top_k=request.top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                            detail=f"GNN model not found for city '{request.city}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GNN inference failed: {str(e)}")

    nodes = []
    for _, row in fault_df.iterrows():
        nodes.append({
            "node_id":           int(row["node_id"]),
            "type":              str(row["type"]),
            "capacity_mw":       round(float(row["capacity_mw"]), 3),
            "fault_probability": round(float(row["fault_probability"]), 4),
        })

    return {
        "city":           request.city,
        "anomaly_score":  request.anomaly_score,
        "top_k":          request.top_k,
        "fault_nodes":    nodes,
        "interpretation": (
            "Nodes ordered by fault probability (highest first). "
            "Kiruna GNN has higher uncertainty (val_loss 0.38) — treat as indicative."
        ),
    }


# =============================================================================
# RAG KNOWLEDGE BASE
# =============================================================================

@app.post("/rag/query", tags=["RAG Knowledge Base"])
def query_knowledge_base(request: RAGQueryRequest):
    """Query the RAG knowledge base with optional HyDE expansion."""
    retriever = get_retriever()

    try:
        context, tier, hyde_para = retriever.query_and_format(
            query_text=request.query, top_k=request.top_k,
            category_filter=request.category_filter, use_hyde=request.use_hyde)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

    if request.use_hyde:
        results, _, _ = retriever.query_with_hyde(
            request.query, request.top_k, request.category_filter)
    else:
        results = retriever.query(request.query, request.top_k, request.category_filter)

    return {
        "query":           request.query,
        "use_hyde":        request.use_hyde,
        "confidence_tier": tier,
        "hyde_paragraph":  hyde_para,
        "results": [
            {
                "rank":     i + 1,
                "title":    r["title"],
                "category": r["category"],
                "score":    round(r["score"], 4),
                "chunk":    r["chunk"][:300] + "..." if len(r["chunk"]) > 300 else r["chunk"],
            }
            for i, r in enumerate(results)
        ],
        "formatted_context": context,
    }


@app.post("/rag/upload", tags=["RAG Knowledge Base"])
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a new document into the live RAG knowledge base.

    Accepts PDF, TXT, or LOG files. The document is:
    1. Read and decoded as text
    2. Chunked with 400-character chunks and 80-character overlap
    3. Embedded using the same sentence-transformer model as the existing index
    4. Appended to the live FAISS index (no rebuild required)
    5. Saved to data/rag_docs/chunks.json for persistence

    This allows operators to add proprietary maintenance manuals, fault logs,
    or equipment documentation without rebuilding the entire knowledge base.
    After upload, the new content is immediately available for retrieval.
    """
    import faiss
    import pickle
    from sentence_transformers import SentenceTransformer

    # Read file content
    content_bytes = await file.read()
    filename      = file.filename or "uploaded_document"

    # Decode text
    try:
        if filename.endswith(".pdf"):
            try:
                import pdfplumber
                from io import BytesIO
                text = ""
                with pdfplumber.open(BytesIO(content_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except ImportError:
                # Fallback: try reading as text
                text = content_bytes.decode("utf-8", errors="ignore")
        else:
            text = content_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Could not read file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File appears to be empty or unreadable.")

    # Chunk the document
    CHUNK_SIZE    = 400
    CHUNK_OVERLAP = 80
    chunks        = []
    start         = 0
    doc_title     = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()

    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "title":    doc_title,
                "category": "uploaded",
                "chunk":    chunk,
                "doc_id":   f"upload_{filename.replace(' ', '_')}",
            })
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP

    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks could be extracted.")

    # Load embedding model and existing index
    model_path  = os.path.join(MODELS_DIR, "rag_embedder_name.pkl")
    index_path  = os.path.join(MODELS_DIR, "rag_faiss.index")
    chunks_path = os.path.join(RAG_DOCS_DIR, "chunks.json")

    with open(model_path, "rb") as f:
        model_name = pickle.load(f)

    embedder = SentenceTransformer(model_name)
    index    = faiss.read_index(index_path)

    # Embed new chunks
    texts      = [c["chunk"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True,
                                  batch_size=32, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Append to FAISS index
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # Append to chunks JSON
    with open(chunks_path, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    existing_chunks.extend(chunks)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, indent=2, ensure_ascii=False)

    # Reload retriever so it picks up the new index
    global _retriever
    _retriever = None

    return {
        "status":       "success",
        "filename":     filename,
        "chunks_added": len(chunks),
        "total_chunks": len(existing_chunks),
        "message":      (f"Successfully ingested {len(chunks)} chunks from '{filename}'. "
                         f"Knowledge base now contains {len(existing_chunks)} total chunks. "
                         f"New content is immediately available for retrieval."),
    }


# =============================================================================
# AGENT PIPELINE
# =============================================================================

@app.post("/agent/diagnose", tags=["Agent Pipeline"])
def run_agent_pipeline(request: AnomalyEventRequest):
    """Run the full 6-agent multi-agent diagnosis pipeline."""
    from energy_ops_agent import run_pipeline

    try:
        state = run_pipeline(
            city=request.city, target=request.target,
            anomaly_score=request.anomaly_score, actual_mw=request.actual_mw,
            forecast_mw=request.forecast_mw, deviation_mw=request.deviation_mw,
            temperature_c=request.temperature_c, timestamp=request.timestamp,
            return_temp_c=request.return_temp_c,
            dual_grid_event=request.dual_grid_event,
            spatial_cluster=request.spatial_cluster,
            phase_inverted=request.phase_inverted,
            user_query=request.user_query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    return {
        "city":           state["city"],
        "target":         state["target"],
        "timestamp":      state["timestamp"],
        "anomaly_score":  state["anomaly_score"],
        "intent":         state["intent"],
        "rag_tier":       state["rag_tier"],
        "fault_nodes":    json.loads(state["fault_nodes"]),
        "demand_summary": state["demand_summary"],
        "diagnosis":      state["diagnosis"],
        "recommendation": state["recommendation"],
        "final_report":   state["final_report"],
        "hyde_paragraph": state.get("hyde_paragraph", ""),
    }


# =============================================================================
# MODEL RESULTS
# =============================================================================

@app.get("/models/forecasting-results", tags=["Model Results"])
def get_forecasting_results():
    """Get XGBoost forecasting model results across all cities and targets."""
    path = os.path.join(PROCESSED_DIR, "xgboost_results.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail="XGBoost results not found. Run train_xgboost.py first.")
    df = pd.read_csv(path)
    return {
        "model":   "XGBoost",
        "results": df.to_dict(orient="records"),
        "note":    ("MAPE for cooling is computed only on hours where demand > 1.0 MW "
                    "to account for near-zero cooling demand in Nordic winter months."),
    }


@app.get("/models/gnn-summary", tags=["Model Results"])
def get_gnn_summary():
    """Get GraphSAGE GNN model summary per city."""
    import torch
    cities = {
        "Stockholm":     "gnn_Stockholm.pt",
        "Gothenburg":    "gnn_Gothenburg.pt",
        "Malmö":         "gnn_Malmo.pt",
        "Uppsala":       "gnn_Uppsala.pt",
        "Umeå":          "gnn_Umea.pt",
        "Kiruna":        "gnn_Kiruna.pt",
        "Åre-Östersund": "gnn_Are_Ostersund.pt",
    }
    results = []
    for city, fname in cities.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                results.append({
                    "city":       city,
                    "n_nodes":    ckpt.get("n_nodes"),
                    "input_dim":  ckpt.get("input_dim"),
                    "hidden_dim": ckpt.get("hidden_dim"),
                    "num_layers": ckpt.get("num_layers"),
                    "status":     "loaded",
                })
            except Exception as e:
                results.append({"city": city, "status": f"error: {str(e)}"})
        else:
            results.append({"city": city, "status": "model_not_found"})
    return {"gnn_models": results}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")