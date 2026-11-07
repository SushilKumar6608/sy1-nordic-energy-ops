# ⚡ SY1 - Nordic Energy Ops Agent

**Multi-agent AI system for Nordic district heating and cooling grid intelligence**

A production-grade ML + agentic AI pipeline built on real Swedish weather data (SMHI), synthetic district grid topology, and a six-agent LangGraph-style orchestration layer. Designed to detect, localise, diagnose, and recommend operator responses to anomalies in district heating and cooling grids across seven Swedish cities.

---

## Architecture Overview

```
SMHI Open Data API (real)          Synthetic Grid Topology
    425K hourly records                 137 nodes, 141 edges
    7 Swedish cities, 2018–2024         7 cities, pre-insulated pipes
           │                                      │
           ▼                                      ▼
   ┌───────────────────────────────────────────────────┐
   │              Layer 1 — Data Ingestion             │
   │  smhi_fetcher · grid_generator · demand_generator │
   └───────────────────────┬───────────────────────────┘
                           │
                           ▼
   ┌───────────────────────────────────────────────────┐
   │           Layer 2 — Time Series Forecasting       │
   │   XGBoost (winner) · LSTM 3h-pooled · Conformal  │
   └───────────────────────┬───────────────────────────┘
                           │
                           ▼
   ┌───────────────────────────────────────────────────┐
   │          Layer 3 — Anomaly Detection              │
   │  Locally Adaptive Conformal Prediction (heating)  │
   │  Hybrid Adaptive Conformal Prediction (cooling)   │
   └───────────────────────┬───────────────────────────┘
                           │
                           ▼
   ┌───────────────────────────────────────────────────┐
   │        Layer 4 — GNN Fault Localisation           │
   │   GraphSAGE · 137 nodes · node-level fault prob   │
   └───────────────────────┬───────────────────────────┘
                           │
                           ▼
   ┌───────────────────────────────────────────────────┐
   │     Layer 5 — Multi-Agent Orchestration           │
   │  Intent Router → HyDE RAG → GNN → Diagnosis      │
   │  → Recommendation → Supervisor (Claude Sonnet)    │
   └───────────────────────┬───────────────────────────┘
                           │
                           ▼
   ┌──────────────────────┐  ┌───────────────────────┐
   │   FastAPI Backend    │  │  Streamlit Dashboard   │
   │   10 REST endpoints  │  │  3 tabs · Plotly bands │
   └──────────────────────┘  └───────────────────────┘
```

---

## Key Technical Highlights

### Forecasting
- **XGBoost** with 16 engineered features (lag-1h/6h/24h/168h, rolling mean/std, cyclical time encodings, weather covariates) — outperforms LSTM on all 7 cities
- **LSTM** with 3-hour pooled sequences (Option C): 7-day lookback compressed from 168 to 56 timesteps, improving gradient flow and cutting training time 3×
- **Time-based split**: train 2018–2022, val 2023 (conformal calibration), test 2024

### Anomaly Detection
- **Heating**: locally adaptive conformal prediction — rolling 24h sigma normalisation eliminates false positives during morning demand ramps; mathematically guaranteed FPR ≤ 5%
- **Cooling**: hybrid adaptive conformal — normalized scores for active summer months, raw residual thresholds for winter zero-demand months; 3h temporal persistence filter eliminates transient false positives
- **Physical realism mask**: cooling forced to zero below 12°C ambient (unless fault signature injected) — mirrors real Nordic grid operator data

### GNN Fault Localisation
- **GraphSAGE** (2-layer, 64 hidden dim) trained per city on synthetic fault snapshots
- Takes city-level anomaly score → outputs per-node fault probability across grid topology
- 500 training snapshots per city with physically grounded fault label generation

### RAG Knowledge Base
- **128 chunks** across 23 documents: fault type descriptions, city operations, SOPs, equipment specs
- **HyDE (Hypothetical Document Embeddings)**: Claude generates an ideal manual paragraph before FAISS search, bridging informal operator language and formal technical language
- **Intent Router**: classifies queries as SPECIFIC_OPERATIONAL / GENERAL_ENGINEERING / HYBRID before hitting the vector database — optimises latency and prevents hallucination of grid-specific facts
- **Confidence tiers**: HIGH (≥0.40) / PARTIAL (0.25–0.40) / MISS (<0.25) → parametric fallback with explicit uncertainty flag

### Multi-Agent Pipeline (6 agents)
1. **Demand agent** — heating or cooling specialist summarises anomaly context
2. **RAG retrieval agent** — Intent Router + HyDE + confidence-gated retrieval
3. **GNN localisation agent** — fault node identification
4. **Diagnosis agent** — root cause synthesis with 11 fault type recognition patterns
5. **Recommendation agent** — prioritised operator action plan
6. **Supervisor agent** — structured incident report assembly

### Fault Types Covered
| Fault | Signature | Detection Method |
|---|---|---|
| Pipe burst | +40–80% demand spike | Conformal heating |
| Supply failure | Demand drop to <20% forecast | Conformal heating |
| Insulation degradation | Slow upward drift 10–20% | Conformal heating |
| PHE fouling | Demand DROP during cold snap + high return temp | Conformal + RAG diagnosis |
| Data centre thermal dump | Simultaneous heating + cooling collapse | Dual-grid flag + RAG |
| Stuck valve | Steady 0.05–0.15 MW during winter shutdown | Conformal cooling |
| Sensor drift | Slow linear ramp, uncorrelated with weather | Conformal cooling |
| Inverse diurnal drift | Demand peak at 02:00, drop at 07:00 | Phase inversion flag |
| Spring thaw pipe stress | Spatial cluster of adjacent node anomalies | GNN spatial pattern |
| Control system misfire | Synchronised drop across non-adjacent nodes | GNN diffuse pattern |
| Anti-cyclical pulse | Cooling spike during freezing temperatures | Conformal cooling |

---

## Data Sources

| Source | Type | Coverage |
|---|---|---|
| SMHI Open Data API | Real — hourly temperature, wind, humidity | 7 Swedish cities, 2018–2024, 425K records |
| Energimarknadsinspektionen (Ei) | Real — district heating volumes | Sweden, annual (scale calibration) |
| Synthetic grid topology | Generated — nodes, pipes, substations | 137 nodes, 141 edges, 7 cities |
| Synthetic demand + faults | Physics-grounded on real SMHI weather | 425K records, ~1% anomaly rate |

---

## Results

### XGBoost Forecasting (2024 test set)

| City | Heating MAE | Cooling MAE |
|---|---|---|
| Stockholm | 15.3 MW | 0.65 MW |
| Gothenburg | 7.2 MW | 0.37 MW |
| Malmö | 4.9 MW | 0.30 MW |
| Uppsala | 2.5 MW | 0.17 MW |
| Umeå | 2.2 MW | 0.09 MW |
| Kiruna | 2.0 MW | 0.03 MW |
| Åre-Östersund | 1.5 MW | 0.04 MW |

### Anomaly Detection (2024 test set)

| City | Heating F1 | Cooling F1 |
|---|---|---|
| Gothenburg | 0.068 | 0.364 |
| Kiruna | 0.191 | 0.123 |
| Malmö | 0.142 | 0.138 |
| Stockholm | 0.069 | 0.198 |
| Umeå | 0.054 | **0.672** |
| Uppsala | 0.101 | 0.294 |
| Åre-Östersund | 0.084 | 0.000* |

*Åre-Östersund has zero active cooling months — insufficient signal for detection.

Heating F1 is constrained by the short duration of injected anomalies (1–3h) relative to conformal threshold sensitivity. The mathematically guaranteed FPR ≤ 5% is maintained. Umeå cooling at F1=0.672 with Precision=0.920 is the standout result.

---

## Project Structure

```
sy1-nordic-energy-ops/
├── src/
│   ├── ingestion/          # Data pipeline: SMHI, grid generator, demand generator
│   ├── forecasting/        # XGBoost, LSTM, feature engineering
│   ├── anomaly/            # Conformal prediction detectors
│   ├── gnn/                # GraphSAGE fault localisation
│   ├── rag/                # Knowledge base builder and HyDE retriever
│   ├── agents/             # 6-agent pipeline with Intent Router
│   ├── api/                # FastAPI REST backend (10 endpoints)
│   └── dashboard/          # Streamlit operator dashboard (3 tabs)
├── data/
│   ├── raw/                # SMHI weather data (gitignored)
│   ├── processed/          # Demand, forecasts, detection results (gitignored)
│   ├── synthetic/          # Grid topology (gitignored)
│   └── rag_docs/           # Knowledge base documents
├── models/
│   └── saved/              # Trained models (gitignored)
├── environment.yml
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/SushilKumar6608/sy1-nordic-energy-ops.git
cd sy1-nordic-energy-ops

# 2. Create conda environment
conda env create -f environment.yml
conda activate sy1-nordic-energy-ops

# 3. Add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 4. Run the data pipeline
python src/ingestion/smhi_fetcher.py
python src/ingestion/grid_generator.py
python src/ingestion/demand_generator.py

# 5. Train forecasting models
python src/forecasting/train_xgboost.py

# 6. Run anomaly detection
python src/anomaly/conformal_detector.py

# 7. Train GNN
python src/gnn/graph_sage.py

# 8. Build knowledge base
python src/rag/build_knowledge_base.py

# 9. Start the system (two terminals)
python src/api/main.py          # Terminal 1 — FastAPI on port 8000
streamlit run src/dashboard/app.py  # Terminal 2 — Streamlit on port 8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health check |
| GET | `/grid/cities` | List monitored cities |
| GET | `/grid/state/{city}` | City demand statistics and detection performance |
| GET | `/grid/demand/{city}` | Hourly demand time series with anomaly labels |
| GET | `/grid/anomaly-summary` | Network-wide detection performance table |
| POST | `/gnn/localise` | GraphSAGE fault node localisation |
| POST | `/rag/query` | Knowledge base query with HyDE expansion |
| POST | `/rag/upload` | Ingest new document into live knowledge base |
| POST | `/agent/diagnose` | Full 6-agent diagnosis pipeline |
| GET | `/models/forecasting-results` | XGBoost model metrics |

Interactive API docs: `http://localhost:8000/docs`

---

## Dashboard

Three-tab Streamlit interface:

**Tab 1 — Grid Overview**: City/month selector, heating and cooling metrics, conformal calibration baseline badge (STABLE / DATA DRIFT WARNING based on temperature delta vs 2023 calibration set), network-wide F1 performance bar chart.

**Tab 2 — Demand Explorer**: Hourly demand chart with dynamic conformal uncertainty bands (semi-transparent ribbon that widens during volatile morning ramp-up and contracts overnight), temperature overlay, red anomaly markers.

**Tab 3 — Agent Ops Centre**: Anomaly event form with all fault flags, natural language query field routed through Intent Router, full incident report output with GNN fault nodes, RAG confidence tier, HyDE audit trail, and downloadable markdown report. Sidebar includes live document ingestion uploader.

---

## Methodological Notes

**Why XGBoost beats LSTM here**: Engineered lag features (t-1h, t-6h, t-24h, t-168h) already encode the temporal dependencies that LSTM tries to learn from raw sequences. XGBoost exploits these directly and efficiently. This is a documented pattern in energy demand forecasting with structured weather covariates.

**Why heating F1 is low**: Conformal prediction guarantees FPR ≤ alpha (5%), not high precision. Heating anomalies last 1–3 hours — at the 95th percentile threshold, normal forecast errors occasionally exceed the band producing false positives. The guarantee holds; the operational trade-off is documented.

**Why cooling uses a hybrid approach**: Sweden has near-zero cooling demand for 8–9 months. Sigma normalisation becomes unstable during zero-demand periods. The hybrid strategy uses normalised scores for active months and raw residual thresholds for winter shutdown months, preventing both false positive floods and false negative droughts.

---

## Stack

**ML/AI**: PyTorch · XGBoost · scikit-learn · PyTorch Geometric · FAISS · sentence-transformers · Anthropic Claude Sonnet

**Data**: SMHI Open Data API · pandas · numpy · scipy

**Backend**: FastAPI · uvicorn · Pydantic

**Frontend**: Streamlit · Plotly

**Infrastructure**: conda · Python 3.11

---

## Author

**Sushil Kumar**-MSc AI & Automation, University West Sweden (2026)

GitHub: [github.com/SushilKumar6608](https://github.com/SushilKumar6608)
