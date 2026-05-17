# MakFleet — Semantic-Aware Spatio-Temporal Data Warehouse

> A doctoral-level Spatio-Temporal Knowledge Graph for bodaboda (motorcycle taxi) fleet safety management on Makerere University campus, Uganda.

**Live Dashboard:** https://makfleet-9h6sbm5jgt3wpn3nsil4rk.streamlit.app  
**Dataset:** https://www.kaggle.com/datasets/sebunyaronaldo/makfleet-bodaboda-telemetry-makerere-university  
**Code:** https://github.com/sebunyaronaldo/makfleet

---

## What is MakFleet?

MakFleet replaces a traditional relational Star Schema data warehouse with an **Ontology-Driven Spatio-Temporal Knowledge Graph (ST-KG)** that:

- Ingests IoT telematics from bodabodas at 5-second intervals
- Applies **Hidden Markov Model map-matching** to snap noisy GPS to real campus roads
- Enriches raw sensor data into **semantic events** (`harsh_braking`, `speeding`, `safe_stop` etc.)
- Stores everything in **Neo4j** as a connected knowledge graph
- Runs a custom **GCN + LSTM Spatio-Temporal Graph Neural Network** for anomaly detection
- Serves a **Streamlit causal BI dashboard** that explains *why* a driver was flagged

The key insight: a sudden deceleration near a mapped zebra crossing is a `ContextualStop` (safe). The same signal on an open road is `harsh_braking` (anomalous). Only a semantically-aware graph architecture can make this distinction.

---

## Architecture Overview

```
IoT Telematics (GPS + Accelerometer + Speed)
         │  5-second stream
         ▼
┌─────────────────────────────┐
│  Phase 1: Ingestion         │
│  • SHA-256 provenance hash  │
│  • Merkle chain per batch   │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│  Phase 2: Semantic Pipeline │
│  • HMM Map-Matching         │  ← snaps GPS to OSM road edges
│  • Rule-based enrichment    │  ← classifies into 6 event types
│  • k-Anonymity + Diff. Priv │  ← privacy before KG ingestion
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│  Neo4j Knowledge Graph      │
│  212 intersections          │
│  476 road segments          │
│  826,260 telemetry events   │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│  ST-GNN (GCN + LSTM)        │
│  AUC = 0.9471 (OOD test)    │
│  Recall = 1.0 (all splits)  │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│  Streamlit Dashboard        │
│  • Campus heatmap           │
│  • Anomaly detection        │
│  • Causal evidence (WHY)    │
└─────────────────────────────┘
```

---

## Project Structure

```
makfleet/
│
├── run_pipeline.py              # Run everything in one command
├── reload_neo4j.py              # Reload Neo4j from disk (no re-simulation)
├── requirements.txt
├── docker-compose.yml           # Neo4j local setup
│
├── config/
│   └── settings.py              # All constants — edit here only
│
├── ontology/
│   └── makfleet_ontology.py     # Node labels, relationship types, severity scores
│
├── data/
│   ├── raw/                     # Simulated JSONL telemetry (Sem1, Sem2)
│   ├── processed/               # Enriched events after semantic pipeline
│   ├── graph_cache/             # Cached OSMnx campus road network (GraphML)
│   └── stgnn_checkpoint.pt      # Trained ST-GNN model weights
│
├── phase1_ingestion/
│   ├── campus_graph.py          # Downloads Makerere campus from OpenStreetMap
│   ├── simulator.py             # Generates synthetic bodaboda telemetry
│   ├── provenance.py            # SHA-256 hashing + Merkle chain
│   └── neo4j_loader.py          # Batch loads data into Neo4j
│
├── phase2_pipeline/
│   ├── map_matcher.py           # HMM map-matching (KD-tree + Viterbi)
│   ├── semantic_enricher.py     # Classifies events into semantic types
│   └── privacy.py               # k-Anonymity, differential privacy, RBAC
│
├── phase3_model/
│   ├── graph_dataset.py         # Builds PyG dataset from Neo4j
│   ├── stgnn.py                 # GCNConv + LSTM architecture
│   ├── trainer.py               # Training with semester-based splits
│   └── anomaly_scorer.py        # NeuralScorer + RuleBasedScorer fallback
│
├── phase4_dashboard/
│   ├── app.py                   # Streamlit app (3 tabs)
│   ├── causal_evidence.py       # Neo4j causal subgraph retrieval
│   ├── map_view.py              # Folium map rendering helpers
│   └── rbac.py                  # Role-based access control
│
└── benchmarks/
    └── star_schema_benchmark.py # ST-KG vs Star Schema query comparison
```

---

## Knowledge Graph Schema

### Node Labels

| Label | Count | Description |
|---|---|---|
| `Intersection` | 212 | Campus road junction with lat/lon, betweenness centrality |
| `RoadSegment` | 476 | Campus road edge with length, type, speed limit |
| `BodaBodaStop` | 6 | Sub-label of Intersection — known pick-up points |
| `ZebraCrossing` | 3 | Sub-label of Intersection — pedestrian crossings |
| `Vehicle` | 15 | Bodaboda unit with plate identifier |
| `Driver` | 15 | Driver with pseudonymised name and risk profile (0–1) |
| `TelematicsEvent` | 826,260 | Enriched IoT record — event type, severity, anomaly flag, SHA-256 hash |
| `ContextualStop` | — | **Distinct label** from TelematicsEvent — safe stops near zebra crossings |
| `ProvenanceAnchor` | 2 | Merkle root hash per semester for tamper detection |

> **Critical:** `MATCH (n:TelematicsEvent)` will **never** return a `ContextualStop` node. This ontological separation is enforced at the storage layer.

### Relationship Types

| Relationship | Meaning |
|---|---|
| `ADJACENT_TO` | Intersection ↔ Intersection (road connection) |
| `LOCATED_AT` | TelematicsEvent/ContextualStop → Intersection |
| `CAUSED_BY` | TelematicsEvent → Driver |
| `DRIVES` | Driver → Vehicle |
| `PRECEDES` | TelematicsEvent → TelematicsEvent (trajectory order) |

---

## Semantic Event Types

| Event Type | Anomaly | Condition |
|---|---|---|
| `normal_travel` | No | Speed within limit, smooth acceleration |
| `speeding` | **Yes** | Speed > 30 km/h × 1.2 |
| `harsh_braking` | **Yes** | ax < −3.5 m/s² sustained 2+ ticks, not near safe zone |
| `rapid_acceleration` | **Yes** | ax > 2.8 m/s² |
| `idling` | No | Speed < 2 km/h for > 30 seconds |
| `safe_stop` | No | Idle within 20m of BodaBodaStop or ZebraCrossing → loaded as `ContextualStop` |

---

## ST-GNN Model

```
Input: T=4 graph snapshots (15-minute windows)
       Node features X ∈ R^{N×6}:
         [lat, lon, betweenness, degree_centrality, event_count, mean_speed]

Spatial Encoder (per snapshot):
  GCNConv(6→64) → ReLU → GCNConv(64→64) → ReLU
  Output: H_t ∈ R^{N×64}

Temporal Aggregator (per node, across T snapshots):
  LSTM(input=64, hidden=64, layers=2)
  Output: h_final ∈ R^{N×64}

Output Head:
  Linear(64→32) → ReLU → Linear(32→1) → Sigmoid
  Anomaly probability per node ∈ [0,1]

Loss: Weighted BCE (pos_weight=19.0)
Optimizer: Adam (lr=1e-3) + CosineAnnealingLR
```

### Evaluation Results

| Split | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Validation (Sem 1 held-out) | 0.9268 | 0.9067 | 0.8293 | **1.000** |
| Temporal OOD (Semester 2) | **0.9471** | 0.9045 | 0.8256 | **1.000** |
| Spatial OOD (Eastern campus) | —† | 1.000 | 1.000 | **1.000** |

†AUC undefined — all samples in eastern zone are positive class.

**Key finding:** Temporal OOD AUC (0.9471) > Validation AUC (0.9268) — the model generalises across semester concept drift rather than memorising Semester 1 patterns.

---

## Dashboard — How It Works

Open **http://localhost:8501** (local) or the live URL above.

### Tab 1 — Campus Map
- **Heatmap** of trip demand across Makerere campus interior roads (colour: blue → red by density)
- **Anomaly markers** clustered by location — click any marker for event details
- **Role filter** in sidebar: admin sees raw GPS, analyst sees anonymised, public sees heatmap only

### Tab 2 — Anomaly Detection
- **Hourly time-series** of anomaly rates by event type
- **Severity distribution** across all flagged events
- **Events table** — filterable by vehicle, semester, event type

### Tab 3 — Causal Evidence *(core contribution)*
- Select any flagged event from the dropdown
- System retrieves the **2-hop causal subgraph** from Neo4j via `ADJACENT_TO` traversal
- **Folium map** renders: red = anomaly node, green = safe zones, blue = adjacent intersections
- **Plotly network diagram** shows the graph structure with edge labels
- **Written explanation** generated: e.g. *"Harsh braking at −4.2 m/s² at 47 km/h. No zebra crossing found within 20m in campus ontology — classified as reckless."*
- **Driver history** shows all prior anomalies for the same driver

### Sidebar Controls
- **Access Role**: admin / analyst / public — controls data visibility
- **Semester filter**: All / Semester 1 / Semester 2
- **Vehicle filter**: All or individual vehicle
- **Neo4j status**: 🟢 Connected / 🟡 Unavailable (Aura paused)
- **Scorer badge**: ST-GNN (neural) or Rule-based (fallback)

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Docker Desktop

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Neo4j
```bash
docker-compose up -d
```
Neo4j browser available at **http://localhost:7474** (login: `neo4j` / `makfleet2024`)

### 3. Run the full pipeline
```bash
python run_pipeline.py
```
This will:
- Download the Makerere campus road network from OpenStreetMap (~2 min, cached after first run)
- Simulate 15 bodabodas across 2 semesters (826,260 records)
- Run HMM map-matching and semantic enrichment
- Load everything into Neo4j
- Train the ST-GNN (10–20 min on CPU — press Ctrl+C to skip, dashboard still works)

### 4. Launch the dashboard
```bash
streamlit run phase4_dashboard/app.py
```
Open **http://192.168.x.x:8501** (use the Network URL, not localhost, to avoid IPv6 issues on Windows)

---

## Running Individual Phases

```bash
# Re-run only Neo4j loading (data already simulated)
python reload_neo4j.py

# Train the ST-GNN only (Neo4j must be loaded)
python -m phase3_model.trainer

# Run benchmarks
python benchmarks/star_schema_benchmark.py
```

---

## Configuration

All constants live in `config/settings.py`. Key ones:

| Constant | Default | Description |
|---|---|---|
| `MAKERERE_BBOX` | `(0.3340, 32.5680, 0.3420, 32.5760)` | Campus bounding box |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | Neo4j connection (env var: `NEO4J_URI`) |
| `NEO4J_PASSWORD` | `makfleet2024` | Neo4j password (env var: `NEO4J_PASSWORD`) |
| `NUM_VEHICLES` | `15` | Fleet size for simulation |
| `GPS_NOISE_STD` | `0.00005` | ~5m GPS drift |
| `ACCEL_HARSH_BRAKE_THRESHOLD` | `-3.5` | m/s² threshold for harsh braking |
| `PRIVACY_K` | `5` | k-anonymity cluster minimum |
| `PRIVACY_EPSILON` | `1.0` | Differential privacy budget |
| `STGNN_EPOCHS` | `10` | Training epochs |

---

## Cloud Deployment

The live dashboard runs on **Streamlit Community Cloud** backed by **Neo4j Aura Free**.

### Neo4j Aura (Important!)
Neo4j Aura Free **auto-pauses after 3 days of inactivity**. If the dashboard shows a yellow badge or empty data:
1. Go to **https://console.neo4j.io**
2. Find instance `1a273157`
3. Click **Resume** — takes ~30 seconds

### Streamlit Secrets
The Streamlit Cloud app reads Neo4j credentials from secrets (set in the Streamlit Cloud dashboard):
```toml
NEO4J_URI      = "neo4j+s://1a273157.databases.neo4j.io"
NEO4J_USER     = "1a273157"
NEO4J_PASSWORD = "your-password"
```

### Deploying updates
```bash
git add .
git commit -m "Your changes"
git push
```
Streamlit Cloud auto-redeploys within ~2 minutes of a push.

---

## Privacy Architecture

| Mechanism | Where | What it does |
|---|---|---|
| SHA-256 provenance hash | `provenance.py` | Every record hashed at ingestion |
| Merkle chain | `provenance.py` | Batch-level tamper detection |
| Spatial k-anonymity (k=5) | `privacy.py` | GPS clusters with < 5 members replaced by centroid |
| Differential privacy (ε=1.0) | `privacy.py` | Gaussian noise on aggregate demand statistics |
| RBAC (3 tiers) | `rbac.py` | admin / analyst / public |
| Audit log | `data/audit_log.jsonl` | Append-only log of all privileged access |

---

## 8-Layer Locator Framework

| Layer | Requirement | Implementation |
|---|---|---|
| L0 | Operational safety + spatio-temporal optimisation | `ontology/`, `config/settings.py` |
| L1 | Weak supervision — IoT + geospatial mapping | `simulator.py` + `campus_graph.py` |
| L2 | Anomaly detection + temporal forecasting | `semantic_enricher.py` + `stgnn.py` |
| L3 | Multivariate time-series + spatial graph data | `graph_dataset.py` PyG snapshots |
| L4 | Continual learning — semester concept drift | `trainer.py` semester-split protocol |
| L5 | Privacy-preserving + OOD robustness | `privacy.py` + spatial OOD split |
| L6 | Edge-to-cloud streaming into semantic warehouse | `neo4j_loader.py` batch pipeline |
| L7 | ST-GNN + rule-based symbolic models (neuro-symbolic) | `stgnn.py` + `semantic_enricher.py` |

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| Dashboard yellow badge / empty data | Neo4j Aura paused | Resume at console.neo4j.io |
| `graph_from_bbox` error | OSMnx 2.x API change | Already fixed — uses `bbox=(west,south,east,north)` |
| `ConnectionResetError` in terminal | Windows IPv6/TCP cleanup | Harmless — use Network URL not localhost |
| Blank white page | Two Streamlit processes on same port | Kill all Python processes, restart |
| Neo4j OOM error | Low heap for large dataset | Increase heap in `docker-compose.yml` |
| Training too slow | CPU-only | Use GPU — full 50 epochs takes ~2 min on A100 |

---

## Team

| Name | Student No. | Role |
|---|---|---|
| Ddumba Jonah | 23/U/17979/EVE | System architecture, ST-GNN pipeline |
| Ssebwana John Paul | 23/U/17415/PS | Knowledge Graph schema, RBAC governance |
| Sebunya Ronaldo | 23/U/24496/PS | IoT simulator, EDA pipeline, dashboard |
| Namuli Evelyn | 11/U/24031/PS | Semantic enrichment, privacy module |

**Supervisor:** Dr. Ggaliwango Marvin  
**Department of Computer Science, COCIS, Makerere University**  
**BIS 3205 — Data Warehousing and Business Intelligence, April 2026**
