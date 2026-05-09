# MakFleet — Semantic-Aware Spatio-Temporal Data Warehouse

**BIS 3205 Doctoral-Level Project**
A Spatio-Temporal Knowledge Graph (ST-KG) for bodaboda (motorcycle taxi) fleet management on Makerere University campus, Uganda.

---

## What This Project Does

MakFleet replaces the traditional Kimball Star Schema with an **Ontology-Driven Knowledge Graph** that natively supports graph traversal, semantic event classification, and causal anomaly explanation. The system:

- Pulls the **real Makerere University road network** from OpenStreetMap
- Simulates realistic bodaboda GPS + accelerometer telemetry across two semesters
- Map-matches noisy GPS to actual road edges using a Hidden Markov Model
- Classifies telemetry into **semantic events** (e.g. `harsh_braking` vs `safe_stop` near a zebra crossing)
- Loads everything into **Neo4j** as a Spatio-Temporal Knowledge Graph
- Trains a custom **ST-GNN** (GCN spatial encoder + LSTM temporal aggregator) for anomaly detection
- Serves a **Streamlit dashboard** with causal evidence generation, heatmaps, and benchmark comparisons
- Enforces **Privacy by Design** via spatial k-anonymity, differential privacy, and RBAC

---

## 8-Layer Locator Framework

| Layer | Requirement | Implementation |
|-------|-------------|----------------|
| L0 | Operational safety + spatio-temporal optimization | `ontology/`, `config/settings.py` |
| L1 | Weak supervision — IoT telematics + geospatial mapping | `simulator.py` + `campus_graph.py` |
| L2 | Anomaly detection + temporal forecasting | `semantic_enricher.py` + `stgnn.py` |
| L3 | Multivariate time-series + spatial graph data | `graph_dataset.py` PyG snapshots |
| L4 | Online/continual learning — semester concept drift | `trainer.py` semester-split protocol |
| L5 | Privacy-preserving + OOD robustness | `privacy.py` + spatial hold-out split |
| L6 | Edge-to-cloud streaming into semantic warehouse | `neo4j_loader.py` batch pipeline |
| L7 | ST-GNN + rule-based symbolic models (neuro-symbolic) | `stgnn.py` + `semantic_enricher.py` |

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| Docker Desktop | Latest | Neo4j container |
| Git | Any | Version control |

**Already installed (if you ran this before):**
`torch`, `torch_geometric`, `neo4j`, `networkx`, `numpy`, `pandas`, `scipy`, `scikit-learn`

---

## Quick Start (First Time)

### 1. Clone / open the project

```bash
cd C:/Users/sebun/OneDrive/Desktop/datawarehousefinal
```

### 2. Install Python dependencies

```bash
pip install osmnx folium streamlit plotly diffprivlib geopandas shapely matplotlib
```

Or install everything from requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Start Neo4j via Docker

```bash
docker-compose up -d
```

Wait ~15 seconds for Neo4j to initialise, then verify it is running:

```bash
docker ps
```

You should see `makfleet-neo4j` in the list. The Neo4j browser is at **http://localhost:7474** (login: `neo4j` / `makfleet2024`).

Alternatively, start Neo4j without docker-compose:

```bash
docker run -d --name makfleet-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/makfleet2024 \
  neo4j:5
```

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

This runs all phases in order:
- Downloads Makerere campus road network from OpenStreetMap (~2 min, cached after first run)
- Simulates 15 bodabodas × 2 semesters of telemetry
- Map-matches GPS traces to road edges
- Classifies telemetry into semantic events
- Loads everything into Neo4j
- Trains the ST-GNN model (~10–20 min on CPU)

To skip model training (dashboard still works via rule-based fallback), press **Ctrl+C** after you see `Loading Semester 2 events…`.

### 5. Launch the dashboard

```bash
streamlit run phase4_dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## Running Phases Individually

If you need to re-run only one part:

```bash
# Re-download campus map (clears cache)
python -c "from phase1_ingestion.campus_graph import load_campus_graph; load_campus_graph()"

# Re-run simulator only
python phase1_ingestion/simulator.py

# Re-run full pipeline minus training
python run_pipeline.py   # then Ctrl+C when training starts

# Train the model only (requires Neo4j to be loaded)
python phase3_model/trainer.py

# Run benchmarks standalone
python benchmarks/star_schema_benchmark.py

# Start dashboard
streamlit run phase4_dashboard/app.py
```

---

## Project Structure

```
datawarehousefinal/
│
├── run_pipeline.py              # ONE-COMMAND pipeline runner (start here)
├── requirements.txt
├── docker-compose.yml
│
├── config/
│   └── settings.py              # ALL constants — import from here everywhere
│                                # (bbox, Neo4j creds, simulation params, thresholds)
│
├── ontology/
│   └── makfleet_ontology.py     # OWL-style node labels, relationship types,
│                                # severity scores, road type encodings
│
├── data/                        # Generated at runtime — do not commit to git
│   ├── raw/
│   │   ├── sem1_telemetry.jsonl # Semester 1 raw GPS + accelerometer records
│   │   ├── sem2_telemetry.jsonl # Semester 2 records
│   │   └── fleet_meta.json      # Vehicle and driver definitions
│   ├── processed/
│   │   └── enriched_events.jsonl # Map-matched + semantically enriched records
│   ├── graph_cache/
│   │   └── makerere_campus.graphml  # Cached OSMnx road network (avoids re-download)
│   ├── stgnn_checkpoint.pt      # Best ST-GNN model weights (created by trainer.py)
│   ├── eval_results.json        # Temporal + spatial OOD evaluation metrics
│   └── audit_log.jsonl          # Append-only RBAC access log
│
├── phase1_ingestion/
│   ├── campus_graph.py          # OSMnx → NetworkX graph, landmark annotation,
│   │                            # centrality features, GraphML cache
│   ├── provenance.py            # SHA-256 per-record hashing + Merkle chain
│   ├── simulator.py             # Synthetic bodaboda telemetry generator
│   │                            # (GPS noise, accelerometer injection, semester splits)
│   └── neo4j_loader.py          # Batch Cypher loader — enforces ContextualStop label
│
├── phase2_pipeline/
│   ├── map_matcher.py           # HMM map-matching (KD-tree + Viterbi sliding window)
│   ├── semantic_enricher.py     # Rule-based event classification:
│   │                            # harsh_braking / safe_stop / speeding / idling etc.
│   └── privacy.py               # Spatial k-anonymity (DBSCAN), differential privacy
│                                # (diffprivlib Gaussian), RBAC, audit logging
│
├── phase3_model/
│   ├── graph_dataset.py         # Neo4j subgraphs → PyG Data objects
│   │                            # Semester-based + spatial OOD splits
│   ├── stgnn.py                 # Custom ST-GNN: GCNConv encoder + LSTM + classifier
│   ├── trainer.py               # Training loop, CosineAnnealingLR, early stopping,
│   │                            # OOD evaluation (temporal + spatial)
│   └── anomaly_scorer.py        # NeuralScorer + RuleBasedScorer fallback (Gap 3)
│                                # load_scorer() auto-selects based on checkpoint
│
├── phase4_dashboard/
│   ├── app.py                   # Streamlit app — 4 tabs (Map, Anomalies,
│   │                            # Causal Evidence, Benchmarks)
│   ├── causal_evidence.py       # Neo4j Cypher for 2-hop subgraph retrieval
│   ├── map_view.py              # Folium helpers (heatmap, trajectory, subgraph map)
│   └── rbac.py                  # Role permissions + audit log writer
│
└── benchmarks/
    └── star_schema_benchmark.py # Pandas Star Schema vs Neo4j Cypher
                                 # 5 queries × 10 runs × statistics.median()
```

---

## Knowledge Graph Schema

The ST-KG has three ontological domains enforced at the Neo4j storage layer:

### Node Labels

| Label | Domain | Key Properties |
|-------|--------|----------------|
| `Intersection` | Spatial | `osm_id`, `lat`, `lon`, `betweenness`, `degree_centrality` |
| `RoadSegment` | Spatial | `segment_id`, `length_m`, `road_type`, `maxspeed` |
| `BodaBodaStop` | Spatial | (sub-label of Intersection) |
| `ZebraCrossing` | Spatial | (sub-label of Intersection) |
| `Vehicle` | Fleet | `vehicle_id`, `plate` |
| `Driver` | Fleet | `driver_id`, `name`, `risk_profile`, `is_reckless` |
| `TelematicsEvent` | Temporal | `event_id`, `event_type`, `severity_score`, `is_anomaly`, `timestamp`, `speed_kmh`, `ax` |
| `ContextualStop` | Temporal | Same as TelematicsEvent — **distinct label**, never returned by `MATCH (n:TelematicsEvent)` |

> **Important:** `ContextualStop` is a first-class label, not a string property. This is the core semantic-awareness distinction — safe stops near zebra crossings are ontologically separate from anomalous events.

### Relationship Types

| Relationship | Direction | Meaning |
|---|---|---|
| `ADJACENT_TO` | Intersection → Intersection | Road connection (bidirectional) |
| `LOCATED_AT` | TelematicsEvent/ContextualStop → Intersection | Where event occurred |
| `CAUSED_BY` | TelematicsEvent → Driver | Who was driving |
| `DRIVES` | Driver → Vehicle | Fleet assignment |
| `PRECEDES` | TelematicsEvent → TelematicsEvent | Trajectory ordering |

### Useful Cypher Queries

```cypher
// Count all nodes by label
MATCH (n) RETURN labels(n) AS label, count(n) AS count ORDER BY count DESC

// All anomalous events for a specific driver
MATCH (e:TelematicsEvent)-[:CAUSED_BY]->(d:Driver {driver_id: 'D001'})
WHERE e.is_anomaly = true
RETURN e ORDER BY e.timestamp DESC LIMIT 20

// Query BOTH TelematicsEvent and ContextualStop
MATCH (n) WHERE n:TelematicsEvent OR n:ContextualStop
RETURN labels(n), count(n)

// 2-hop neighbours of the Main Gate bodaboda stop
MATCH (stop:BodaBodaStop {landmark_name: 'Main Gate'})-[:ADJACENT_TO*1..2]-(n:Intersection)
RETURN n.osm_id, n.lat, n.lon

// Causal subgraph for a flagged event
MATCH (e:TelematicsEvent {event_id: 'YOUR-EVENT-ID'})
MATCH (e)-[:LOCATED_AT]->(i:Intersection)
MATCH (e)-[:CAUSED_BY]->(d:Driver)-[:DRIVES]->(v:Vehicle)
MATCH (i)-[:ADJACENT_TO*1..2]-(neighbor:Intersection)
RETURN e, i, d, v, collect(neighbor)

// Reckless drivers near safe zones (semantic query)
MATCH (e:TelematicsEvent)-[:CAUSED_BY]->(d:Driver)
MATCH (e)-[:LOCATED_AT]->(i:Intersection)
WHERE e.is_anomaly = true AND d.risk_profile > 0.7
  AND (i:ZebraCrossing OR i:BodaBodaStop)
RETURN d.name, count(e) AS incidents ORDER BY incidents DESC
```

---

## Configuration Reference

All constants are in `config/settings.py`. Change them there — never hardcode values in other files.

| Constant | Default | What it controls |
|----------|---------|-----------------|
| `MAKERERE_BBOX` | `(0.3340, 32.5680, 0.3420, 32.5760)` | Campus bounding box for OSMnx |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_PASSWORD` | `makfleet2024` | Neo4j auth |
| `NUM_VEHICLES` | `15` | Bodabodas to simulate |
| `SIM_DAYS_SEM1` | `90` | Days in Semester 1 (training data) |
| `SIM_DAYS_SEM2` | `60` | Days in Semester 2 (test data) |
| `GPS_NOISE_STD` | `0.00005` | GPS drift (~5 metres) |
| `ACCEL_HARSH_BRAKE_THRESHOLD` | `-3.5` | m/s² threshold for harsh braking |
| `RECKLESS_BRAKE_RATE` | `0.05` | 5% of records injected with harsh braking |
| `PRIVACY_K` | `5` | k-anonymity minimum cluster size |
| `PRIVACY_EPSILON` | `1.0` | Differential privacy budget |
| `WINDOW_SIZE_MIN` | `15` | Minutes per ST-GNN graph snapshot |
| `T_SNAPSHOTS` | `4` | Number of snapshots per training sample |
| `SPATIAL_OOD_LON` | `32.574` | Eastern campus boundary for OOD split |
| `STGNN_EPOCHS` | `50` | Maximum training epochs |
| `BENCHMARK_RUNS` | `10` | Timing repetitions per benchmark query |

---

## ST-GNN Architecture

```
Input: T=4 graph snapshots (each 15-minute window)
       Node features X_t ∈ R^{N×6}:
         [lat, lon, betweenness, degree_centrality, event_count, mean_speed]

Spatial Encoder (shared weights, applied to each snapshot independently):
  GCNConv(6 → 64) → ReLU
  GCNConv(64 → 64) → ReLU
  Output: H_t ∈ R^{N×64}

Temporal Aggregator (per node, across all T snapshots):
  Stack: H ∈ R^{N × T × 64}
  LSTM(input=64, hidden=64, layers=2, batch_first=True)
  Take last hidden state: h_final ∈ R^{N×64}

Output Head:
  Linear(64 → 32) → ReLU → Linear(32 → 1) → Sigmoid
  Output: anomaly probability per node ∈ [0, 1]

Loss: Weighted BCE (pos_weight=19.0 for ~5% anomaly rate)
Optimizer: Adam (lr=1e-3) + CosineAnnealingLR
```

This is architecturally equivalent to A3T-GCN but implemented from first principles using `torch_geometric.nn.GCNConv` and `torch.nn.LSTM` — every design decision is explicit and auditable.

---

## Data Splits (No Leakage)

Random train/test splits are **strictly forbidden** as they leak future routing behaviour into training.

| Split | Data | Purpose |
|-------|------|---------|
| Train | Semester 1, first 80% of windows | Model learning |
| Validation | Semester 1, last 20% of windows | Early stopping |
| Temporal OOD Test | Semester 2 (all windows) | Tests generalisation across concept drift |
| Spatial OOD Test | Eastern campus (lon > 32.574) | Tests geographic generalisation |

---

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| Campus Map | Folium heatmap of trip demand + anomaly event markers. Role-filtered. |
| Anomaly Detection | Hourly time-series of event types + full flagged events table |
| Causal Evidence | Select any anomaly → see 2-hop causal subgraph, driver history, explanation of WHY it was flagged. Works without trained model (rule-based fallback). |
| Benchmarks | ST-KG vs Star Schema: 5 queries × 10 runs × median latency |

### Dashboard Access Roles

| Role | What you can see |
|------|-----------------|
| `admin` | Everything: raw GPS, driver names, full trajectories, audit log |
| `analyst` | Anonymized trajectories, event types, causal evidence (no driver identity) |
| `public` | k-anonymized heatmaps and aggregates only |

---

## Privacy Architecture

| Mechanism | Where | What it does |
|-----------|-------|-------------|
| SHA-256 provenance hash | `provenance.py` | Every record hashed at ingestion — tamper detection |
| Merkle chain | `provenance.py` | Batch-level root hash stored in Neo4j as `ProvenanceAnchor` |
| Spatial k-anonymity | `privacy.py` | GPS clusters with < 5 members replaced by centroid |
| Differential privacy | `privacy.py` | Gaussian noise on aggregate demand statistics (ε=1.0) |
| RBAC | `rbac.py` | Three-tier access control enforced per resource |
| Audit log | `data/audit_log.jsonl` | Append-only log of all privileged data access |

---

## Common Issues

**Neo4j connection refused**
```bash
docker ps                        # check container is running
docker-compose up -d             # start if not running
docker logs makfleet-neo4j       # check for startup errors
```

**OSMnx download fails / times out**
The campus graph is cached after the first successful download at `data/graph_cache/makerere_campus.graphml`. If download fails mid-way, delete that file and retry:
```bash
del data\graph_cache\makerere_campus.graphml
python run_pipeline.py
```

**Dashboard crashes on startup**
The dashboard is designed to never crash. If it does, check:
```bash
pip install streamlit folium plotly   # missing dashboard deps
python -c "from phase4_dashboard.app import *"   # import check
```

**ST-GNN training is too slow**
Press Ctrl+C to skip training. The dashboard automatically falls back to the rule-based scorer and all 4 tabs still work. Train later with:
```bash
python phase3_model/trainer.py
```

**"No anomaly data" in dashboard**
The Neo4j loader must complete before the dashboard shows data. Re-run:
```bash
python run_pipeline.py
```
Then check Neo4j browser: `MATCH (n) RETURN count(n)` should return > 1000.

---

## Continuing Development

### Adding a new campus landmark

In `config/settings.py`, add an entry to the `LANDMARKS` dict:
```python
LANDMARKS = {
    ...
    "New Building Name": (lat, lon, "BodaBodaStop"),  # or "ZebraCrossing"
}
```
Then re-run `campus_graph.py` (delete the cache file first).

### Adding a new semantic event type

1. Add the event type to `ontology/makfleet_ontology.py` in `SEVERITY` and `IS_ANOMALY`
2. Add classification logic to `phase2_pipeline/semantic_enricher.py` in `enrich_vehicle()`
3. If it should be a distinct Neo4j label (like `ContextualStop`), add a new branch in `phase1_ingestion/neo4j_loader.py` in `load_telemetry_events()`

### Changing the ST-GNN architecture

Edit `phase3_model/stgnn.py`. The `STGNNModel` class is self-contained. After changes, retrain:
```bash
python phase3_model/trainer.py
```

### Using real GPS data

Replace `phase1_ingestion/simulator.py` with a loader that reads your actual data source. The output schema must match:
```python
{
    "event_id": str,       # uuid
    "vehicle_id": str,
    "driver_id": str,
    "timestamp": str,      # ISO 8601
    "lat": float,
    "lon": float,
    "speed_kmh": float,
    "heading_deg": float,
    "ax": float,           # accelerometer x-axis (m/s²)
    "ay": float,
    "az": float,
    "engine_state": str,   # "ON" or "IDLE"
    "edge_u": int,         # OSM node IDs (can be 0 if unknown)
    "edge_v": int,
    "semester": int,       # 1 or 2
}
```
The rest of the pipeline (map-matching, enrichment, Neo4j loading) works unchanged.

### Scaling up (more vehicles, longer simulation)

In `config/settings.py`:
```python
NUM_VEHICLES = 50          # increase fleet size
SIM_DAYS_SEM1 = 120        # longer semester
NUM_TRIPS_PER_VEHICLE_PER_DAY = 12
```
Neo4j batch size is controlled by `BATCH_SIZE = 500` in `neo4j_loader.py` — increase if loading is slow.

---

## File Dependency Order

When modifying files, understand what depends on what:

```
config/settings.py              ← everything imports from here
ontology/makfleet_ontology.py   ← neo4j_loader, semantic_enricher, anomaly_scorer

campus_graph.py                 ← simulator, map_matcher, semantic_enricher
provenance.py                   ← simulator, neo4j_loader
simulator.py                    ← run_pipeline
map_matcher.py                  ← run_pipeline (after simulator)
semantic_enricher.py            ← run_pipeline (after map_matcher)
privacy.py                      ← run_pipeline, app.py
neo4j_loader.py                 ← run_pipeline (after enricher)

graph_dataset.py                ← trainer (requires Neo4j loaded)
stgnn.py                        ← trainer, anomaly_scorer
trainer.py                      ← run_pipeline
anomaly_scorer.py               ← app.py (loaded at dashboard startup)

causal_evidence.py              ← app.py (requires Neo4j)
rbac.py                         ← app.py, privacy.py
map_view.py                     ← app.py
app.py                          ← streamlit entry point

star_schema_benchmark.py        ← app.py Tab 4 (requires Neo4j + pandas)
```

---

## Tech Stack Summary

| Component | Technology | Why |
|-----------|-----------|-----|
| Campus road network | OSMnx + OpenStreetMap | Real Makerere topology, no synthetic map needed |
| Knowledge Graph | Neo4j 5 | Native graph traversal, Cypher, sub-second spatial queries |
| Graph ML | PyTorch Geometric | GCNConv + LSTM from primitives, auditable architecture |
| Map-matching | Custom HMM (scipy KDTree + Viterbi) | Correct algorithm for GPS-to-road snapping |
| Privacy | diffprivlib + scikit-learn DBSCAN | Calibrated DP noise + k-anonymity |
| Dashboard | Streamlit + Folium + Plotly | Rapid deployment, interactive maps, no JS required |
| Containerisation | Docker + docker-compose | One-command Neo4j setup |
