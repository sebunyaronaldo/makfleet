# MakFleet Live Analytics Dashboard (FastAPI + Railway)

Real-time bodaboda fleet analytics dashboard deployed on Railway.

**Live URL:** https://makfleet-production.up.railway.app

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI backend — serves API endpoints + SSE live positions |
| `static/index.html` | Frontend — dark-themed analytics dashboard |
| `requirements_live.txt` | Python dependencies for Railway deployment |
| `Procfile` | Railway startup command |

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the HTML dashboard |
| `GET /api/kpis` | KPI summary (total events, anomaly rate, top zone) |
| `GET /api/anomalies` | Recent anomalous events from Neo4j |
| `GET /api/demand` | Traffic demand by campus intersection |
| `GET /api/event-types` | Event type distribution |
| `GET /api/hourly` | Hourly anomaly breakdown |
| `GET /api/semester-comparison` | Semester 1 vs 2 stats |
| `GET /sse/positions` | Server-Sent Events stream of live vehicle positions |
| `GET /api/debug` | Neo4j connection diagnostics |

## Dashboard Tabs

1. **Overview** — KPI cards, hourly demand chart, live trip feed
2. **Spatial Analysis** — Campus heatmap with live moving bodaboda dots
3. **Temporal Patterns** — Weekly demand, semester comparison, intraday analysis
4. **Semantic Dimensions** — Event types, ontology tags, semantic event log
5. **DW Schema** — Constellation schema visualization
6. **OLAP Query** — Sample OLAP queries with simulated results

## Environment Variables (set in Railway)

| Variable | Value |
|---|---|
| `NEO4J_URI` | `neo4j+s://1a273157.databases.neo4j.io` |
| `NEO4J_PASSWORD` | Your Aura password |

## Local Development

```bash
cd fastapi_dashboard
pip install -r requirements_live.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Deployment

Push to `main` branch — Railway auto-deploys within ~2 minutes.

```bash
git push origin main
```

## Troubleshooting

**Dots not showing on map:**
Check Railway logs for `[tracker] Loaded X events from Neo4j`. If 0, Neo4j Aura may be paused — resume at console.neo4j.io.

**API returning zeros:**
Check `/api/debug` endpoint for Neo4j connection status.
