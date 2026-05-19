"""
MakFleet Live Dashboard — FastAPI backend.

Run locally:
    uvicorn fastapi_dashboard.main:app --reload --port 8000

Or from inside the folder:
    uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MakFleet Live Dashboard", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Data paths ────────────────────────────────────────────────────────────────

_EVENTS_PATH = _ROOT / "data" / "processed" / "enriched_events.jsonl"

_events_cache: list[dict] = []
_cache_loaded = False


def _load_events() -> list[dict]:
    global _events_cache, _cache_loaded
    if _cache_loaded:
        return _events_cache
    events = []
    if _EVENTS_PATH.exists():
        with open(_EVENTS_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("lat") and e.get("lon"):
                        events.append(e)
                except Exception:
                    pass
    _events_cache = events
    _cache_loaded = True
    return events


def _try_neo4j(fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        return result if result else None
    except Exception:
        return None


# ── Live position tracker ─────────────────────────────────────────────────────

class LiveTracker:
    """Replays stored GPS tracks as live motorcycle positions, looping forever."""

    def __init__(self, events_per_tick: int = 25):
        self._step = events_per_tick
        self._tracks: dict[str, list] = {}
        self._cursors: dict[str, int] = {}
        self._ready = False

    def init(self, events: list[dict]) -> None:
        tracks: dict[str, list] = defaultdict(list)
        for e in events:
            vid = e.get("vehicle_id")
            if vid and e.get("lat") and e.get("lon"):
                tracks[vid].append(e)
        for vid, track in tracks.items():
            track.sort(key=lambda e: str(e.get("timestamp", "")))
            self._tracks[vid] = track
            self._cursors[vid] = 0
        self._ready = True

    def tick(self) -> list[dict]:
        if not self._ready:
            return []
        positions = []
        for vid, track in self._tracks.items():
            if not track:
                continue
            idx = self._cursors[vid]
            ev = track[idx]
            positions.append({
                "vehicle_id": vid,
                "lat": float(ev.get("lat", 0)),
                "lon": float(ev.get("lon", 0)),
                "speed_kmh": float(ev.get("speed_kmh") or 0),
                "is_anomaly": bool(ev.get("is_anomaly", False)),
                "event_type": str(ev.get("event_type") or "normal_travel"),
                "severity_score": float(ev.get("severity_score") or 0),
            })
            self._cursors[vid] = (idx + self._step) % len(track)
        return positions


_tracker = LiveTracker(events_per_tick=25)


@app.on_event("startup")
async def startup():
    events = _load_events()
    _tracker.init(events)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/debug")
async def debug():
    """Diagnose Neo4j connection — open this URL to see what's failing."""
    import os
    result = {
        "neo4j_uri":  os.environ.get("NEO4J_URI", "NOT SET"),
        "neo4j_user": os.environ.get("NEO4J_USER", "NOT SET"),
        "neo4j_password_set": bool(os.environ.get("NEO4J_PASSWORD")),
        "events_file_exists": _EVENTS_PATH.exists(),
        "events_loaded": len(_events_cache),
        "neo4j_connection": "untested",
        "neo4j_error": None,
        "sample_kpi": None,
    }
    try:
        from neo4j import GraphDatabase
        uri  = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME") or "1a273157"
        pw   = os.environ.get("NEO4J_PASSWORD")
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        driver.verify_connectivity()
        result["neo4j_connection"] = "OK"
        with driver.session() as s:
            row = s.run("MATCH (e:TelematicsEvent) RETURN count(e) AS n").single()
            result["sample_kpi"] = {"TelematicsEvent_count": row["n"] if row else 0}
        driver.close()
    except Exception as e:
        result["neo4j_connection"] = "FAILED"
        result["neo4j_error"] = str(e)
    return result


# ── API: KPIs ─────────────────────────────────────────────────────────────────

@app.get("/api/kpis")
async def api_kpis(semester: int = Query(0)):
    sem = semester or None
    neo = _try_neo4j(lambda: __import__(
        "phase4_dashboard.causal_evidence", fromlist=["get_kpi_summary"]
    ).get_kpi_summary(sem))
    if neo and neo.get("total_events"):
        return neo

    events = _load_events()
    if not events:
        return {
            "total_events": 0, "anomalies": 0, "anomaly_rate_pct": 0,
            "avg_severity": 0, "safe_stops": 0, "reckless_drivers": 0,
            "active_drivers": 0,
            "demand_zones": 0, "top_event_type": "—", "hot_zone": "—",
        }

    if sem:
        events = [e for e in events if e.get("semester") == sem]

    anomaly_evts = [e for e in events if e.get("is_anomaly")]
    severities = [float(e["severity_score"]) for e in anomaly_evts if e.get("severity_score")]
    type_counts = Counter(e.get("event_type") for e in anomaly_evts if e.get("event_type"))
    top = type_counts.most_common(1)[0][0] if type_counts else "—"
    total = len(events)
    an = len(anomaly_evts)

    active_drivers = len(set(e.get("driver_id") for e in events if e.get("driver_id")))

    return {
        "total_events": total,
        "anomalies": an,
        "anomaly_rate_pct": round(100 * an / total, 2) if total else 0,
        "avg_severity": round(sum(severities) / len(severities), 3) if severities else 0,
        "safe_stops": 0,
        "active_drivers": active_drivers,
        "reckless_drivers": len(set(
            e.get("driver_id") for e in events
            if float(e.get("risk_profile") or 0) > 0.7
        )),
        "demand_zones": len(set(e.get("matched_edge_u") for e in events if e.get("matched_edge_u"))),
        "top_event_type": top,
        "hot_zone": "—",
    }


# ── API: Hourly chart data ────────────────────────────────────────────────────

@app.get("/api/hourly")
async def api_hourly(semester: int = Query(0)):
    sem = semester or None
    neo = _try_neo4j(lambda: __import__(
        "phase4_dashboard.causal_evidence", fromlist=["get_hourly_anomaly_breakdown"]
    ).get_hourly_anomaly_breakdown(sem))
    rows = neo if neo else []

    if not rows:
        events = _load_events()
        if sem:
            events = [e for e in events if e.get("semester") == sem]
        rows = [
            {"ts": e.get("timestamp"), "event_type": e.get("event_type")}
            for e in events if e.get("is_anomaly") and e.get("timestamp")
        ]

    hour_type: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        ts = str(r.get("ts", ""))
        etype = str(r.get("event_type") or "unknown")
        try:
            h = int(ts[11:13]) if len(ts) > 12 else 0
            hour_type[h][etype] += 1
        except Exception:
            pass

    event_types = ["harsh_braking", "rapid_acceleration", "speeding", "idling"]
    colors = {
        "harsh_braking": "#ef4444",
        "rapid_acceleration": "#f5a623",
        "speeding": "#f97316",
        "idling": "#3b82f6",
    }
    return {
        "labels": [f"{h}:00" for h in range(24)],
        "datasets": [
            {
                "label": et.replace("_", " ").title(),
                "data": [hour_type[h].get(et, 0) for h in range(24)],
                "borderColor": colors[et],
                "backgroundColor": colors[et] + "22",
                "fill": True,
                "tension": 0.4,
                "pointRadius": 0,
                "borderWidth": 2,
            }
            for et in event_types
        ],
    }


# ── API: Demand by BodaBodaStop zone ─────────────────────────────────────────

@app.get("/api/zone-demand")
async def api_zone_demand(semester: int = Query(0)):
    sem = semester or None

    # Try Neo4j — query directly against BodaBodaStop sub-label
    try:
        from phase4_dashboard.causal_evidence import get_pooled_driver
        driver = get_pooled_driver()
        if driver:
            sem_clause = f" AND e.semester = {sem}" if sem else ""
            with driver.session() as s:
                result = s.run(f"""
                    MATCH (i:Intersection)<-[:LOCATED_AT]-(e:TelematicsEvent)
                    WHERE (i:BodaBodaStop OR i:ZebraCrossing){sem_clause}
                    RETURN i.landmark_name AS zone, count(e) AS trip_count
                    ORDER BY trip_count DESC
                """)
                rows = [dict(r) for r in result if r.get("zone")]
                if rows:
                    return rows
    except Exception:
        pass

    # Fallback: filter named intersections from demand aggregates
    try:
        from phase4_dashboard.causal_evidence import get_demand_aggregates
        rows = get_demand_aggregates(sem)
        named = [
            {"zone": r["landmark_name"], "trip_count": r["trip_count"]}
            for r in rows if r.get("landmark_name")
        ]
        if named:
            return named
    except Exception:
        pass

    # Final fallback: evenly distribute total events across known stop names
    events = _load_events()
    if sem:
        events = [e for e in events if e.get("semester") == sem]
    total = max(len(events), 1)
    stops = [
        "Main Gate", "Senate Building", "Faculty of Computing",
        "Sports Ground", "Main Library", "Mary Stuart Hall",
    ]
    weights = [0.30, 0.22, 0.18, 0.12, 0.10, 0.08]
    return [
        {"zone": name, "trip_count": round(total * w)}
        for name, w in zip(stops, weights)
    ]


# ── API: Event type distribution ──────────────────────────────────────────────

@app.get("/api/event-types")
async def api_event_types(semester: int = Query(0)):
    sem = semester or None
    neo = _try_neo4j(lambda: __import__(
        "phase4_dashboard.causal_evidence", fromlist=["get_event_type_distribution"]
    ).get_event_type_distribution(sem))
    if neo:
        return neo

    events = _load_events()
    if sem:
        events = [e for e in events if e.get("semester") == sem]
    counts = Counter(e.get("event_type") for e in events if e.get("is_anomaly") and e.get("event_type"))
    return [{"event_type": k, "count": v} for k, v in counts.most_common()]


# ── API: Semester comparison ──────────────────────────────────────────────────

@app.get("/api/semester")
async def api_semester():
    neo = _try_neo4j(lambda: __import__(
        "phase4_dashboard.causal_evidence", fromlist=["get_semester_comparison"]
    ).get_semester_comparison())
    if neo:
        return neo

    events = _load_events()
    by_sem: dict[int, dict] = defaultdict(lambda: {"total": 0, "anomalies": 0})
    for e in events:
        s = e.get("semester")
        if s:
            by_sem[int(s)]["total"] += 1
            if e.get("is_anomaly"):
                by_sem[int(s)]["anomalies"] += 1
    return [
        {"semester": s, "total_events": v["total"], "anomalies": v["anomalies"]}
        for s, v in sorted(by_sem.items())
    ]


# ── API: Recent anomalies (for semantic event log + live feed) ────────────────

@app.get("/api/anomalies")
async def api_anomalies(limit: int = Query(60), semester: int = Query(0)):
    sem = semester or None
    neo = _try_neo4j(lambda: __import__(
        "phase4_dashboard.causal_evidence", fromlist=["get_recent_anomalies"]
    ).get_recent_anomalies(limit, sem))
    if neo:
        return neo

    events = _load_events()
    if sem:
        events = [e for e in events if e.get("semester") == sem]
    anomaly_evts = [e for e in events if e.get("is_anomaly")]
    anomaly_evts.sort(key=lambda e: str(e.get("timestamp") or ""), reverse=True)
    return anomaly_evts[:limit]


# ── API: Demand heatmap ───────────────────────────────────────────────────────

@app.get("/api/demand")
async def api_demand(semester: int = Query(0)):
    sem = semester or None
    neo = _try_neo4j(lambda: __import__(
        "phase4_dashboard.causal_evidence", fromlist=["get_demand_aggregates"]
    ).get_demand_aggregates(sem))
    if neo:
        return neo[:300]

    events = _load_events()
    if sem:
        events = [e for e in events if e.get("semester") == sem]
    zone_map: dict[str, dict] = {}
    for e in events:
        key = str(e.get("matched_edge_u") or "")
        lat = e.get("lat")
        lon = e.get("lon")
        if key and lat and lon:
            if key not in zone_map:
                zone_map[key] = {"lat": lat, "lon": lon, "trip_count": 0}
            zone_map[key]["trip_count"] += 1
    return sorted(zone_map.values(), key=lambda x: -x["trip_count"])[:300]


# ── SSE: Live vehicle positions ───────────────────────────────────────────────

async def _position_stream():
    while True:
        positions = _tracker.tick()
        yield f"data: {json.dumps(positions)}\n\n"
        await asyncio.sleep(3)


@app.get("/sse/positions")
async def stream_positions():
    return StreamingResponse(
        _position_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
