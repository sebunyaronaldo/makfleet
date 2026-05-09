"""
Benchmark: ST-KG (Neo4j Cypher) vs Kimball Star Schema (pandas).

Gap 2: Each query runs BENCHMARK_RUNS times; median latency reported via
statistics.median() — robust to JIT warmup on run 0.

5 queries chosen to expose expressiveness gaps:
  Q1 — Graph traversal (2-hop) — star schema requires self-joins
  Q2 — Traffic density per segment (control — both handle this)
  Q3 — Spatial predicate: reckless drivers near school zones
  Q4 — Causal chain traversal — infeasible in star schema
  Q5 — Semantic zone aggregation — star schema competitive
"""

import statistics
import time
from datetime import datetime

try:
    import pandas as pd
    _PD_OK = True
except ImportError:
    _PD_OK = False

try:
    from neo4j import GraphDatabase
    _NEO4J_OK = True
except ImportError:
    _NEO4J_OK = False

from config.settings import NEO4J_URI, NEO4J_AUTH, BENCHMARK_RUNS


# ---------------------------------------------------------------------------
# Timing harness (Gap 2)
# ---------------------------------------------------------------------------

def timed_query(fn, *args, runs: int = BENCHMARK_RUNS):
    """Run fn(*args) `runs` times, return (result, median_ms)."""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(times)


# ---------------------------------------------------------------------------
# Star Schema (in-memory pandas)
# ---------------------------------------------------------------------------

def build_star_schema(neo4j_driver=None) -> dict[str, "pd.DataFrame"]:
    """
    Construct a Kimball-style star schema from Neo4j data (or generate
    synthetic fallback data if Neo4j is unavailable).
    """
    if not _PD_OK:
        return {}

    if _NEO4J_OK and neo4j_driver:
        with neo4j_driver.session() as s:
            events = s.run("""
                MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
                MATCH (e)-[:CAUSED_BY]->(d:Driver)
                RETURN e.event_id AS event_id,
                       e.timestamp AS timestamp,
                       e.speed_kmh AS speed_kmh,
                       e.event_type AS event_type,
                       e.is_anomaly AS is_anomaly,
                       e.severity_score AS severity_score,
                       e.semester AS semester,
                       e.vehicle_id AS vehicle_id,
                       e.driver_id AS driver_id,
                       i.osm_id AS location_key,
                       i.lat AS lat, i.lon AS lon,
                       i.landmark_type AS zone_type,
                       d.risk_profile AS driver_risk
                LIMIT 20000
            """)
            rows = [dict(r) for r in events]
    else:
        rows = _synthetic_star_data()

    if not rows:
        rows = _synthetic_star_data()

    fact = pd.DataFrame(rows)
    if "timestamp" in fact.columns:
        fact["timestamp"] = pd.to_datetime(fact["timestamp"], errors="coerce", utc=True)
        fact["hour"] = fact["timestamp"].dt.floor("h")

    dim_time = fact[["timestamp", "hour", "semester"]].drop_duplicates() if "timestamp" in fact.columns else pd.DataFrame()
    dim_location = fact[["location_key", "lat", "lon", "zone_type"]].drop_duplicates() if "location_key" in fact.columns else pd.DataFrame()
    dim_driver = fact[["driver_id", "driver_risk"]].drop_duplicates() if "driver_id" in fact.columns else pd.DataFrame()
    dim_vehicle = fact[["vehicle_id"]].drop_duplicates() if "vehicle_id" in fact.columns else pd.DataFrame()

    return {
        "fact": fact,
        "dim_time": dim_time,
        "dim_location": dim_location,
        "dim_driver": dim_driver,
        "dim_vehicle": dim_vehicle,
    }


def _synthetic_star_data(n: int = 5000) -> list[dict]:
    import random, math
    rows = []
    for i in range(n):
        rows.append({
            "event_id": f"ev_{i}",
            "timestamp": f"2024-01-15T{random.randint(6,22):02d}:{random.randint(0,59):02d}:00Z",
            "speed_kmh": random.uniform(5, 55),
            "event_type": random.choice(["normal_travel", "harsh_braking", "speeding", "idling"]),
            "is_anomaly": random.random() < 0.08,
            "severity_score": random.uniform(0, 1),
            "semester": random.choice([1, 2]),
            "vehicle_id": f"V{random.randint(1,15):03d}",
            "driver_id": f"D{random.randint(1,15):03d}",
            "location_key": str(random.randint(100000, 999999)),
            "lat": random.uniform(0.334, 0.342),
            "lon": random.uniform(32.568, 32.576),
            "zone_type": random.choice(["", "BodaBodaStop", "ZebraCrossing"]),
            "driver_risk": random.uniform(0, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# ST-KG queries (Cypher)
# ---------------------------------------------------------------------------

def _kg_q1(driver):
    """Q1 — 2-hop graph traversal from any BodaBodaStop."""
    with driver.session() as s:
        r = s.run("""
            MATCH (stop:BodaBodaStop)-[:ADJACENT_TO*1..2]-(neighbor:Intersection)
            RETURN count(DISTINCT neighbor) AS reachable_nodes
        """)
        return r.single()


def _kg_q2(driver):
    """Q2 — Traffic density per road segment (control)."""
    with driver.session() as s:
        r = s.run("""
            MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
            RETURN i.osm_id AS segment, count(e) AS trip_count
            ORDER BY trip_count DESC LIMIT 20
        """)
        return list(r)


def _kg_q3(driver):
    """Q3 — Reckless drivers near school/pedestrian zones."""
    with driver.session() as s:
        r = s.run("""
            MATCH (e:TelematicsEvent)-[:CAUSED_BY]->(d:Driver)
            MATCH (e)-[:LOCATED_AT]->(i:Intersection)
            WHERE e.is_anomaly = true AND d.risk_profile > 0.7
              AND (i:ZebraCrossing OR i:BodaBodaStop)
            RETURN d.driver_id AS driver, count(e) AS incidents, i.landmark_name AS zone
            ORDER BY incidents DESC LIMIT 10
        """)
        return list(r)


def _kg_q4(driver):
    """Q4 — Causal chain: anomaly → location → 2-hop intersections → nearby stops."""
    with driver.session() as s:
        r = s.run("""
            MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
            WHERE e.is_anomaly = true
            MATCH (i)-[:ADJACENT_TO*1..2]-(n:Intersection)
            WHERE n:BodaBodaStop OR n:ZebraCrossing
            RETURN count(DISTINCT e) AS anomalies_near_stops
        """)
        return r.single()


def _kg_q5(driver):
    """Q5 — Demand aggregated by semantic zone type."""
    with driver.session() as s:
        r = s.run("""
            MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
            RETURN i.landmark_type AS zone_type, count(e) AS total_events,
                   avg(e.speed_kmh) AS avg_speed
            ORDER BY total_events DESC
        """)
        return list(r)


# ---------------------------------------------------------------------------
# Star Schema queries (pandas)
# ---------------------------------------------------------------------------

def _ss_q1(schema):
    """Q1 — Approximate 2-hop via lat/lon range (no true graph traversal)."""
    fact = schema.get("fact", pd.DataFrame())
    if fact.empty:
        return 0
    stops = fact[fact["zone_type"] == "BodaBodaStop"][["lat", "lon"]].drop_duplicates()
    # Approximate: find all locations within ~200m bounding box of any stop
    results = set()
    for _, row in stops.iterrows():
        nearby = fact[
            (fact["lat"].between(row["lat"] - 0.002, row["lat"] + 0.002)) &
            (fact["lon"].between(row["lon"] - 0.002, row["lon"] + 0.002))
        ]["location_key"]
        results.update(nearby.tolist())
    return len(results)


def _ss_q2(schema):
    """Q2 — Traffic density per segment."""
    fact = schema.get("fact", pd.DataFrame())
    if fact.empty:
        return pd.DataFrame()
    return fact.groupby("location_key").size().reset_index(name="trip_count").nlargest(20, "trip_count")


def _ss_q3(schema):
    """Q3 — Reckless drivers near school zones (requires multi-condition filter)."""
    fact = schema.get("fact", pd.DataFrame())
    if fact.empty:
        return pd.DataFrame()
    reckless = fact[(fact["is_anomaly"] == True) & (fact["driver_risk"] > 0.7)]
    near_zone = reckless[reckless["zone_type"].isin(["ZebraCrossing", "BodaBodaStop"])]
    return near_zone.groupby("driver_id").size().reset_index(name="incidents").nlargest(10, "incidents")


def _ss_q4(schema):
    """
    Q4 — Causal chain traversal (NOT expressible in star schema without many
    self-joins and spatial approximation). Returns approximate count.
    """
    fact = schema.get("fact", pd.DataFrame())
    if fact.empty:
        return 0
    # Can only approximate: anomalies near known stop locations
    stops = fact[fact["zone_type"].isin(["BodaBodaStop", "ZebraCrossing"])][["lat", "lon"]]
    anomalies = fact[fact["is_anomaly"] == True]
    count = 0
    for _, ev in anomalies.iterrows():
        for _, stop in stops.iterrows():
            dist = ((ev["lat"] - stop["lat"]) ** 2 + (ev["lon"] - stop["lon"]) ** 2) ** 0.5
            if dist < 0.003:
                count += 1
                break
    return count


def _ss_q5(schema):
    """Q5 — Demand by semantic zone type."""
    fact = schema.get("fact", pd.DataFrame())
    if fact.empty:
        return pd.DataFrame()
    return fact.groupby("zone_type").agg(
        total_events=("event_id", "count"),
        avg_speed=("speed_kmh", "mean"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Run all benchmarks
# ---------------------------------------------------------------------------

def run_all_benchmarks() -> dict:
    results = {}

    neo4j_driver = None
    if _NEO4J_OK:
        try:
            neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            neo4j_driver.verify_connectivity()
        except Exception:
            neo4j_driver = None

    schema = build_star_schema(neo4j_driver) if _PD_OK else {}

    query_pairs = [
        ("Q1: 2-hop graph traversal",       _kg_q1, _ss_q1, True,  True),
        ("Q2: Traffic density (control)",    _kg_q2, _ss_q2, True,  True),
        ("Q3: Reckless drivers near zones",  _kg_q3, _ss_q3, True,  True),
        ("Q4: Causal chain traversal",       _kg_q4, _ss_q4, True,  False),
        ("Q5: Semantic zone aggregation",    _kg_q5, _ss_q5, True,  True),
    ]

    for name, kg_fn, ss_fn, kg_expr, ss_expr in query_pairs:
        kg_ms = 0.0
        ss_ms = 0.0

        if neo4j_driver and kg_expr:
            try:
                _, kg_ms = timed_query(kg_fn, neo4j_driver)
            except Exception as e:
                kg_ms = -1.0

        if schema and ss_expr:
            try:
                _, ss_ms = timed_query(ss_fn, schema)
            except Exception as e:
                ss_ms = -1.0

        results[name] = {
            "stkg_median_ms": round(kg_ms, 2),
            "star_median_ms": round(ss_ms, 2),
            "stkg_expressible": kg_expr,
            "star_expressible": ss_expr,
        }

    if neo4j_driver:
        neo4j_driver.close()

    return results


if __name__ == "__main__":
    import json
    res = run_all_benchmarks()
    print(json.dumps(res, indent=2))
