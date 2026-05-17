"""
Neo4j data layer for the MakFleet analytics dashboard.

Provides KPI summaries, filtered anomaly/trajectory queries, demand aggregates,
and causal subgraph retrieval for the Causal Evidence tab.
"""

from __future__ import annotations

from functools import lru_cache

from config.settings import NEO4J_URI, NEO4J_AUTH

try:
    from neo4j import GraphDatabase
    _NEO4J_OK = True
except ImportError:
    _NEO4J_OK = False


def _get_driver():
    if not _NEO4J_OK:
        return None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_pooled_driver():
    """Reuse a single Bolt driver across dashboard queries."""
    return _get_driver()


def _semester_clause(alias: str = "e", semester: int | None = None) -> tuple[str, dict]:
    if semester:
        return f" AND {alias}.semester = $sem", {"sem": semester}
    return "", {}


def _vehicle_clause(alias: str = "e", vehicle_id: str | None = None) -> tuple[str, dict]:
    if vehicle_id and vehicle_id != "All":
        return f" AND {alias}.vehicle_id = $vid", {"vid": vehicle_id}
    return "", {}


def get_kpi_summary(semester: int | None = None) -> dict:
    """Executive KPIs for the dashboard header."""
    driver = get_pooled_driver()
    if driver is None:
        return _empty_kpis()

    sem_clause, sem_params = _semester_clause("e", semester)
    c_sem = sem_clause.replace("e.", "c.") if sem_clause else ""
    try:
        with driver.session() as s:
            row = s.run(f"""
                MATCH (e:TelematicsEvent)
                WHERE true{sem_clause}
                RETURN count(e) AS total_events,
                       sum(CASE WHEN e.is_anomaly = true THEN 1 ELSE 0 END) AS anomalies,
                       avg(CASE WHEN e.is_anomaly = true THEN e.severity_score END) AS avg_severity
            """, **sem_params).single()

            safe_row = s.run(f"""
                MATCH (c:ContextualStop) WHERE true{c_sem}
                RETURN count(c) AS safe_stops
            """, **sem_params).single()

            reckless_row = s.run(f"""
                MATCH (e:TelematicsEvent)-[:CAUSED_BY]->(d:Driver)
                WHERE e.is_anomaly = true AND d.risk_profile > 0.7{sem_clause}
                RETURN count(DISTINCT d) AS reckless_drivers
            """, **sem_params).single()

            top_type = s.run(f"""
                MATCH (e:TelematicsEvent)
                WHERE e.is_anomaly = true{sem_clause}
                RETURN e.event_type AS event_type, count(*) AS cnt
                ORDER BY cnt DESC LIMIT 1
            """, **sem_params).single()

            hot_zone = s.run(f"""
                MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
                WHERE e.is_anomaly = true{sem_clause}
                RETURN i.osm_id AS zone, count(e) AS cnt
                ORDER BY cnt DESC LIMIT 1
            """, **sem_params).single()

            zone_demand = s.run(f"""
                MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
                WHERE true{sem_clause}
                RETURN count(DISTINCT i) AS zones
            """, **sem_params).single()

        if row is None:
            return _empty_kpis()

        total = int(row["total_events"] or 0)
        anomalies = int(row["anomalies"] or 0)
        return {
            "total_events": total,
            "anomalies": anomalies,
            "anomaly_rate_pct": round(100 * anomalies / total, 2) if total else 0.0,
            "avg_severity": round(float(row["avg_severity"] or 0), 3),
            "safe_stops": int(safe_row["safe_stops"] or 0) if safe_row else 0,
            "reckless_drivers": int(reckless_row["reckless_drivers"] or 0) if reckless_row else 0,
            "demand_zones": int(zone_demand["zones"] or 0) if zone_demand else 0,
            "top_event_type": (top_type["event_type"] if top_type else "—") or "—",
            "hot_zone": str(hot_zone["zone"]) if hot_zone and hot_zone["zone"] else "—",
        }
    except Exception:
        return _empty_kpis()


def get_semester_comparison() -> list[dict]:
    """Per-semester anomaly counts for concept-drift chart."""
    driver = get_pooled_driver()
    if driver is None:
        return []
    try:
        with driver.session() as s:
            result = s.run("""
                MATCH (e:TelematicsEvent)
                WHERE e.semester IS NOT NULL
                RETURN e.semester AS semester,
                       count(e) AS total_events,
                       sum(CASE WHEN e.is_anomaly = true THEN 1 ELSE 0 END) AS anomalies
                ORDER BY semester
            """)
            return [dict(r) for r in result]
    except Exception:
        return []


def get_anomaly_subgraph(event_id: str) -> dict:
    driver = get_pooled_driver()
    if driver is None:
        return _empty_subgraph(event_id)

    try:
        with driver.session() as s:
            result = s.run("""
                MATCH (e:TelematicsEvent {event_id: $eid})
                OPTIONAL MATCH (e)-[:LOCATED_AT]->(i:Intersection)
                OPTIONAL MATCH (e)-[:CAUSED_BY]->(d:Driver)
                OPTIONAL MATCH (d)-[:DRIVES]->(v:Vehicle)
                OPTIONAL MATCH (i)-[:ADJACENT_TO*1..2]-(neighbor:Intersection)
                OPTIONAL MATCH (d)<-[:CAUSED_BY]-(past:TelematicsEvent)
                    WHERE past.is_anomaly = true AND past.event_id <> $eid
                RETURN e, i, d, v,
                       collect(DISTINCT neighbor) AS neighbors,
                       collect(DISTINCT past) AS past_anomalies
                LIMIT 1
            """, eid=event_id)

            row = result.single()
            if row is None:
                return _empty_subgraph(event_id)

            e = dict(row["e"]) if row["e"] else {}
            i = dict(row["i"]) if row["i"] else {}
            d = dict(row["d"]) if row["d"] else {}
            v = dict(row["v"]) if row["v"] else {}
            neighbors = [dict(n) for n in (row["neighbors"] or [])]
            past = [dict(p) for p in (row["past_anomalies"] or [])]

            labels = []
            if i:
                label_result = s.run(
                    "MATCH (n) WHERE n.osm_id = $oid RETURN labels(n) AS lbls LIMIT 1",
                    oid=i.get("osm_id"),
                ).single()
                if label_result:
                    labels = list(label_result["lbls"] or [])

            return {
                "event": e,
                "location": i,
                "location_labels": labels,
                "driver": d,
                "vehicle": v,
                "neighbors": neighbors,
                "past_anomalies": past[:10],
            }
    except Exception:
        return _empty_subgraph(event_id)


def get_recent_anomalies(
    limit: int = 100,
    semester: int | None = None,
    vehicle_id: str | None = None,
) -> list[dict]:
    driver = get_pooled_driver()
    if driver is None:
        return []

    sem_clause, sem_params = _semester_clause("e", semester)
    veh_clause, veh_params = _vehicle_clause("e", vehicle_id)
    params = {**sem_params, **veh_params, "limit": limit}

    try:
        with driver.session() as s:
            result = s.run(f"""
                MATCH (e:TelematicsEvent)
                WHERE e.is_anomaly = true{sem_clause}{veh_clause}
                RETURN e
                ORDER BY e.timestamp DESC
                LIMIT $limit
            """, **params)
            return [dict(r["e"]) for r in result]
    except Exception:
        return []


def get_demand_aggregates(semester: int | None = None) -> list[dict]:
    driver = get_pooled_driver()
    if driver is None:
        return []

    sem_clause, sem_params = _semester_clause("e", semester)
    try:
        with driver.session() as s:
            result = s.run(f"""
                MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
                WHERE true{sem_clause}
                WITH i.osm_id AS zone, i.lat AS lat, i.lon AS lon,
                     i.landmark_name AS landmark_name,
                     i.landmark_type AS landmark_type,
                     count(e) AS trip_count
                RETURN zone, lat, lon, landmark_name, landmark_type, trip_count
                ORDER BY trip_count DESC
                LIMIT 200
            """, **sem_params)
            return [dict(r) for r in result]
    except Exception:
        return []


def get_hourly_anomaly_breakdown(
    semester: int | None = None,
    vehicle_id: str | None = None,
) -> list[dict]:
    """Pre-aggregated hour × event_type counts for charts."""
    driver = get_pooled_driver()
    if driver is None:
        return []

    sem_clause, sem_params = _semester_clause("e", semester)
    veh_clause, veh_params = _vehicle_clause("e", vehicle_id)
    params = {**sem_params, **veh_params}

    try:
        with driver.session() as s:
            result = s.run(f"""
                MATCH (e:TelematicsEvent)
                WHERE e.is_anomaly = true{sem_clause}{veh_clause}
                RETURN e.timestamp AS ts, e.event_type AS event_type,
                       e.severity_score AS severity_score
            """, **params)
            return [dict(r) for r in result]
    except Exception:
        return []


def get_event_type_distribution(semester: int | None = None) -> list[dict]:
    driver = get_pooled_driver()
    if driver is None:
        return []

    sem_clause, sem_params = _semester_clause("e", semester)
    try:
        with driver.session() as s:
            result = s.run(f"""
                MATCH (e:TelematicsEvent)
                WHERE e.is_anomaly = true{sem_clause}
                RETURN e.event_type AS event_type, count(*) AS count
                ORDER BY count DESC
            """, **sem_params)
            return [dict(r) for r in result]
    except Exception:
        return []


def get_landmarks() -> list[dict]:
    """Campus landmarks for map overlay."""
    driver = get_pooled_driver()
    if driver is None:
        return []
    try:
        with driver.session() as s:
            result = s.run("""
                MATCH (i:Intersection)
                WHERE i:ZebraCrossing OR i:BodaBodaStop
                RETURN i.osm_id AS osm_id, i.lat AS lat, i.lon AS lon,
                       i.landmark_name AS name, i.landmark_type AS landmark_type
            """)
            return [dict(r) for r in result]
    except Exception:
        return []


def get_vehicle_trajectory(
    vehicle_id: str,
    semester: int | None = None,
    limit: int = 2000,
) -> list[dict]:
    driver = get_pooled_driver()
    if driver is None:
        return []

    sem_clause, sem_params = _semester_clause("e", semester)
    params = {**sem_params, "vid": vehicle_id, "limit": limit}
    try:
        with driver.session() as s:
            result = s.run(f"""
                MATCH (e:TelematicsEvent {{vehicle_id: $vid}})
                WHERE true{sem_clause}
                RETURN e.lat AS lat, e.lon AS lon,
                       e.speed_kmh AS speed_kmh,
                       e.timestamp AS timestamp,
                       e.event_type AS event_type,
                       e.is_anomaly AS is_anomaly,
                       e.severity_score AS severity_score
                ORDER BY e.timestamp
                LIMIT $limit
            """, **params)
            return [dict(r) for r in result]
    except Exception:
        return []


def _empty_kpis() -> dict:
    return {
        "total_events": 0,
        "anomalies": 0,
        "anomaly_rate_pct": 0.0,
        "avg_severity": 0.0,
        "safe_stops": 0,
        "reckless_drivers": 0,
        "demand_zones": 0,
        "top_event_type": "—",
        "hot_zone": "—",
    }


def _empty_subgraph(event_id: str) -> dict:
    return {
        "event": {"event_id": event_id},
        "location": {},
        "location_labels": [],
        "driver": {},
        "vehicle": {},
        "neighbors": [],
        "past_anomalies": [],
    }
