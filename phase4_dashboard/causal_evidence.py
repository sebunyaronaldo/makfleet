"""
Neo4j causal subgraph retrieval for the Causal Evidence tab.

When the dashboard flags an anomalous TelematicsEvent, this module fetches
the 2-hop ego-subgraph from Neo4j: the event's location, adjacent
intersections, driver history, and vehicle — the full causal context that
led to the anomaly flag.

This transforms the BI platform from a passive chart renderer into an
interactive causal explanation engine, answering 'why' rather than 'what'.
"""

from config.settings import NEO4J_URI, NEO4J_AUTH

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    _NEO4J_OK = True
except ImportError:
    _NEO4J_OK = False
    ServiceUnavailable = Exception
    AuthError = Exception


def _get_driver():
    if not _NEO4J_OK:
        return None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


def get_anomaly_subgraph(event_id: str) -> dict:
    """
    Fetch the causal subgraph for a flagged TelematicsEvent.

    Returns a dict with:
      event       — the event properties
      location    — the Intersection it occurred at
      driver      — driver details
      vehicle     — vehicle details
      neighbors   — list of adjacent intersections (2-hop)
      past_anomalies — driver's prior anomalous events
    """
    driver = _get_driver()
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

            return {
                "event": e,
                "location": i,
                "driver": d,
                "vehicle": v,
                "neighbors": neighbors,
                "past_anomalies": past[:10],  # cap for dashboard readability
            }
    finally:
        driver.close()


def get_recent_anomalies(limit: int = 50, semester: int | None = None) -> list[dict]:
    """Fetch the most recent anomalous TelematicsEvent records for the dashboard table."""
    driver = _get_driver()
    if driver is None:
        return []
    try:
        with driver.session() as s:
            if semester:
                result = s.run("""
                    MATCH (e:TelematicsEvent)
                    WHERE e.is_anomaly = true AND e.semester = $sem
                    RETURN e LIMIT $limit
                """, sem=semester, limit=limit)
            else:
                result = s.run("""
                    MATCH (e:TelematicsEvent)
                    WHERE e.is_anomaly = true
                    RETURN e LIMIT $limit
                """, limit=limit)
            return [dict(r["e"]) for r in result]
    finally:
        driver.close()


def get_demand_aggregates(semester: int | None = None) -> list[dict]:
    """Hourly trip demand aggregated per campus zone for heatmap rendering."""
    driver = _get_driver()
    if driver is None:
        return []
    try:
        with driver.session() as s:
            query = """
                MATCH (e:TelematicsEvent)-[:LOCATED_AT]->(i:Intersection)
                WITH i.osm_id AS zone, i.lat AS lat, i.lon AS lon,
                     count(e) AS trip_count
                RETURN zone, lat, lon, trip_count
                ORDER BY trip_count DESC
                LIMIT 200
            """
            result = s.run(query)
            return [dict(r) for r in result]
    finally:
        driver.close()


def get_vehicle_trajectory(vehicle_id: str, semester: int | None = None) -> list[dict]:
    """Fetch all map-matched points for a vehicle (for trajectory map rendering)."""
    driver = _get_driver()
    if driver is None:
        return []
    try:
        with driver.session() as s:
            if semester:
                result = s.run("""
                    MATCH (e:TelematicsEvent {vehicle_id: $vid})
                    WHERE e.semester = $sem
                    RETURN e.lat AS lat, e.lon AS lon,
                           e.speed_kmh AS speed_kmh,
                           e.timestamp AS timestamp,
                           e.event_type AS event_type,
                           e.is_anomaly AS is_anomaly
                    ORDER BY e.timestamp
                    LIMIT 2000
                """, vid=vehicle_id, sem=semester)
            else:
                result = s.run("""
                    MATCH (e:TelematicsEvent {vehicle_id: $vid})
                    RETURN e.lat AS lat, e.lon AS lon,
                           e.speed_kmh AS speed_kmh,
                           e.timestamp AS timestamp,
                           e.event_type AS event_type,
                           e.is_anomaly AS is_anomaly
                    ORDER BY e.timestamp
                    LIMIT 2000
                """, vid=vehicle_id)
            return [dict(r) for r in result]
    finally:
        driver.close()


def _empty_subgraph(event_id: str) -> dict:
    return {
        "event": {"event_id": event_id},
        "location": {},
        "driver": {},
        "vehicle": {},
        "neighbors": [],
        "past_anomalies": [],
    }
