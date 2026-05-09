"""
Batch-loads the Makerere campus topology, fleet entities, and enriched
telemetry events into Neo4j using UNWIND Cypher for performance.

Gap 1 enforcement: safe_stop records are loaded as :ContextualStop nodes,
not as :TelematicsEvent nodes, enforcing the ontological separation at the
storage layer rather than at query time.
"""

import json
from pathlib import Path

import networkx as nx
from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_AUTH
from ontology.makfleet_ontology import Rel, TemporalEventEntity, ROAD_TYPE_ENCODING, DEFAULT_ROAD_TYPE

BATCH_SIZE = 500


def get_driver(uri=NEO4J_URI, auth=NEO4J_AUTH):
    return GraphDatabase.driver(uri, auth=auth)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def create_constraints(driver) -> None:
    with driver.session() as s:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Intersection) REQUIRE n.osm_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vehicle) REQUIRE n.vehicle_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Driver) REQUIRE n.driver_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:TelematicsEvent) REQUIRE n.event_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ContextualStop) REQUIRE n.event_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RoadSegment) REQUIRE n.segment_id IS UNIQUE",
        ]
        for c in constraints:
            s.run(c)
    print("[neo4j] Constraints created")


# ---------------------------------------------------------------------------
# Campus spatial graph
# ---------------------------------------------------------------------------

def load_spatial_graph(G: nx.MultiDiGraph, driver) -> None:
    nodes = []
    for nid, data in G.nodes(data=True):
        nodes.append({
            "osm_id": str(nid),
            "lat": float(data.get("y", 0)),
            "lon": float(data.get("x", 0)),
            "betweenness": float(data.get("betweenness", 0)),
            "degree_centrality": float(data.get("degree_centrality", 0)),
            "landmark_type": data.get("landmark_type", ""),
            "landmark_name": data.get("landmark_name", ""),
        })

    edges = []
    for u, v, key, data in G.edges(data=True, keys=True):
        rtype = data.get("highway", "unclassified")
        if isinstance(rtype, list):
            rtype = rtype[0]
        edges.append({
            "segment_id": f"{u}_{v}_{key}",
            "u": str(u),
            "v": str(v),
            "length_m": float(data.get("length", 0)),
            "road_type": str(rtype),
            "road_type_enc": ROAD_TYPE_ENCODING.get(str(rtype), DEFAULT_ROAD_TYPE),
            "maxspeed": _parse_speed(data.get("maxspeed", "30")),
        })

    with driver.session() as s:
        # Nodes — split BodaBodaStop/ZebraCrossing with extra label
        for batch in _chunks(nodes, BATCH_SIZE):
            s.run("""
                UNWIND $batch AS n
                MERGE (i:Intersection {osm_id: n.osm_id})
                SET i.lat = n.lat, i.lon = n.lon,
                    i.betweenness = n.betweenness,
                    i.degree_centrality = n.degree_centrality,
                    i.landmark_type = n.landmark_type,
                    i.landmark_name = n.landmark_name
            """, batch=batch)

        # Extra labels for landmark nodes
        s.run("""
            MATCH (i:Intersection) WHERE i.landmark_type = 'BodaBodaStop'
            SET i:BodaBodaStop
        """)
        s.run("""
            MATCH (i:Intersection) WHERE i.landmark_type = 'ZebraCrossing'
            SET i:ZebraCrossing
        """)

        # Road segment nodes + ADJACENT_TO relationships
        for batch in _chunks(edges, BATCH_SIZE):
            s.run("""
                UNWIND $batch AS e
                MERGE (r:RoadSegment {segment_id: e.segment_id})
                SET r.length_m = e.length_m, r.road_type = e.road_type,
                    r.road_type_enc = e.road_type_enc, r.maxspeed = e.maxspeed
                WITH r, e
                MATCH (u:Intersection {osm_id: e.u})
                MATCH (v:Intersection {osm_id: e.v})
                MERGE (u)-[:ADJACENT_TO {length_m: e.length_m, road_type: e.road_type}]->(v)
                MERGE (r)-[:LOCATED_AT]->(u)
            """, batch=batch)

    print(f"[neo4j] Loaded {len(nodes)} intersections, {len(edges)} road segments")


# ---------------------------------------------------------------------------
# Fleet entities
# ---------------------------------------------------------------------------

def load_fleet_entities(vehicles: list[dict], drivers: list[dict], driver_db) -> None:
    with driver_db.session() as s:
        s.run("""
            UNWIND $vehicles AS v
            MERGE (n:Vehicle {vehicle_id: v.vehicle_id})
            SET n.plate = v.plate
        """, vehicles=vehicles)

        s.run("""
            UNWIND $drivers AS d
            MERGE (dr:Driver {driver_id: d.driver_id})
            SET dr.name = d.name, dr.risk_profile = d.risk_profile,
                dr.is_reckless = d.is_reckless
            WITH dr, d
            MATCH (v:Vehicle {vehicle_id: d.vehicle_id})
            MERGE (dr)-[:DRIVES]->(v)
        """, drivers=drivers)

    print(f"[neo4j] Loaded {len(vehicles)} vehicles, {len(drivers)} drivers")


# ---------------------------------------------------------------------------
# Telemetry events (Gap 1: ContextualStop as distinct label)
# ---------------------------------------------------------------------------

def load_telemetry_events(records: list[dict], driver_db) -> None:
    """
    Branch on event_type:
      safe_stop  → :ContextualStop node
      everything else → :TelematicsEvent node
    """
    normal_events = [r for r in records if r.get("event_type") != "safe_stop"]
    contextual_stops = [r for r in records if r.get("event_type") == "safe_stop"]

    with driver_db.session() as s:
        # TelematicsEvent nodes
        for batch in _chunks(normal_events, BATCH_SIZE):
            s.run("""
                UNWIND $batch AS e
                CREATE (t:TelematicsEvent {
                    event_id: e.event_id,
                    vehicle_id: e.vehicle_id,
                    driver_id: e.driver_id,
                    timestamp: e.timestamp,
                    lat: e.lat, lon: e.lon,
                    speed_kmh: e.speed_kmh,
                    heading_deg: e.heading_deg,
                    ax: e.ax, ay: e.ay, az: e.az,
                    engine_state: e.engine_state,
                    event_type: e.event_type,
                    severity_score: e.severity_score,
                    is_anomaly: e.is_anomaly,
                    semester: e.semester,
                    matched_edge_u: e.matched_edge_u,
                    matched_edge_v: e.matched_edge_v,
                    map_match_confidence: e.map_match_confidence,
                    provenance_hash: e.provenance_hash
                })
                WITH t, e
                MATCH (i:Intersection {osm_id: toString(e.matched_edge_u)})
                CREATE (t)-[:LOCATED_AT]->(i)
                WITH t, e
                MATCH (d:Driver {driver_id: e.driver_id})
                CREATE (t)-[:CAUSED_BY]->(d)
            """, batch=batch)

        # ContextualStop nodes (ontologically distinct — NOT TelematicsEvent)
        for batch in _chunks(contextual_stops, BATCH_SIZE):
            s.run("""
                UNWIND $batch AS e
                CREATE (c:ContextualStop {
                    event_id: e.event_id,
                    vehicle_id: e.vehicle_id,
                    driver_id: e.driver_id,
                    timestamp: e.timestamp,
                    lat: e.lat, lon: e.lon,
                    speed_kmh: e.speed_kmh,
                    event_type: 'safe_stop',
                    severity_score: 0.0,
                    is_anomaly: false,
                    semester: e.semester,
                    matched_edge_u: e.matched_edge_u,
                    provenance_hash: e.provenance_hash
                })
                WITH c, e
                MATCH (i:Intersection {osm_id: toString(e.matched_edge_u)})
                CREATE (c)-[:LOCATED_AT]->(i)
                WITH c, e
                MATCH (d:Driver {driver_id: e.driver_id})
                CREATE (c)-[:CAUSED_BY]->(d)
            """, batch=batch)

    print(f"[neo4j] Loaded {len(normal_events)} TelematicsEvent, {len(contextual_stops)} ContextualStop")


def load_provenance_anchor(root_hash: str, semester: int, record_count: int, driver_db) -> None:
    with driver_db.session() as s:
        s.run("""
            MERGE (p:ProvenanceAnchor {semester: $sem})
            SET p.merkle_root = $root, p.record_count = $count,
                p.created_at = datetime()
        """, sem=semester, root=root_hash, count=record_count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _parse_speed(val) -> float:
    if isinstance(val, list):
        val = val[0]
    try:
        return float(str(val).replace(" mph", "").replace(" km/h", ""))
    except (ValueError, TypeError):
        return 30.0


def clear_database(driver_db) -> None:
    """Drop all nodes/relationships — use only when rebuilding from scratch."""
    with driver_db.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("[neo4j] Database cleared")
