"""
Converts Neo4j telemetry subgraphs into PyTorch Geometric Data objects.

Each sample is a sequence of T_SNAPSHOTS graph snapshots, each covering
WINDOW_SIZE_MIN minutes of activity on the campus road graph.

Split protocol (causal validity, no data leakage):
  - Semester 1 windows → train (first 80%) + val (last 20%)
  - Semester 2 windows → temporal OOD test
  - Eastern campus zone (lon > SPATIAL_OOD_LON) → spatial OOD test
"""

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
from torch_geometric.data import Data

from config.settings import (
    NEO4J_URI, NEO4J_AUTH,
    WINDOW_SIZE_MIN, T_SNAPSHOTS, SPATIAL_OOD_LON,
)

try:
    from neo4j import GraphDatabase
    _NEO4J_OK = True
except ImportError:
    _NEO4J_OK = False


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _get_driver():
    if not _NEO4J_OK:
        raise RuntimeError("neo4j driver not installed")
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def _query_window(session, start_iso: str, end_iso: str, semester: int) -> list[dict]:
    result = session.run("""
        MATCH (e:TelematicsEvent)
        WHERE e.timestamp >= $start AND e.timestamp < $end AND e.semester = $sem
        RETURN e.event_id AS event_id,
               e.lat AS lat, e.lon AS lon,
               e.speed_kmh AS speed_kmh,
               e.is_anomaly AS is_anomaly,
               e.matched_edge_u AS u,
               e.matched_edge_v AS v,
               e.severity_score AS severity_score
    """, start=start_iso, end=end_iso, sem=semester)
    return [dict(r) for r in result]


def _query_graph_structure(session) -> tuple[list[dict], list[dict]]:
    """Fetch all intersections and their adjacency for building the graph structure."""
    nodes = session.run("""
        MATCH (i:Intersection)
        RETURN i.osm_id AS osm_id, i.lat AS lat, i.lon AS lon,
               i.betweenness AS betweenness, i.degree_centrality AS degree_centrality
    """)
    edges = session.run("""
        MATCH (a:Intersection)-[r:ADJACENT_TO]->(b:Intersection)
        RETURN a.osm_id AS u, b.osm_id AS v, r.length_m AS length_m
    """)
    return [dict(n) for n in nodes], [dict(e) for e in edges]


# ---------------------------------------------------------------------------
# Graph snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot(
    node_list: list[dict],
    node_index: dict[str, int],
    edge_list: list[dict],
    window_events: list[dict],
) -> Data:
    N = len(node_list)
    X = torch.zeros((N, 6), dtype=torch.float)
    y = torch.zeros(N, dtype=torch.float)

    # Base node features: lat, lon, betweenness, degree_centrality, event_count, mean_speed
    for i, node in enumerate(node_list):
        X[i, 0] = float(node.get("lat") or 0)
        X[i, 1] = float(node.get("lon") or 0)
        X[i, 2] = float(node.get("betweenness") or 0)
        X[i, 3] = float(node.get("degree_centrality") or 0)

    # Aggregate events onto nodes
    node_events: dict[int, list] = defaultdict(list)
    for ev in window_events:
        u_id = str(ev.get("u") or "")
        if u_id in node_index:
            node_events[node_index[u_id]].append(ev)

    for node_idx, events in node_events.items():
        X[node_idx, 4] = len(events)
        speeds = [e.get("speed_kmh") or 0 for e in events]
        X[node_idx, 5] = float(np.mean(speeds)) if speeds else 0.0
        if any(e.get("is_anomaly") for e in events):
            y[node_idx] = 1.0

    # Edge index
    src, dst, edge_attrs = [], [], []
    for e in edge_list:
        u_id, v_id = str(e["u"]), str(e["v"])
        if u_id in node_index and v_id in node_index:
            src.append(node_index[u_id])
            dst.append(node_index[v_id])
            edge_attrs.append([float(e.get("length_m") or 0), 0.0])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float)

    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _query_all_events(session, semester: int) -> list[dict]:
    """Single bulk query — fetch all events for a semester at once."""
    result = session.run("""
        MATCH (e:TelematicsEvent)
        WHERE e.semester = $sem
        RETURN e.event_id    AS event_id,
               e.timestamp   AS timestamp,
               e.lat         AS lat,
               e.lon         AS lon,
               e.speed_kmh   AS speed_kmh,
               e.is_anomaly  AS is_anomaly,
               e.matched_edge_u AS u,
               e.matched_edge_v AS v,
               e.severity_score AS severity_score
    """, sem=semester)
    return [dict(r) for r in result]


def build_dataset(semester: int) -> list[dict]:
    """
    Returns list of {snapshots: list[Data], label: float, window_start: str,
                      is_ood_spatial: bool} dicts.
    Each sample covers T_SNAPSHOTS consecutive windows.

    Uses a single bulk Neo4j query then partitions into windows in Python
    — avoids 8,000+ round trips.
    """
    if not _NEO4J_OK:
        print("[graph_dataset] Neo4j not available — returning empty dataset")
        return []

    driver = _get_driver()
    samples = []

    with driver.session() as session:
        node_list, edge_list = _query_graph_structure(session)
        if not node_list:
            print("[graph_dataset] No intersection nodes in Neo4j. Run neo4j_loader first.")
            driver.close()
            return []

        node_index = {str(n["osm_id"]): i for i, n in enumerate(node_list)}

        print(f"[graph_dataset] Bulk querying all Sem{semester} events...")
        all_events = _query_all_events(session, semester)
        print(f"[graph_dataset] Fetched {len(all_events):,} events — partitioning into windows...")

    driver.close()

    # Determine time range
    if semester == 1:
        base = datetime(2024, 1, 15)
        total_days = 90
    else:
        base = datetime(2024, 8, 1)
        total_days = 60

    end_time     = base + timedelta(days=total_days)
    window_delta = timedelta(minutes=WINDOW_SIZE_MIN)

    # Index events by window bucket in Python (no more Neo4j queries)
    window_events: dict[str, list] = defaultdict(list)
    for ev in all_events:
        ts_str = ev.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", ""))
        except ValueError:
            continue
        # Which window does this event belong to?
        offset_min = int((ts - base).total_seconds() / 60)
        bucket_min = (offset_min // WINDOW_SIZE_MIN) * WINDOW_SIZE_MIN
        bucket_dt  = base + timedelta(minutes=bucket_min)
        if base <= bucket_dt < end_time:
            window_events[bucket_dt.isoformat()].append(ev)

    # Build ordered window list
    current = base
    windows: list[Data] = []
    window_starts: list[str] = []

    while current + window_delta <= end_time:
        start_iso = current.isoformat()
        evs  = window_events.get(start_iso, [])
        snap = _build_snapshot(node_list, node_index, edge_list, evs)
        windows.append(snap)
        window_starts.append(start_iso)
        current += window_delta

    if not windows:
        print("[graph_dataset] No windows built — check Neo4j data")
        return []

    # Build T_SNAPSHOTS-length samples
    for i in range(len(windows) - T_SNAPSHOTS + 1):
        snaps = windows[i: i + T_SNAPSHOTS]
        # Sample label: anomaly present in the last snapshot
        label = float(snaps[-1].y.max().item() > 0)
        # Spatial OOD: any node in last snapshot in eastern zone
        is_ood = any(
            float(node_list[j].get("lon") or 0) > SPATIAL_OOD_LON
            and snaps[-1].y[j].item() > 0
            for j in range(len(node_list))
        )
        samples.append({
            "snapshots": snaps,
            "label": label,
            "window_start": window_starts[i],
            "is_ood_spatial": is_ood,
            "semester": semester,
        })

    print(f"[graph_dataset] Sem{semester}: {len(samples)} samples "
          f"({sum(s['label'] for s in samples):.0f} positive)")
    return samples


def split_dataset(samples: list[dict]) -> tuple[list, list, list, list]:
    """
    Returns (train, val, test_temporal, test_spatial_ood).
    train/val come from Semester 1 only.
    test_temporal = Semester 2.
    test_spatial_ood = eastern campus samples from either semester.
    """
    sem1 = [s for s in samples if s["semester"] == 1]
    sem2 = [s for s in samples if s["semester"] == 2]

    split_idx = int(len(sem1) * 0.8)
    train = sem1[:split_idx]
    val = sem1[split_idx:]
    test_temporal = sem2
    test_spatial_ood = [s for s in samples if s["is_ood_spatial"]]

    print(f"[split] train={len(train)} val={len(val)} "
          f"test_temporal={len(test_temporal)} test_spatial_ood={len(test_spatial_ood)}")
    return train, val, test_temporal, test_spatial_ood
