"""
Hidden Markov Model map-matcher.

Snaps noisy GPS coordinates to the nearest road edge in the campus graph
using emission + transition probabilities and Viterbi decoding over a
sliding window of observations.

Sliding window (size=10) keeps the algorithm tractable:
full Viterbi is O(N² × T) where N = candidate edges; the window caps T.
This is the standard engineering trade-off for real-time map-matching
(Newson & Krumm 2009 approach, simplified for single-vehicle trajectories).
"""

import math
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from config.settings import GPS_NOISE_STD

CANDIDATE_RADIUS_DEG = 0.0003   # ~33 metres in degrees latitude
WINDOW_SIZE = 10                 # Viterbi sliding window


# ---------------------------------------------------------------------------
# Spatial index over edge midpoints
# ---------------------------------------------------------------------------

def build_edge_index(G: nx.MultiDiGraph) -> tuple[KDTree, list[tuple]]:
    """
    Return a KD-tree over edge midpoints and a parallel list of
    (u, v, key) tuples so we can look up the edge from the KD-tree result index.
    """
    points = []
    edges = []
    for u, v, key, data in G.edges(data=True, keys=True):
        y_u = G.nodes[u]["y"]
        x_u = G.nodes[u]["x"]
        y_v = G.nodes[v]["y"]
        x_v = G.nodes[v]["x"]
        mid_lat = (y_u + y_v) / 2
        mid_lon = (x_u + x_v) / 2
        points.append([mid_lat, mid_lon])
        edges.append((u, v, key))
    return KDTree(np.array(points)), edges


def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
    """Euclidean distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _emission_prob(obs_lat, obs_lon, u, v, G, sigma=GPS_NOISE_STD) -> float:
    """Gaussian emission probability based on distance from GPS point to edge."""
    y_u, x_u = G.nodes[u]["y"], G.nodes[u]["x"]
    y_v, x_v = G.nodes[v]["y"], G.nodes[v]["x"]
    dist = _point_to_segment_dist(obs_lat, obs_lon, y_u, x_u, y_v, x_v)
    return math.exp(-(dist ** 2) / (2 * sigma ** 2))


def _transition_prob(G, u1, v1, u2, v2) -> float:
    """
    Transition probability: edges that share a node get high probability,
    non-adjacent edges get low probability proportional to path distance.
    """
    shared = {u1, v1} & {u2, v2}
    if shared:
        return 1.0
    try:
        d = nx.shortest_path_length(G, v1, u2, weight="length")
        return math.exp(-d / 100.0)
    except nx.NetworkXNoPath:
        return 1e-9


# ---------------------------------------------------------------------------
# Viterbi over sliding window
# ---------------------------------------------------------------------------

def _viterbi_window(
    observations: list[tuple[float, float]],
    candidates: list[list[tuple]],
    G: nx.MultiDiGraph,
) -> list[Optional[tuple]]:
    """
    Run Viterbi over a window of observations.
    candidates[t] = list of (u, v, key) candidate edges for observation t.
    Returns the most likely edge sequence.
    """
    T = len(observations)
    if T == 0:
        return []

    # Init
    viterbi = [{}]
    backtrack = [{}]

    obs_lat, obs_lon = observations[0]
    for edge in candidates[0]:
        u, v, _ = edge
        p = _emission_prob(obs_lat, obs_lon, u, v, G)
        viterbi[0][edge] = p
        backtrack[0][edge] = None

    # Forward pass
    for t in range(1, T):
        viterbi.append({})
        backtrack.append({})
        obs_lat, obs_lon = observations[t]
        for edge in candidates[t]:
            u, v, _ = edge
            em = _emission_prob(obs_lat, obs_lon, u, v, G)
            best_prob, best_prev = 0.0, None
            for prev_edge in candidates[t - 1]:
                pu, pv, _ = prev_edge
                tr = _transition_prob(G, pu, pv, u, v)
                prob = viterbi[t - 1].get(prev_edge, 0.0) * tr * em
                if prob > best_prob:
                    best_prob, best_prev = prob, prev_edge
            viterbi[t][edge] = best_prob if best_prob > 0 else em * 1e-9
            backtrack[t][edge] = best_prev

    # Backtrack
    if not viterbi[T - 1]:
        return [None] * T
    best_last = max(viterbi[T - 1], key=viterbi[T - 1].get)
    path = [best_last]
    for t in range(T - 1, 0, -1):
        prev = backtrack[t].get(path[-1]) if path[-1] is not None else None
        path.append(prev)
    return list(reversed(path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_trajectory(
    records: list[dict],
    G: nx.MultiDiGraph,
    edge_tree: KDTree,
    edge_list: list[tuple],
) -> list[dict]:
    """
    Map-match a list of raw telemetry records for one vehicle.
    Enriches each record with:
        matched_edge_u, matched_edge_v, snap_distance_m, map_match_confidence
    """
    obs = [(r["lat"], r["lon"]) for r in records]
    candidate_sets = []

    for lat, lon in obs:
        idxs = edge_tree.query_ball_point([lat, lon], r=CANDIDATE_RADIUS_DEG)
        if not idxs:
            # Fall back to nearest single edge
            _, nearest_idx = edge_tree.query([lat, lon])
            idxs = [nearest_idx]
        candidate_sets.append([edge_list[i] for i in idxs])

    # Process in sliding windows
    results: list[Optional[tuple]] = []
    step = WINDOW_SIZE
    for start in range(0, len(obs), step):
        end = min(start + step, len(obs))
        matched = _viterbi_window(obs[start:end], candidate_sets[start:end], G)
        results.extend(matched)

    # Enrich records
    for i, record in enumerate(records):
        edge = results[i] if i < len(results) else None
        if edge is None:
            record["matched_edge_u"] = record.get("edge_u", 0)
            record["matched_edge_v"] = record.get("edge_v", 0)
            record["snap_distance_m"] = 999.0
            record["map_match_confidence"] = 0.0
        else:
            u, v, _ = edge
            y_u, x_u = G.nodes[u]["y"], G.nodes[u]["x"]
            y_v, x_v = G.nodes[v]["y"], G.nodes[v]["x"]
            dist_deg = _point_to_segment_dist(
                record["lat"], record["lon"], y_u, x_u, y_v, x_v
            )
            dist_m = dist_deg * 111_320  # ~metres per degree at equator
            em = _emission_prob(record["lat"], record["lon"], u, v, G)
            record["matched_edge_u"] = int(u)
            record["matched_edge_v"] = int(v)
            record["snap_distance_m"] = round(dist_m, 2)
            record["map_match_confidence"] = round(min(em * 10, 1.0), 4)
    return records


def match_all_vehicles(
    records: list[dict],
    G: nx.MultiDiGraph,
) -> list[dict]:
    """Group records by vehicle, map-match each, return flattened results."""
    from collections import defaultdict
    edge_tree, edge_list = build_edge_index(G)

    by_vehicle: dict[str, list] = defaultdict(list)
    for r in records:
        by_vehicle[r["vehicle_id"]].append(r)

    matched = []
    for vid, recs in by_vehicle.items():
        recs_sorted = sorted(recs, key=lambda x: x["timestamp"])
        matched.extend(match_trajectory(recs_sorted, G, edge_tree, edge_list))
    print(f"[map_matcher] Matched {len(matched):,} records across {len(by_vehicle)} vehicles")
    return matched
