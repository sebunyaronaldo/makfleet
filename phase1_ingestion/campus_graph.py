"""
Pull the real Makerere University road network from OpenStreetMap via OSMnx,
annotate it with campus landmarks, compute graph-theoretic node features,
and cache the result as GraphML so downstream phases never re-download.
"""

import sys
import networkx as nx

try:
    import osmnx as ox
except ImportError:
    print("osmnx not installed. Run: pip install osmnx")
    sys.exit(1)

from config.settings import MAKERERE_BBOX, GRAPH_CACHE_DIR, LANDMARKS

CACHE_FILE = GRAPH_CACHE_DIR / "makerere_campus.graphml"
LANDMARK_ATTR = "landmark_type"   # extra node attribute for BodaBodaStop / ZebraCrossing


def load_campus_graph() -> nx.MultiDiGraph:
    """Return the campus graph, loading from cache if available."""
    if CACHE_FILE.exists():
        print(f"[campus_graph] Loading from cache: {CACHE_FILE}")
        G = ox.load_graphml(CACHE_FILE)
        return G

    print("[campus_graph] Downloading Makerere road network from OpenStreetMap…")
    south, west, north, east = MAKERERE_BBOX
    # osmnx 2.x API: bbox=(west, south, east, north) i.e. (left, bottom, right, top)
    # network_type="all" captures internal campus footways/service roads, not just driveable roads
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type="all",
        retain_all=True,
    )
    print(f"[campus_graph] Downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    G = _add_landmarks(G)
    G = _compute_node_features(G)

    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, CACHE_FILE)
    print(f"[campus_graph] Saved to cache: {CACHE_FILE}")
    return G


def _add_landmarks(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Inject synthetic nodes for known Makerere landmarks (bodaboda stops,
    zebra crossings) that OSM may not model explicitly.  Each landmark is
    snapped to the nearest existing OSM node so the graph stays connected.
    """
    for name, (lat, lon, ltype) in LANDMARKS.items():
        nearest_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        G.nodes[nearest_node][LANDMARK_ATTR] = ltype
        G.nodes[nearest_node]["landmark_name"] = name
        G.nodes[nearest_node]["landmark_lat"] = lat
        G.nodes[nearest_node]["landmark_lon"] = lon
    print(f"[campus_graph] Annotated {len(LANDMARKS)} landmarks")
    return G


def _compute_node_features(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Pre-compute betweenness centrality and degree centrality for every node.
    These become ST-GNN node features (coordinates L3 data geometry requirement).
    Using the undirected view for betweenness keeps computation tractable.
    """
    U = G.to_undirected()
    bc = nx.betweenness_centrality(U, normalized=True)
    dc = nx.degree_centrality(U)
    for node in G.nodes:
        G.nodes[node]["betweenness"] = round(bc.get(node, 0.0), 6)
        G.nodes[node]["degree_centrality"] = round(dc.get(node, 0.0), 6)
    print("[campus_graph] Node centrality features computed")
    return G


def get_landmark_nodes(G: nx.MultiDiGraph) -> dict[str, list[int]]:
    """Return {landmark_type: [node_id, …]} for spatial proximity checks."""
    result: dict[str, list[int]] = {}
    for nid, data in G.nodes(data=True):
        ltype = data.get(LANDMARK_ATTR)
        if ltype:
            result.setdefault(ltype, []).append(nid)
    return result


def node_coords(G: nx.MultiDiGraph, node_id: int) -> tuple[float, float]:
    """Return (lat, lon) for a node."""
    d = G.nodes[node_id]
    return float(d["y"]), float(d["x"])


if __name__ == "__main__":
    G = load_campus_graph()
    landmarks = get_landmark_nodes(G)
    print(f"Landmark nodes: { {k: len(v) for k, v in landmarks.items()} }")
