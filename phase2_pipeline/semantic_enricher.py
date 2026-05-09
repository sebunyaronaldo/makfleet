"""
Semantic event enricher — the core 'semantic-aware' contribution.

Classifies each map-matched GPS record into a meaningful semantic event
based on accelerometer readings, speed, and spatial context (proximity to
known zebra crossings and bodaboda stops from the ontology).

The safe_stop vs harsh_braking distinction is the key academic differentiator:
the system differentiates a sudden stop near a zebra crossing (ContextualStop)
from reckless braking on an open road (anomalous TelematicsEvent) by querying
the spatial ontology — not by relying on speed/acceleration alone.
"""

import math
from collections import deque

import networkx as nx
import numpy as np

from config.settings import (
    ACCEL_HARSH_BRAKE_THRESHOLD, ACCEL_RAPID_ACCEL_THRESHOLD,
    IDLE_SPEED_THRESHOLD_KMH,
)
from ontology.makfleet_ontology import SEVERITY, IS_ANOMALY
from phase1_ingestion.campus_graph import get_landmark_nodes, node_coords

# Distance threshold for spatial context check (safe_stop detection)
SAFE_STOP_RADIUS_M = 20.0   # metres
SUSTAINED_BRAKE_TICKS = 2   # consecutive ticks for harsh_braking confirmation


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _build_safe_zone_coords(G: nx.MultiDiGraph) -> list[tuple[float, float]]:
    """Return (lat, lon) for every BodaBodaStop and ZebraCrossing node."""
    lm = get_landmark_nodes(G)
    coords = []
    for ltype in ("BodaBodaStop", "ZebraCrossing"):
        for nid in lm.get(ltype, []):
            coords.append(node_coords(G, nid))
    return coords


def _is_near_safe_zone(lat, lon, safe_zones: list[tuple[float, float]]) -> bool:
    return any(
        _haversine_m(lat, lon, sz_lat, sz_lon) <= SAFE_STOP_RADIUS_M
        for sz_lat, sz_lon in safe_zones
    )


def _get_edge_maxspeed(G: nx.MultiDiGraph, u: int, v: int) -> float:
    try:
        data = G.edges[u, v, 0]
        spd = data.get("maxspeed", "30")
        if isinstance(spd, list):
            spd = spd[0]
        return float(str(spd).replace(" mph", "").replace(" km/h", ""))
    except (KeyError, ValueError, TypeError):
        return 30.0


# ---------------------------------------------------------------------------
# Per-vehicle enrichment (processes records in time order)
# ---------------------------------------------------------------------------

def enrich_vehicle(
    records: list[dict],
    G: nx.MultiDiGraph,
    safe_zones: list[tuple[float, float]],
) -> list[dict]:
    """
    Classify each record for one vehicle.  Must be called with records sorted
    by timestamp (map_matcher guarantees this for per-vehicle output).
    """
    ax_window: deque[float] = deque(maxlen=SUSTAINED_BRAKE_TICKS + 1)
    idle_start_idx = None

    for i, rec in enumerate(records):
        ax = rec["ax"]
        speed = rec["speed_kmh"]
        lat, lon = rec["lat"], rec["lon"]
        u = rec.get("matched_edge_u", rec.get("edge_u", 0))
        v = rec.get("matched_edge_v", rec.get("edge_v", 0))

        ax_window.append(ax)

        # Sustained harsh braking: last SUSTAINED_BRAKE_TICKS all below threshold
        is_harsh_brake = (
            len(ax_window) >= SUSTAINED_BRAKE_TICKS
            and all(a < ACCEL_HARSH_BRAKE_THRESHOLD for a in list(ax_window)[-SUSTAINED_BRAKE_TICKS:])
        )

        is_rapid_accel = ax > ACCEL_RAPID_ACCEL_THRESHOLD

        is_idle = speed < IDLE_SPEED_THRESHOLD_KMH
        if is_idle and idle_start_idx is None:
            idle_start_idx = i
        elif not is_idle:
            idle_start_idx = None

        # Edge speed limit
        try:
            limit = _get_edge_maxspeed(G, u, v)
        except Exception:
            limit = 30.0
        is_speeding = speed > limit * 1.2

        # Classify
        if is_idle or (is_harsh_brake and not is_rapid_accel):
            # Check spatial context before committing to harsh_braking
            near_safe = _is_near_safe_zone(lat, lon, safe_zones)
            if near_safe and is_idle:
                event_type = "safe_stop"
            elif is_harsh_brake and not near_safe:
                event_type = "harsh_braking"
            elif is_idle:
                event_type = "idling"
            else:
                event_type = "normal_travel"
        elif is_rapid_accel and not is_harsh_brake:
            event_type = "rapid_acceleration"
        elif is_speeding:
            event_type = "speeding"
        else:
            event_type = "normal_travel"

        rec["event_type"] = event_type
        rec["severity_score"] = SEVERITY[event_type]
        rec["is_anomaly"] = IS_ANOMALY[event_type]

    return records


def enrich_all(records: list[dict], G: nx.MultiDiGraph) -> list[dict]:
    """Group by vehicle, enrich each group, flatten."""
    from collections import defaultdict
    safe_zones = _build_safe_zone_coords(G)

    by_vehicle: dict[str, list] = defaultdict(list)
    for r in records:
        by_vehicle[r["vehicle_id"]].append(r)

    enriched = []
    anomaly_count = 0
    safe_stop_count = 0
    for vid, recs in by_vehicle.items():
        recs_sorted = sorted(recs, key=lambda x: x["timestamp"])
        result = enrich_vehicle(recs_sorted, G, safe_zones)
        enriched.extend(result)
        anomaly_count += sum(1 for r in result if r["is_anomaly"])
        safe_stop_count += sum(1 for r in result if r["event_type"] == "safe_stop")

    total = len(enriched)
    print(
        f"[enricher] {total:,} records enriched | "
        f"{anomaly_count:,} anomalies ({100*anomaly_count/max(total,1):.1f}%) | "
        f"{safe_stop_count:,} contextual stops"
    )
    return enriched
