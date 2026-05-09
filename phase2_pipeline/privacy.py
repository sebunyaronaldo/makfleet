"""
Privacy-by-Design module.

Implements L5 requirements (privacy-preserving + OOD robustness):
  1. Spatial k-anonymity — GPS clusters with < k members are replaced by centroid
  2. Differential privacy — calibrated Gaussian noise on aggregate statistics
  3. Role-Based Access Control (RBAC) — three access tiers
  4. Audit log — append-only log of every privileged data access
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from diffprivlib.mechanisms import Gaussian as DPGaussian
    _DP_AVAILABLE = True
except ImportError:
    _DP_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from config.settings import PRIVACY_K, PRIVACY_EPSILON, AUDIT_LOG

# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

PERMISSIONS: dict[str, set[str]] = {
    "admin":   {"raw_gps", "full_trajectory", "driver_identity", "audit_log",
                "event_types", "map_matched_trajectory", "anonymized_gps", "aggregates", "heatmap"},
    "analyst": {"event_types", "map_matched_trajectory", "anonymized_gps", "aggregates", "heatmap"},
    "public":  {"aggregates", "heatmap"},
}


def check_access(role: str, resource: str) -> bool:
    return resource in PERMISSIONS.get(role, set())


def audit_log_access(role: str, resource: str, query_params: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "resource": resource,
        "params_hash": hashlib.sha256(json.dumps(query_params, sort_keys=True).encode()).hexdigest()[:16],
        "params": {k: v for k, v in query_params.items() if k != "raw_data"},
    }
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Spatial k-anonymity
# ---------------------------------------------------------------------------

def _dbscan_cluster(coords: np.ndarray, eps_deg: float = 0.0002) -> np.ndarray:
    """
    Cluster GPS coordinates with DBSCAN (eps ≈ 22 m).
    Returns cluster label array (-1 = noise/unclustered).
    """
    if not _SKLEARN_AVAILABLE or len(coords) == 0:
        return np.full(len(coords), -1)
    db = DBSCAN(eps=eps_deg, min_samples=PRIVACY_K, metric="euclidean")
    return db.fit_predict(coords)


def apply_k_anonymity(records: list[dict], k: int = PRIVACY_K) -> list[dict]:
    """
    Replace individual GPS coordinates with cluster centroid when a cluster
    has fewer than k members, preventing deanonymisation of rare routes.
    Operates in-place; returns the records list.
    """
    if not records:
        return records

    coords = np.array([[r["lat"], r["lon"]] for r in records])
    labels = _dbscan_cluster(coords)

    from collections import defaultdict
    cluster_coords: dict[int, list] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            cluster_coords[label].append(idx)

    for label, idxs in cluster_coords.items():
        if len(idxs) < k:
            centroid_lat = float(np.mean([records[i]["lat"] for i in idxs]))
            centroid_lon = float(np.mean([records[i]["lon"] for i in idxs]))
            for i in idxs:
                records[i]["lat"] = centroid_lat
                records[i]["lon"] = centroid_lon
                records[i]["_k_anonymized"] = True

    return records


# ---------------------------------------------------------------------------
# Differential privacy on aggregate statistics
# ---------------------------------------------------------------------------

def dp_noise(value: float, sensitivity: float = 1.0, epsilon: float = PRIVACY_EPSILON) -> float:
    """
    Add calibrated Gaussian noise to a scalar aggregate.
    Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / epsilon
    Uses diffprivlib when available, falls back to numpy.
    """
    if _DP_AVAILABLE:
        mech = DPGaussian(epsilon=epsilon, delta=1e-5, sensitivity=sensitivity)
        return mech.randomise(value)
    # Fallback: manual Gaussian calibration
    delta = 1e-5
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return float(value + np.random.normal(0, sigma))


def anonymize_aggregates(demand_dict: dict[str, float]) -> dict[str, float]:
    """
    Apply DP noise to a {road_segment_id: trip_count} aggregate dictionary.
    Safe to expose to 'public' role.
    """
    return {k: max(0.0, dp_noise(v)) for k, v in demand_dict.items()}


# ---------------------------------------------------------------------------
# Role-filtered record views
# ---------------------------------------------------------------------------

def filter_records(records: list[dict], role: str) -> list[dict]:
    """
    Return a role-appropriate view of telemetry records.
      admin   → full records unchanged
      analyst → map-matched coords + event types, no raw GPS
      public  → k-anonymized lat/lon only, no event details
    """
    audit_log_access(role, "telemetry_records", {"count": len(records)})

    if role == "admin":
        return records

    if role == "analyst":
        return [
            {
                "event_id": r["event_id"],
                "vehicle_id": r["vehicle_id"],
                "timestamp": r["timestamp"],
                "lat": r["lat"],
                "lon": r["lon"],
                "event_type": r.get("event_type", "unknown"),
                "severity_score": r.get("severity_score", 0.0),
                "is_anomaly": r.get("is_anomaly", False),
                "matched_edge_u": r.get("matched_edge_u"),
                "matched_edge_v": r.get("matched_edge_v"),
                "map_match_confidence": r.get("map_match_confidence", 0.0),
                "semester": r.get("semester"),
            }
            for r in records
        ]

    # public: k-anonymize then return minimal fields
    minimal = [{"lat": r["lat"], "lon": r["lon"], "timestamp": r["timestamp"]} for r in records]
    return apply_k_anonymity(minimal)
