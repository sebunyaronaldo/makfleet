"""Role-based access control and audit logging for the MakFleet dashboard."""

import json
import hashlib
from datetime import datetime, timezone

from config.settings import AUDIT_LOG

PERMISSIONS: dict[str, set[str]] = {
    "admin":   {
        "raw_gps", "full_trajectory", "driver_identity", "audit_log",
        "event_types", "map_matched_trajectory", "anonymized_gps",
        "aggregates", "heatmap", "causal_subgraph",
    },
    "analyst": {
        "event_types", "map_matched_trajectory", "anonymized_gps",
        "aggregates", "heatmap", "causal_subgraph",
    },
    "public":  {"aggregates", "heatmap"},
}

ROLE_DESCRIPTIONS = {
    "admin":   "Full access — raw GPS, driver identity, audit log",
    "analyst": "Anonymized trajectory, event types, causal evidence",
    "public":  "Aggregate heatmaps only — k-anonymized",
}


def check_access(role: str, resource: str) -> bool:
    return resource in PERMISSIONS.get(role, set())


def require_access(role: str, resource: str) -> None:
    if not check_access(role, resource):
        raise PermissionError(f"Role '{role}' does not have access to '{resource}'")


def audit_log(role: str, resource: str, params: dict | None = None) -> None:
    params = params or {}
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "resource": resource,
        "params_hash": hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
