"""Privacy filters, styling, and benchmark helpers for the MakFleet dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import DATA_DIR, PRIVACY_EPSILON

BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"

DEMO_CSS = """
<style>
    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .makfleet-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 45%, #0f3460 100%);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        color: #f8f9fa;
    }
    .makfleet-hero h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .makfleet-hero p { margin: 0.35rem 0 0; opacity: 0.88; font-size: 0.95rem; }
    .makfleet-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.75rem;
        margin-right: 0.35rem;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
"""


def inject_demo_styles():
    import streamlit as st
    st.markdown(DEMO_CSS, unsafe_allow_html=True)


def render_hero(scorer_label: str, neo4j_ok: bool):
    import streamlit as st
    status = "Neo4j Connected" if neo4j_ok else "Neo4j Offline — demo data limited"
    st.markdown(
        f"""
        <div class="makfleet-hero">
            <h1>🛵 MakFleet Analytics Platform</h1>
            <p>Semantic-Aware Spatio-Temporal Knowledge Graph · Makerere University Campus</p>
            <p style="margin-top:0.6rem">
                <span class="makfleet-badge">{scorer_label}</span>
                <span class="makfleet-badge">{status}</span>
                <span class="makfleet-badge">Privacy-by-Design</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def privatize_demand(demand: list[dict], role: str) -> list[dict]:
    """Apply differential privacy to aggregate trip counts for public role."""
    if role != "public":
        return demand
    from phase2_pipeline.privacy import dp_noise
    out = []
    for d in demand:
        raw = float(d.get("trip_count") or 0)
        noisy = max(0.0, dp_noise(raw, sensitivity=max(5.0, raw * 0.1), epsilon=PRIVACY_EPSILON))
        out.append({**d, "trip_count": round(noisy, 1), "_dp_applied": True})
    return out


def privatize_anomalies(anomalies: list[dict], role: str) -> list[dict]:
    """Role-appropriate anomaly records for tables and maps."""
    from phase2_pipeline.privacy import apply_k_anonymity, filter_records

    if role == "admin":
        return anomalies
    if role == "analyst":
        return filter_records(anomalies, "analyst")
    return []


def privatize_trajectory(points: list[dict], role: str) -> list[dict]:
    if role == "admin":
        return points
    if role == "analyst":
        return [
            {
                "lat": p["lat"],
                "lon": p["lon"],
                "speed_kmh": p.get("speed_kmh"),
                "timestamp": p.get("timestamp"),
                "event_type": p.get("event_type"),
                "is_anomaly": p.get("is_anomaly"),
            }
            for p in points
            if p.get("lat") and p.get("lon")
        ]
    minimal = [{"lat": p["lat"], "lon": p["lon"]} for p in points if p.get("lat")]
    return apply_k_anonymity(minimal)


def load_benchmark_results() -> dict | None:
    if BENCHMARK_RESULTS_PATH.exists():
        try:
            return json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_benchmark_results(results: dict) -> None:
    from datetime import datetime, timezone
    results = dict(results)
    results["_cached_at"] = datetime.now(timezone.utc).isoformat()
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_and_cache_benchmarks() -> dict:
    from benchmarks.star_schema_benchmark import run_all_benchmarks
    results = run_all_benchmarks()
    save_benchmark_results(results)
    return results
