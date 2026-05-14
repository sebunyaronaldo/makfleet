"""
MakFleet Causal BI Dashboard — Streamlit entry point.

4 tabs:
  1. Campus Map      — traffic demand heatmap + anomaly markers
  2. Anomaly Detection — time-series + flagged events table
  3. Causal Evidence  — subgraph explanation for selected anomaly

Gap 3: model loading never crashes the dashboard.
       load_scorer() returns NeuralScorer OR RuleBasedScorer transparently.
"""

import sys, os
from pathlib import Path

# Ensure project root is on sys.path when running from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# Inject Streamlit secrets into environment variables so config/settings.py picks them up
for _key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

st.set_page_config(
    page_title="MakFleet — Semantic BI Dashboard",
    page_icon="🛵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy imports (keep startup fast; import heavy libs only when needed)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_scorer():
    try:
        from phase3_model.anomaly_scorer import load_scorer
        return load_scorer()
    except Exception:
        from phase3_model.anomaly_scorer import RuleBasedScorer
        return RuleBasedScorer(), "Rule-based (fallback)"


@st.cache_data(ttl=600, show_spinner="Fetching anomalies…")
def _get_anomalies(semester):
    from phase4_dashboard.causal_evidence import get_recent_anomalies
    return get_recent_anomalies(limit=50, semester=semester if semester != 0 else None)


@st.cache_data(ttl=600, show_spinner="Fetching demand…")
def _get_demand(semester):
    from phase4_dashboard.causal_evidence import get_demand_aggregates
    return get_demand_aggregates(semester=semester if semester != 0 else None)


@st.cache_data(ttl=60)
def _get_trajectory(vehicle_id, semester):
    from phase4_dashboard.causal_evidence import get_vehicle_trajectory
    return get_vehicle_trajectory(vehicle_id, semester=semester if semester != 0 else None)


@st.cache_data(ttl=300)
def _get_subgraph(event_id):
    from phase4_dashboard.causal_evidence import get_anomaly_subgraph
    return get_anomaly_subgraph(event_id)





def _map_html(folium_map) -> str | None:
    if folium_map is None:
        return None
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            folium_map.save(f.name)
            fname = f.name
        html = open(fname, encoding="utf-8").read()
        os.unlink(fname)
        return html
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛵 MakFleet ST-KG")
st.sidebar.markdown("*Semantic-Aware Spatio-Temporal Data Warehouse*")
st.sidebar.divider()

role = st.sidebar.selectbox(
    "Access Role",
    ["admin", "analyst", "public"],
    help="Controls data visibility (RBAC)"
)

semester_opt = st.sidebar.selectbox(
    "Semester filter",
    [0, 1, 2],
    format_func=lambda x: {0: "All", 1: "Semester 1 (Jan–Apr)", 2: "Semester 2 (Aug–Oct)"}[x],
)

vehicle_ids = [f"V{i:03d}" for i in range(1, 16)]
selected_vehicle = st.sidebar.selectbox("Vehicle", ["All"] + vehicle_ids)

# Load scorer (Gap 3: never crashes)
scorer, scorer_label = _load_scorer()
st.sidebar.divider()
st.sidebar.caption(f"Scorer: **{scorer_label}**")

from phase4_dashboard.rbac import ROLE_DESCRIPTIONS, audit_log
st.sidebar.caption(ROLE_DESCRIPTIONS.get(role, ""))
audit_log(role, "dashboard_access")

# Neo4j connectivity status indicator
try:
    from phase4_dashboard.causal_evidence import _get_driver
    _test_driver = _get_driver()
    if _test_driver:
        _test_driver.close()
        st.sidebar.success("Neo4j: Connected", icon="🟢")
    else:
        st.sidebar.warning("Neo4j: Unavailable — Aura instance may be paused. Visit console.neo4j.io to resume.", icon="🟡")
except Exception:
    st.sidebar.warning("Neo4j: Unavailable", icon="🟡")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🗺️ Campus Map",
    "⚠️ Anomaly Detection",
    "🔍 Causal Evidence",
])


# ============================================================
# TAB 1 — Campus Map
# ============================================================
with tab1:
    st.header("Campus Traffic & Anomaly Map")

    col1, col2, col3 = st.columns(3)
    anomalies = _get_anomalies(semester_opt)
    demand = _get_demand(semester_opt)

    col1.metric("Total Anomalies", len(anomalies))
    col2.metric("Demand Zones", len(demand))
    col3.metric("Active Role", role.upper())

    with st.spinner("Rendering campus map…"):
        try:
            from phase4_dashboard.map_view import render_campus_map
            # Cap markers to 50 to keep map responsive
            fmap = render_campus_map(demand, anomalies[:50], role)
            html = _map_html(fmap)
            if html:
                st.components.v1.html(html, height=520, scrolling=False)
            else:
                st.info("Install `folium` to enable map: `pip install folium`")
        except Exception as e:
            st.warning(f"Map render error: {e}")

    if role in ("admin", "analyst") and anomalies:
        import pandas as pd
        st.subheader("Recent Anomalies")
        df = pd.DataFrame(anomalies)[
            ["event_id", "vehicle_id", "event_type", "severity_score", "speed_kmh", "timestamp"]
        ].head(20)
        st.dataframe(df, use_container_width=True)


# ============================================================
# TAB 2 — Anomaly Detection
# ============================================================
with tab2:
    st.header("Anomaly Detection — Temporal Analysis")

    try:
     anomalies2 = _get_anomalies(semester_opt)
    except Exception as e:
     st.error(f"Could not load anomalies: {e}")
     anomalies2 = []
    if not anomalies2:
        st.info("No anomaly data in Neo4j yet. Run the pipeline first.")
    else:
        import pandas as pd
        import plotly.express as px

        df_a = pd.DataFrame(anomalies2)

        if "timestamp" in df_a.columns:
            df_a["timestamp"] = pd.to_datetime(df_a["timestamp"], errors="coerce", utc=True)
            df_a["hour"] = df_a["timestamp"].dt.floor("h")
            hourly = df_a.groupby(["hour", "event_type"]).size().reset_index(name="count")

            fig = px.bar(
                hourly,
                x="hour",
                y="count",
                color="event_type",
                title="Anomaly Events per Hour by Type",
                color_discrete_map={
                    "harsh_braking": "#e74c3c",
                    "rapid_acceleration": "#e67e22",
                    "speeding": "#c0392b",
                    "idling": "#3498db",
                },
                labels={"hour": "Time", "count": "Event Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Severity distribution
        if "severity_score" in df_a.columns:
            fig2 = px.histogram(
                df_a, x="severity_score", nbins=20,
                title="Severity Score Distribution",
                color_discrete_sequence=["#e74c3c"],
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Full events table
        st.subheader("Flagged Events")
        display_cols = [c for c in [
            "event_id", "vehicle_id", "driver_id", "event_type",
            "severity_score", "speed_kmh", "ax", "timestamp",
        ] if c in df_a.columns]
        st.dataframe(
            df_a[display_cols].sort_values("severity_score", ascending=False).head(100),
            use_container_width=True,
        )


# ============================================================
# TAB 3 — Causal Evidence
# ============================================================
with tab3:
    st.header("Causal Evidence Generation")
    st.markdown(
        "_This tab demonstrates semantic evidence generation — answering **why** an anomaly "
        "was flagged, not just **what** was recorded._"
    )

    from phase4_dashboard.rbac import check_access
    if not check_access(role, "causal_subgraph"):
        st.warning(f"Role **{role}** does not have access to causal evidence. Switch to analyst or admin.")
    else:
        anomalies3 = _get_anomalies(semester_opt)
        if not anomalies3:
            st.info("No anomaly data found. Run the pipeline and loader first.")
        else:
            import pandas as pd

            df3 = pd.DataFrame(anomalies3)
            options = df3["event_id"].tolist()
            labels = {
                row["event_id"]: f"{row.get('event_type','?')} | {row.get('vehicle_id','?')} | {str(row.get('timestamp',''))[:19]}"
                for _, row in df3.iterrows()
            }

            selected_eid = st.selectbox(
                "Select anomalous event to investigate:",
                options,
                format_func=lambda x: labels.get(x, x),
            )

            if selected_eid:
                subgraph = _get_subgraph(selected_eid)
                ev = subgraph.get("event", {})
                loc = subgraph.get("location", {})
                drv = subgraph.get("driver", {})
                veh = subgraph.get("vehicle", {})
                neighbors = subgraph.get("neighbors", [])
                past = subgraph.get("past_anomalies", [])

                # Score the event
                result = scorer.score(ev)

                col_a, col_b = st.columns([1, 1])

                with col_a:
                    st.subheader("Event Details")
                    st.markdown(f"**Type:** `{ev.get('event_type', 'unknown')}`")
                    st.markdown(f"**Confidence:** {result.confidence:.2%}")
                    st.markdown(f"**Severity:** {ev.get('severity_score', 0):.2f}")
                    st.markdown(f"**Speed:** {ev.get('speed_kmh', 0):.1f} km/h")
                    st.markdown(f"**Ax:** {ev.get('ax', 0):.3f} m/s²")
                    st.markdown(f"**Time:** {str(ev.get('timestamp', ''))[:19]}")
                    st.divider()
                    st.subheader("Causal Explanation")
                    st.info(result.explanation)
                    st.caption(f"_Scorer: {result.scorer_type}_")

                with col_b:
                    st.subheader("Driver & Vehicle")
                    if role in ("admin", "analyst"):
                        st.markdown(f"**Driver:** {drv.get('name', 'N/A')}")
                        st.markdown(f"**Risk Profile:** {drv.get('risk_profile', 0):.2f}")
                        st.markdown(f"**Vehicle:** {veh.get('vehicle_id', 'N/A')} — {veh.get('plate', 'N/A')}")
                        if past:
                            st.markdown(f"**Prior anomalies:** {len(past)}")
                    st.subheader("Location")
                    st.markdown(f"**Intersection:** `{loc.get('osm_id', 'N/A')}`")
                    st.markdown(f"**Lat/Lon:** {loc.get('lat', 0):.5f}, {loc.get('lon', 0):.5f}")
                    st.markdown(f"**Adjacent nodes:** {len(neighbors)}")

                # Causal subgraph map
                st.subheader("Causal Subgraph — 2-hop Campus Topology")
                center_lat = float(ev.get("lat") or loc.get("lat") or 0.3382)
                center_lon = float(ev.get("lon") or loc.get("lon") or 32.5701)

                try:
                    from phase4_dashboard.map_view import render_causal_subgraph_map
                    fmap3 = render_causal_subgraph_map(
                        center_lat, center_lon,
                        result.subgraph_nodes or [],
                        result.subgraph_edges or [],
                        event_type=ev.get("event_type", ""),
                    )
                    html3 = _map_html(fmap3)
                    if html3:
                        st.components.v1.html(html3, height=400)
                    else:
                        st.caption("Install folium for map rendering.")
                except Exception as e:
                    st.caption(f"Map render error: {e}")

                # Plotly subgraph network diagram
                if result.subgraph_nodes:
                    import plotly.graph_objects as go
                    node_x = [n.get("lon", 0) for n in result.subgraph_nodes]
                    node_y = [n.get("lat", 0) for n in result.subgraph_nodes]
                    node_labels = [n.get("label", "Node") for n in result.subgraph_nodes]
                    node_colors = [
                        "red" if (abs(n.get("lat", 0) - center_lat) < 0.0002
                                  and abs(n.get("lon", 0) - center_lon) < 0.0002)
                        else ("green" if "Stop" in n.get("label", "") or "Crossing" in n.get("label", "")
                              else "royalblue")
                        for n in result.subgraph_nodes
                    ]

                    fig_net = go.Figure()
                    for edge in result.subgraph_edges:
                        u_id = str(edge.get("u", ""))
                        v_id = str(edge.get("v", ""))
                        u_node = next((n for n in result.subgraph_nodes if str(n.get("osm_id")) == u_id), None)
                        v_node = next((n for n in result.subgraph_nodes if str(n.get("osm_id")) == v_id), None)
                        if u_node and v_node:
                            fig_net.add_trace(go.Scatter(
                                x=[u_node.get("lon"), v_node.get("lon"), None],
                                y=[u_node.get("lat"), v_node.get("lat"), None],
                                mode="lines", line=dict(color="gray", width=1),
                                showlegend=False, hoverinfo="none",
                            ))
                    fig_net.add_trace(go.Scatter(
                        x=node_x, y=node_y, mode="markers+text",
                        text=node_labels, textposition="top center",
                        marker=dict(color=node_colors, size=12, line=dict(color="white", width=1)),
                        hoverinfo="text",
                        name="Campus Nodes",
                    ))
                    fig_net.update_layout(
                        title="Causal Subgraph (2-hop ego-graph)",
                        xaxis_title="Longitude", yaxis_title="Latitude",
                        showlegend=False, height=350,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_net, use_container_width=True)

                # Past anomaly history
                if past and role in ("admin", "analyst"):
                    st.subheader("Driver's Prior Anomaly History")
                    import pandas as pd
                    hist_df = pd.DataFrame(past)[
                        [c for c in ["event_type", "severity_score", "speed_kmh", "timestamp"] if c in pd.DataFrame(past).columns]
                    ].head(10)
                    st.dataframe(hist_df, use_container_width=True)


