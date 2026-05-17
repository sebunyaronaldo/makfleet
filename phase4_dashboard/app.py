"""
MakFleet Analytics Platform — Streamlit dashboard.

Tabs: Campus Map | Fleet Trajectories | Anomaly Analytics | Causal Evidence | Benchmarks
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

for _key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

st.set_page_config(
    page_title="MakFleet Analytics",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

from phase4_dashboard.dashboard_data import (
    inject_demo_styles,
    load_benchmark_results,
    privatize_anomalies,
    privatize_demand,
    privatize_trajectory,
    render_hero,
    run_and_cache_benchmarks,
)
from phase4_dashboard.rbac import ROLE_DESCRIPTIONS, audit_log, check_access


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_scorer():
    try:
        from phase3_model.anomaly_scorer import load_scorer
        return load_scorer()
    except Exception:
        from phase3_model.anomaly_scorer import RuleBasedScorer
        return RuleBasedScorer(), "Rule-based (fallback)"


def _sem(semester_opt: int) -> int | None:
    return semester_opt if semester_opt != 0 else None


def _veh(selected: str) -> str | None:
    return None if selected == "All" else selected


@st.cache_data(ttl=300, show_spinner=False)
def _kpis(semester_opt: int):
    from phase4_dashboard.causal_evidence import get_kpi_summary
    return get_kpi_summary(_sem(semester_opt))


@st.cache_data(ttl=300, show_spinner=False)
def _anomalies(semester_opt: int, vehicle: str):
    from phase4_dashboard.causal_evidence import get_recent_anomalies
    return get_recent_anomalies(100, _sem(semester_opt), _veh(vehicle))


@st.cache_data(ttl=300, show_spinner=False)
def _demand(semester_opt: int):
    from phase4_dashboard.causal_evidence import get_demand_aggregates
    return get_demand_aggregates(_sem(semester_opt))


@st.cache_data(ttl=300, show_spinner=False)
def _landmarks():
    from phase4_dashboard.causal_evidence import get_landmarks
    return get_landmarks()


@st.cache_data(ttl=300, show_spinner=False)
def _semester_cmp():
    from phase4_dashboard.causal_evidence import get_semester_comparison
    return get_semester_comparison()


@st.cache_data(ttl=300, show_spinner=False)
def _event_types(semester_opt: int):
    from phase4_dashboard.causal_evidence import get_event_type_distribution
    return get_event_type_distribution(_sem(semester_opt))


@st.cache_data(ttl=300, show_spinner=False)
def _hourly(semester_opt: int, vehicle: str):
    from phase4_dashboard.causal_evidence import get_hourly_anomaly_breakdown
    return get_hourly_anomaly_breakdown(_sem(semester_opt), _veh(vehicle))


@st.cache_data(ttl=300, show_spinner=False)
def _trajectory(vehicle: str, semester_opt: int):
    from phase4_dashboard.causal_evidence import get_vehicle_trajectory
    return get_vehicle_trajectory(vehicle, _sem(semester_opt))


@st.cache_data(ttl=300, show_spinner=False)
def _subgraph(event_id: str):
    from phase4_dashboard.causal_evidence import get_anomaly_subgraph
    return get_anomaly_subgraph(event_id)


def _map_html(folium_map) -> str | None:
    if folium_map is None:
        return None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            folium_map.save(f.name)
            fname = f.name
        html = open(fname, encoding="utf-8").read()
        os.unlink(fname)
        return html
    except Exception:
        return None


def _neo4j_ok() -> bool:
    try:
        from phase4_dashboard.causal_evidence import get_pooled_driver
        return get_pooled_driver() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

inject_demo_styles()

st.sidebar.title("🛵 MakFleet")
st.sidebar.markdown("**Analytics Platform**")
st.sidebar.divider()

role = st.sidebar.selectbox("Access Role", ["admin", "analyst", "public"])
semester_opt = st.sidebar.selectbox(
    "Semester",
    [0, 1, 2],
    format_func=lambda x: {0: "All semesters", 1: "Semester 1 (Jan–Apr)", 2: "Semester 2 (Aug–Oct)"}[x],
)
vehicle_ids = [f"V{i:03d}" for i in range(1, 16)]
selected_vehicle = st.sidebar.selectbox("Vehicle filter", ["All"] + vehicle_ids)

scorer, scorer_label = _load_scorer()
st.sidebar.divider()
st.sidebar.caption(f"**Scorer:** {scorer_label}")
st.sidebar.caption(ROLE_DESCRIPTIONS.get(role, ""))
audit_log(role, "dashboard_access", {"semester": semester_opt, "vehicle": selected_vehicle})

if _neo4j_ok():
    st.sidebar.success("Neo4j connected", icon="✅")
else:
    st.sidebar.warning("Neo4j offline — run `python run_pipeline.py`", icon="⚠️")

if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------

render_hero(scorer_label, _neo4j_ok())
kpis = _kpis(semester_opt)

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Telematics Events", f"{kpis['total_events']:,}")
k2.metric("Flagged Anomalies", f"{kpis['anomalies']:,}")
k3.metric("Anomaly Rate", f"{kpis['anomaly_rate_pct']}%")
k4.metric("Avg Severity", f"{kpis['avg_severity']:.2f}")
k5.metric("Safe Stops", f"{kpis['safe_stops']:,}", help="ContextualStop nodes (ontology)")
k6.metric("Reckless Drivers", kpis["reckless_drivers"])
k7.metric("Top Event", kpis["top_event_type"])
k8.metric("Hot Zone", str(kpis["hot_zone"])[:12])

st.divider()

# Load shared datasets
raw_demand = _demand(semester_opt)
raw_anomalies = _anomalies(semester_opt, selected_vehicle)
demand = privatize_demand(raw_demand, role)
anomalies = privatize_anomalies(raw_anomalies, role)
landmarks = _landmarks() if role != "public" else []

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_map, tab_fleet, tab_anomaly, tab_causal, tab_bench = st.tabs([
    "🗺️ Campus Map",
    "🛵 Fleet & Trajectories",
    "📊 Anomaly Analytics",
    "🔍 Causal Evidence",
    "⚡ ST-KG vs Star Schema",
])


# === TAB 1: Campus Map ======================================================
with tab_map:
    st.subheader("Campus Traffic & Safety Map")
    c1, c2, c3 = st.columns(3)
    c1.metric("Demand zones", len(demand))
    c2.metric("Anomalies shown", len(anomalies) if role != "public" else 0)
    c3.metric("Landmarks", len(landmarks))

    show_heat = st.checkbox("Demand heatmap", value=True)
    show_anom = st.checkbox("Anomaly markers", value=role != "public", disabled=role == "public")

    if role == "public":
        st.info("Public role: differential-privacy noise applied to demand aggregates. Anomaly points hidden.")

    with st.spinner("Rendering map…"):
        try:
            from phase4_dashboard.map_view import render_campus_map
            fmap = render_campus_map(
                demand, anomalies[:50], role,
                landmarks=landmarks,
                show_heatmap=show_heat,
                show_anomalies=show_anom,
            )
            html = _map_html(fmap)
            if html:
                st.components.v1.html(html, height=560, scrolling=False)
            else:
                st.warning("Install folium: `pip install folium`")
        except Exception as ex:
            st.warning(f"Map error: {ex}")

    if anomalies and check_access(role, "event_types"):
        import pandas as pd
        st.subheader("Recent flagged events")
        cols = [c for c in [
            "event_id", "vehicle_id", "event_type", "severity_score", "speed_kmh", "timestamp",
        ] if c in pd.DataFrame(anomalies).columns]
        st.dataframe(pd.DataFrame(anomalies)[cols].head(25), use_container_width=True, hide_index=True)


# === TAB 2: Fleet ===========================================================
with tab_fleet:
    st.subheader("Vehicle Trajectory Explorer")
    st.caption("Speed-colored path: green → orange → red. Red markers = anomalies.")

    if role == "public":
        st.warning("Trajectory view requires **analyst** or **admin** role.")
    else:
        vid = selected_vehicle if selected_vehicle != "All" else "V001"
        st.markdown(f"Showing trajectory for **{vid}**")
        pts = privatize_trajectory(_trajectory(vid, semester_opt), role)

        if not pts:
            st.info("No trajectory data. Ensure Neo4j is loaded (`python run_pipeline.py`).")
        else:
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("GPS points", len(pts))
            fc2.metric("Anomaly points", sum(1 for p in pts if p.get("is_anomaly")))
            speeds = [float(p["speed_kmh"]) for p in pts if p.get("speed_kmh") is not None]
            fc3.metric("Avg speed", f"{sum(speeds)/len(speeds):.1f} km/h" if speeds else "—")

            from phase4_dashboard.map_view import render_trajectory
            tmap = render_trajectory(pts, vehicle_id=vid)
            html_t = _map_html(tmap)
            if html_t:
                st.components.v1.html(html_t, height=520, scrolling=False)

            import pandas as pd
            traj_df = pd.DataFrame(pts)
            if "timestamp" in traj_df.columns and "speed_kmh" in traj_df.columns:
                import plotly.express as px
                traj_df["timestamp"] = pd.to_datetime(traj_df["timestamp"], errors="coerce", utc=True)
                fig_spd = px.line(
                    traj_df, x="timestamp", y="speed_kmh",
                    title=f"Speed profile — {vid}",
                    labels={"speed_kmh": "Speed (km/h)", "timestamp": "Time"},
                )
                fig_spd.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_spd, use_container_width=True)


# === TAB 3: Anomaly Analytics ===============================================
with tab_anomaly:
    st.subheader("Temporal & Semantic Anomaly Analytics")

    if not anomalies and role != "public":
        st.info("No anomalies for current filters. Try **All semesters** or another vehicle.")
    elif role == "public":
        st.warning("Detailed anomaly analytics are not available for the public role.")
    else:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        hourly_raw = _hourly(semester_opt, selected_vehicle)
        df_h = pd.DataFrame(hourly_raw)
        if not df_h.empty and "ts" in df_h.columns:
            df_h["ts"] = pd.to_datetime(df_h["ts"], errors="coerce", utc=True)
            df_h["hour"] = df_h["ts"].dt.floor("h")
            df_h["dow"] = df_h["ts"].dt.day_name()
            hourly = df_h.groupby(["hour", "event_type"]).size().reset_index(name="count")

            fig_bar = px.bar(
                hourly, x="hour", y="count", color="event_type",
                title="Anomalies per hour by event type",
                color_discrete_map={
                    "harsh_braking": "#e74c3c",
                    "rapid_acceleration": "#e67e22",
                    "speeding": "#c0392b",
                    "idling": "#3498db",
                },
            )
            fig_bar.update_layout(height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

            heat = df_h.groupby(["dow", df_h["ts"].dt.hour]).size().reset_index(name="count")
            heat.columns = ["day", "hour", "count"]
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            fig_heat = px.density_heatmap(
                heat, x="hour", y="day", z="count",
                title="Anomaly density (day × hour)",
                category_orders={"day": day_order},
                color_continuous_scale="Reds",
            )
            fig_heat.update_layout(height=320)
            st.plotly_chart(fig_heat, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            types = _event_types(semester_opt)
            if types:
                fig_pie = px.pie(
                    pd.DataFrame(types), names="event_type", values="count",
                    title="Anomaly mix by semantic type", hole=0.35,
                )
                fig_pie.update_layout(height=340)
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            sem_cmp = _semester_cmp()
            if sem_cmp:
                df_sem = pd.DataFrame(sem_cmp)
                fig_sem = go.Figure()
                fig_sem.add_trace(go.Bar(
                    name="Total events", x=[f"Sem {int(s['semester'])}" for s in sem_cmp],
                    y=df_sem["total_events"], marker_color="#3498db",
                ))
                fig_sem.add_trace(go.Bar(
                    name="Anomalies", x=[f"Sem {int(s['semester'])}" for s in sem_cmp],
                    y=df_sem["anomalies"], marker_color="#e74c3c",
                ))
                fig_sem.update_layout(
                    title="Concept drift: Semester 1 vs 2",
                    barmode="group", height=340,
                )
                st.plotly_chart(fig_sem, use_container_width=True)

        df_a = pd.DataFrame(anomalies)
        if "severity_score" in df_a.columns:
            fig_hist = px.histogram(
                df_a, x="severity_score", nbins=24,
                title="Severity distribution", color_discrete_sequence=["#e74c3c"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Flagged events registry")
        display_cols = [c for c in [
            "event_id", "vehicle_id", "driver_id", "event_type",
            "severity_score", "speed_kmh", "ax", "timestamp",
        ] if c in df_a.columns]
        st.dataframe(
            df_a[display_cols].sort_values("severity_score", ascending=False).head(100),
            use_container_width=True, hide_index=True,
        )


# === TAB 4: Causal Evidence =================================================
with tab_causal:
    st.subheader("Neuro-Symbolic Causal Evidence")
    st.markdown(
        "Select an anomaly to inspect the **2-hop ST-KG subgraph**, ontology context, "
        "and rule-based / neural explanation."
    )

    if not check_access(role, "causal_subgraph"):
        st.warning(f"Role **{role}** cannot access causal evidence. Switch to analyst or admin.")
    elif not anomalies:
        st.info("No anomalies available for investigation.")
    else:
        import pandas as pd
        import plotly.graph_objects as go

        df3 = pd.DataFrame(anomalies)
        labels = {
            row["event_id"]: f"{row.get('event_type','?')} | {row.get('vehicle_id','?')} | {str(row.get('timestamp',''))[:19]}"
            for _, row in df3.iterrows()
        }
        selected_eid = st.selectbox(
            "Investigate event",
            df3["event_id"].tolist(),
            format_func=lambda x: labels.get(x, x),
        )

        if selected_eid:
            subgraph = _subgraph(selected_eid)
            ev = subgraph.get("event", {})
            loc = subgraph.get("location", {})
            loc_labels = subgraph.get("location_labels", [])
            drv = subgraph.get("driver", {})
            veh = subgraph.get("vehicle", {})
            past = subgraph.get("past_anomalies", [])
            result = scorer.score(ev)

            ca, cb, cc = st.columns(3)
            with ca:
                st.metric("Confidence", f"{result.confidence:.1%}")
            with cb:
                st.metric("Severity", f"{ev.get('severity_score', 0):.2f}")
            with cc:
                st.metric("Prior anomalies (driver)", len(past))

            left, right = st.columns(2)
            with left:
                st.markdown("#### Event")
                st.markdown(f"**Type:** `{ev.get('event_type', '?')}`")
                st.markdown(f"**Speed:** {ev.get('speed_kmh', 0):.1f} km/h · **Ax:** {ev.get('ax', 0):.2f} m/s²")
                st.markdown(f"**Time:** {str(ev.get('timestamp', ''))[:19]}")
                st.info(result.explanation)
                st.caption(f"Scorer: {result.scorer_type}")

            with right:
                st.markdown("#### Ontology context")
                if loc_labels:
                    st.markdown("**Node labels:** " + ", ".join(f"`{l}`" for l in loc_labels))
                st.markdown(f"**Intersection:** `{loc.get('osm_id', 'N/A')}`")
                if loc.get("landmark_name"):
                    st.markdown(f"**Landmark:** {loc['landmark_name']}")
                if role in ("admin", "analyst"):
                    st.markdown(f"**Driver:** {drv.get('name', 'N/A')} (risk {drv.get('risk_profile', 0):.2f})")
                    st.markdown(f"**Vehicle:** {veh.get('vehicle_id', 'N/A')}")

            center_lat = float(ev.get("lat") or loc.get("lat") or 0.3382)
            center_lon = float(ev.get("lon") or loc.get("lon") or 32.5701)

            st.markdown("#### Causal subgraph map")
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
                    st.components.v1.html(html3, height=420, scrolling=False)
            except Exception as ex:
                st.caption(f"Map: {ex}")

            if result.subgraph_nodes:
                node_x = [n.get("lon", 0) for n in result.subgraph_nodes]
                node_y = [n.get("lat", 0) for n in result.subgraph_nodes]
                fig_net = go.Figure()
                for edge in result.subgraph_edges:
                    u_id, v_id = str(edge.get("u", "")), str(edge.get("v", ""))
                    u_n = next((n for n in result.subgraph_nodes if str(n.get("osm_id")) == u_id), None)
                    v_n = next((n for n in result.subgraph_nodes if str(n.get("osm_id")) == v_id), None)
                    if u_n and v_n:
                        fig_net.add_trace(go.Scatter(
                            x=[u_n.get("lon"), v_n.get("lon"), None],
                            y=[u_n.get("lat"), v_n.get("lat"), None],
                            mode="lines", line=dict(color="#95a5a6", width=1), showlegend=False,
                        ))
                fig_net.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    text=[n.get("label", "") for n in result.subgraph_nodes],
                    textposition="top center",
                    marker=dict(size=11, color="#e74c3c"),
                ))
                fig_net.update_layout(
                    title="2-hop ego-graph (campus topology)",
                    height=360, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_net, use_container_width=True)

            if past and role in ("admin", "analyst"):
                st.subheader("Driver prior anomaly history")
                st.dataframe(pd.DataFrame(past).head(10), use_container_width=True, hide_index=True)


# === TAB 5: Benchmarks ======================================================
with tab_bench:
    st.subheader("ST-Knowledge Graph vs Kimball Star Schema")
    st.markdown(
        "Five analytical queries run **10× each**; median latency (ms) reported. "
        "Q4 (causal chain) is **not expressible** in star schema without expensive spatial joins."
    )

    bench = load_benchmark_results()
    col_run, col_info = st.columns([1, 3])
    with col_run:
        if st.button("Run benchmarks", type="primary", use_container_width=True):
            with st.spinner("Running 5 queries × 10 iterations on Neo4j + pandas…"):
                try:
                    bench = run_and_cache_benchmarks()
                    st.success("Complete — results cached.")
                except Exception as ex:
                    st.error(str(ex))

    with col_info:
        if bench:
            st.caption(f"Cached at `{bench.get('_cached_at', 'previous run')}`")
        else:
            st.caption("Click **Run benchmarks** (requires Neo4j + ~30s).")

    if bench:
        import pandas as pd
        import plotly.express as px

        rows = []
        for name, data in bench.items():
            if name.startswith("_"):
                continue
            rows.append({
                "Query": name.split(":")[0],
                "Description": name.split(":", 1)[-1].strip() if ":" in name else name,
                "ST-KG (ms)": data.get("stkg_median_ms", 0),
                "Star Schema (ms)": data.get("star_median_ms", 0),
                "ST-KG OK": data.get("stkg_expressible", True),
                "Star OK": data.get("star_expressible", True),
            })
        df_b = pd.DataFrame(rows)
        if not df_b.empty:
            fig_b = px.bar(
                df_b.melt(id_vars=["Query", "Description"], value_vars=["ST-KG (ms)", "Star Schema (ms)"],
                          var_name="Architecture", value_name="Median ms"),
                x="Query", y="Median ms", color="Architecture", barmode="group",
                title="Query latency: ST-KG vs Star Schema (median of 10 runs)",
                color_discrete_map={"ST-KG (ms)": "#2ecc71", "Star Schema (ms)": "#95a5a6"},
            )
            fig_b.update_layout(height=420)
            st.plotly_chart(fig_b, use_container_width=True)

            st.dataframe(
                df_b.style.format({"ST-KG (ms)": "{:.2f}", "Star Schema (ms)": "{:.2f}"}),
                use_container_width=True,
            )

            q4 = next((r for r in rows if r["Query"] == "Q4"), None)
            if q4:
                st.success(
                    "**Q4 Causal chain:** Native graph traversal in Neo4j vs approximate spatial "
                    "self-joins in pandas — demonstrates ST-KG expressiveness for campus safety analytics."
                )
    else:
        st.info("No benchmark results yet. Start Neo4j and click **Run benchmarks**.")

st.caption("MakFleet · BIS 3205 · Semantic ST-KG · Privacy-by-Design · Makerere University")
