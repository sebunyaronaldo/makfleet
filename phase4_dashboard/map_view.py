"""Folium map rendering helpers for the MakFleet dashboard."""

import math

try:
    import folium
    from folium.plugins import HeatMap
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

from config.settings import MAKERERE_BBOX

# Campus centre
_CENTER_LAT = (MAKERERE_BBOX[0] + MAKERERE_BBOX[2]) / 2
_CENTER_LON = (MAKERERE_BBOX[1] + MAKERERE_BBOX[3]) / 2

_EVENT_COLORS = {
    "harsh_braking":      "red",
    "rapid_acceleration": "orange",
    "speeding":           "darkred",
    "idling":             "blue",
    "safe_stop":          "green",
    "normal_travel":      "gray",
}


def _speed_color(speed_kmh: float) -> str:
    if speed_kmh < 10:
        return "green"
    if speed_kmh < 25:
        return "orange"
    return "red"


def base_map(zoom: int = 16) -> "folium.Map | None":
    if not _FOLIUM_OK:
        return None
    return folium.Map(
        location=[_CENTER_LAT, _CENTER_LON],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )


def render_campus_map(
    demand_data: list[dict],
    anomalies: list[dict],
    role: str = "analyst",
) -> "folium.Map | None":
    """
    Tab 1 map: heatmap of trip demand + anomaly markers.
    k-anonymized GPS is used for non-admin roles (demand_data is already filtered).
    """
    if not _FOLIUM_OK:
        return None

    m = base_map()

    # Demand heatmap
    heat_points = [
        [float(d.get("lat") or _CENTER_LAT),
         float(d.get("lon") or _CENTER_LON),
         float(d.get("trip_count") or 1)]
        for d in demand_data
        if d.get("lat") and d.get("lon")
    ]
    if heat_points:
        HeatMap(heat_points, radius=20, blur=15, max_zoom=18).add_to(m)

    # Anomaly markers (shown to analyst and admin)
    if role in ("admin", "analyst"):
        for ev in anomalies[:100]:
            lat = ev.get("lat")
            lon = ev.get("lon")
            if not lat or not lon:
                continue
            etype = ev.get("event_type", "unknown")
            color = _EVENT_COLORS.get(etype, "gray")
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{etype}</b><br>"
                    f"Vehicle: {ev.get('vehicle_id', '?')}<br>"
                    f"Speed: {ev.get('speed_kmh', 0):.1f} km/h<br>"
                    f"Severity: {ev.get('severity_score', 0):.2f}<br>"
                    f"Time: {ev.get('timestamp', '')[:19]}",
                    max_width=220,
                ),
            ).add_to(m)

    return m


def render_trajectory(
    trajectory_points: list[dict],
    role: str = "analyst",
) -> "folium.Map | None":
    """Color-coded vehicle trajectory (green→orange→red by speed)."""
    if not _FOLIUM_OK or not trajectory_points:
        return None

    m = base_map()
    for pt in trajectory_points:
        lat = pt.get("lat")
        lon = pt.get("lon")
        speed = float(pt.get("speed_kmh") or 0)
        if not lat or not lon:
            continue
        color = _speed_color(speed)
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{speed:.1f} km/h — {pt.get('event_type', '')}",
        ).add_to(m)

    return m


def render_causal_subgraph_map(
    center_lat: float,
    center_lon: float,
    subgraph_nodes: list[dict],
    subgraph_edges: list[dict],
    event_type: str = "harsh_braking",
) -> "folium.Map | None":
    """
    Highlight the 2-hop causal subgraph around a flagged anomaly.
    Center node is rendered in red; neighbors in blue; edges as lines.
    """
    if not _FOLIUM_OK:
        return None

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="OpenStreetMap")

    node_coords: dict[str, tuple] = {}
    for node in subgraph_nodes:
        lat = node.get("lat") or center_lat
        lon = node.get("lon") or center_lon
        osm_id = str(node.get("osm_id", ""))
        node_coords[osm_id] = (lat, lon)
        label = node.get("label", "Intersection")
        is_center = (abs(lat - center_lat) < 0.0002 and abs(lon - center_lon) < 0.0002)
        color = "red" if is_center else ("green" if "Stop" in label or "Crossing" in label else "blue")
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 if is_center else 5,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=f"{label}<br>{osm_id}",
        ).add_to(m)

    # Draw edges
    for edge in subgraph_edges:
        u_id = str(edge.get("u", ""))
        v_id = str(edge.get("v", ""))
        if u_id in node_coords and v_id in node_coords:
            folium.PolyLine(
                [node_coords[u_id], node_coords[v_id]],
                color="navy",
                weight=2,
                opacity=0.6,
            ).add_to(m)

    # Star marker at anomaly location
    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
        popup=f"Anomaly: {event_type}",
    ).add_to(m)

    return m
