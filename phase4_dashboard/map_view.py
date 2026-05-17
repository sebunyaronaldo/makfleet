"""Folium map rendering helpers for the MakFleet dashboard."""

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

from config.settings import MAKERERE_BBOX

_CENTER_LAT = (MAKERERE_BBOX[0] + MAKERERE_BBOX[2]) / 2
_CENTER_LON = (MAKERERE_BBOX[1] + MAKERERE_BBOX[3]) / 2

_EVENT_COLORS = {
    "harsh_braking": "red",
    "rapid_acceleration": "orange",
    "speeding": "darkred",
    "idling": "blue",
    "safe_stop": "green",
    "normal_travel": "gray",
}

_LANDMARK_ICONS = {
    "ZebraCrossing": ("crosswalk", "green"),
    "BodaBodaStop": ("motorcycle", "blue"),
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
        tiles="CartoDB positron",
        control_scale=True,
    )


def _add_landmarks(m: "folium.Map", landmarks: list[dict]) -> None:
    for lm in landmarks:
        lat, lon = lm.get("lat"), lm.get("lon")
        if not lat or not lon:
            continue
        ltype = lm.get("landmark_type", "")
        _, color = _LANDMARK_ICONS.get(ltype, ("info-sign", "purple"))
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=10,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{lm.get('name', 'Landmark')}</b><br>{ltype}",
                max_width=200,
            ),
            tooltip=lm.get("name") or ltype,
        ).add_to(m)


def render_campus_map(
    demand_data: list[dict],
    anomalies: list[dict],
    role: str = "analyst",
    landmarks: list[dict] | None = None,
    show_heatmap: bool = True,
    show_anomalies: bool = True,
) -> "folium.Map | None":
    if not _FOLIUM_OK:
        return None

    m = base_map()
    if landmarks:
        _add_landmarks(m, landmarks)

    if show_heatmap:
        heat_points = [
            [float(d["lat"]), float(d["lon"]), float(d.get("trip_count") or 1)]
            for d in demand_data
            if d.get("lat") and d.get("lon")
        ]
        if heat_points:
            HeatMap(heat_points, radius=22, blur=18, max_zoom=18).add_to(m)

    if show_anomalies and role in ("admin", "analyst") and anomalies:
        cluster = MarkerCluster(name="Anomalies").add_to(m)
        for ev in anomalies[:80]:
            lat, lon = ev.get("lat"), ev.get("lon")
            if not lat or not lon:
                continue
            etype = ev.get("event_type", "unknown")
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=7,
                color=_EVENT_COLORS.get(etype, "gray"),
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(
                    f"<b>{etype}</b><br>"
                    f"Vehicle: {ev.get('vehicle_id', '?')}<br>"
                    f"Speed: {ev.get('speed_kmh', 0):.1f} km/h<br>"
                    f"Severity: {ev.get('severity_score', 0):.2f}",
                    max_width=220,
                ),
            ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


def render_trajectory(
    trajectory_points: list[dict],
    vehicle_id: str = "",
) -> "folium.Map | None":
    if not _FOLIUM_OK or not trajectory_points:
        return None

    lats = [float(p["lat"]) for p in trajectory_points if p.get("lat")]
    lons = [float(p["lon"]) for p in trajectory_points if p.get("lon")]
    if not lats:
        return None

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles="CartoDB positron")

    path_coords = []
    for pt in trajectory_points:
        lat, lon = pt.get("lat"), pt.get("lon")
        if not lat or not lon:
            continue
        coord = [float(lat), float(lon)]
        path_coords.append(coord)
        speed = float(pt.get("speed_kmh") or 0)
        is_anom = pt.get("is_anomaly")
        color = "red" if is_anom else _speed_color(speed)
        radius = 8 if is_anom else 4
        folium.CircleMarker(
            location=coord,
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.75,
            popup=f"{speed:.1f} km/h — {pt.get('event_type', '')}",
        ).add_to(m)

    if len(path_coords) > 1:
        folium.PolyLine(path_coords, color="#3498db", weight=3, opacity=0.7).add_to(m)

    if vehicle_id:
        folium.Marker(path_coords[0], popup=f"Start · {vehicle_id}").add_to(m)
        folium.Marker(path_coords[-1], popup=f"End · {vehicle_id}").add_to(m)

    return m


def render_causal_subgraph_map(
    center_lat: float,
    center_lon: float,
    subgraph_nodes: list[dict],
    subgraph_edges: list[dict],
    event_type: str = "harsh_braking",
) -> "folium.Map | None":
    if not _FOLIUM_OK:
        return None

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="CartoDB positron")
    node_coords: dict[str, tuple] = {}

    for node in subgraph_nodes:
        lat = node.get("lat") or center_lat
        lon = node.get("lon") or center_lon
        osm_id = str(node.get("osm_id", ""))
        node_coords[osm_id] = (lat, lon)
        label = node.get("label", "Intersection")
        is_center = abs(lat - center_lat) < 0.0002 and abs(lon - center_lon) < 0.0002
        color = "red" if is_center else ("green" if "Stop" in label or "Crossing" in label else "blue")
        folium.CircleMarker(
            location=[lat, lon],
            radius=9 if is_center else 5,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=f"{label}<br>{osm_id}",
        ).add_to(m)

    for edge in subgraph_edges:
        u_id, v_id = str(edge.get("u", "")), str(edge.get("v", ""))
        if u_id in node_coords and v_id in node_coords:
            folium.PolyLine(
                [node_coords[u_id], node_coords[v_id]],
                color="navy",
                weight=2,
                opacity=0.65,
            ).add_to(m)

    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.Icon(color="red", icon="warning-sign"),
        popup=f"Anomaly: {event_type}",
    ).add_to(m)
    return m
