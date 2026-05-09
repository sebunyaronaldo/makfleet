"""
Synthetic bodaboda telemetry generator.

Produces physically realistic GPS + accelerometer traces for NUM_VEHICLES
vehicles making random trips along the real Makerere campus road network.
Reckless drivers (risk_profile > 0.7) have injected harsh-braking and
rapid-acceleration events at calibrated rates.

Outputs
-------
data/raw/sem1_telemetry.jsonl   (Semester 1 — training split)
data/raw/sem2_telemetry.jsonl   (Semester 2 — temporal OOD test split)
"""

import json
import math
import random
import uuid
from datetime import datetime, timedelta

import networkx as nx
import numpy as np

from config.settings import (
    NUM_VEHICLES, NUM_TRIPS_PER_VEHICLE_PER_DAY,
    SIM_DAYS_SEM1, SIM_DAYS_SEM2, SEM1_START, SEM2_START,
    GPS_NOISE_STD, GPS_SAMPLE_INTERVAL_S,
    ACCEL_HARSH_BRAKE_THRESHOLD, ACCEL_RAPID_ACCEL_THRESHOLD,
    IDLE_SPEED_THRESHOLD_KMH,
    RECKLESS_BRAKE_RATE, RECKLESS_ACCEL_RATE,
    RAW_DIR, UGANDAN_NAMES,
)
from phase1_ingestion.campus_graph import load_campus_graph, node_coords
from phase1_ingestion.provenance import stamp_batch

rng = np.random.default_rng(42)
random.seed(42)


# ---------------------------------------------------------------------------
# Fleet generation
# ---------------------------------------------------------------------------

def _make_fleet() -> tuple[list[dict], list[dict]]:
    vehicles, drivers = [], []
    for i in range(NUM_VEHICLES):
        vid = f"V{i+1:03d}"
        did = f"D{i+1:03d}"
        risk = round(rng.uniform(0.0, 1.0), 3)
        plate = f"UA{random.randint(100,999)}{random.choice('ABCDEFGHJKLMNPRSTUVWXY')}"
        name = UGANDAN_NAMES[i % len(UGANDAN_NAMES)]
        vehicles.append({"vehicle_id": vid, "plate": plate})
        drivers.append({
            "driver_id": did, "vehicle_id": vid,
            "name": name, "risk_profile": risk,
            "is_reckless": risk > 0.7,
        })
    return vehicles, drivers


# ---------------------------------------------------------------------------
# Trip simulation
# ---------------------------------------------------------------------------

def _interpolate_edge(G: nx.MultiDiGraph, u: int, v: int) -> list[tuple[float, float]]:
    """Return evenly spaced (lat, lon) points along an OSM edge."""
    edge_data = G.edges[u, v, 0]
    if "geometry" in edge_data:
        coords = list(edge_data["geometry"].coords)  # (lon, lat) from shapely
        return [(lat, lon) for lon, lat in coords]
    lat_u, lon_u = node_coords(G, u)
    lat_v, lon_v = node_coords(G, v)
    return [(lat_u, lon_u), (lat_v, lon_v)]


def _edge_speed_kmh(G: nx.MultiDiGraph, u: int, v: int) -> float:
    data = G.edges[u, v, 0]
    spd = data.get("maxspeed", "30")
    if isinstance(spd, list):
        spd = spd[0]
    try:
        return float(str(spd).replace(" mph", "").replace(" km/h", ""))
    except ValueError:
        return 30.0


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _heading(lat1, lon1, lat2, lon2) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return math.degrees(math.atan2(dlon, dlat)) % 360


def _accel_reading(is_reckless: bool, inject_brake: bool, inject_accel: bool):
    if inject_brake:
        ax = float(rng.normal(-4.2, 0.5))
    elif inject_accel:
        ax = float(rng.normal(3.1, 0.4))
    else:
        ax = float(rng.normal(0.0, 0.3))
    ay = float(rng.normal(0.0, 0.2))
    az = float(rng.normal(9.81, 0.1))
    return ax, ay, az


def _simulate_trip(
    G: nx.MultiDiGraph,
    vehicle_id: str,
    driver: dict,
    trip_start: datetime,
    semester: int,
) -> list[dict]:
    nodes = list(G.nodes)
    if len(nodes) < 2:
        return []

    origin = random.choice(nodes)
    dest = random.choice(nodes)
    while dest == origin:
        dest = random.choice(nodes)

    try:
        path = nx.shortest_path(G, origin, dest, weight="length")
    except nx.NetworkXNoPath:
        path = [origin, dest] if G.has_edge(origin, dest) else [origin]

    if len(path) < 2:
        return []

    records = []
    current_time = trip_start
    is_reckless = driver["is_reckless"]
    idle_counter = 0

    for idx in range(len(path) - 1):
        u, v = path[idx], path[idx + 1]
        if not G.has_edge(u, v):
            continue

        points = _interpolate_edge(G, u, v)
        speed_kmh = _edge_speed_kmh(G, u, v)

        for pi, (lat, lon) in enumerate(points):
            # GPS noise
            lat_n = lat + float(rng.normal(0, GPS_NOISE_STD))
            lon_n = lon + float(rng.normal(0, GPS_NOISE_STD))

            # Speed variation
            spd = max(0.0, float(rng.normal(speed_kmh, 4.0)))

            # Engine state
            if spd < IDLE_SPEED_THRESHOLD_KMH:
                idle_counter += 1
                engine_state = "IDLE"
            else:
                idle_counter = 0
                engine_state = "ON"

            # Accelerometer
            inject_brake = is_reckless and rng.random() < RECKLESS_BRAKE_RATE
            inject_accel = is_reckless and (not inject_brake) and rng.random() < RECKLESS_ACCEL_RATE
            ax, ay, az = _accel_reading(is_reckless, inject_brake, inject_accel)

            # Heading
            if pi + 1 < len(points):
                nlat, nlon = points[pi + 1]
            else:
                nlat, nlon = lat, lon
            heading = _heading(lat, lon, nlat, nlon)

            records.append({
                "event_id": str(uuid.uuid4()),
                "vehicle_id": vehicle_id,
                "driver_id": driver["driver_id"],
                "timestamp": current_time.isoformat() + "Z",
                "lat": round(lat_n, 7),
                "lon": round(lon_n, 7),
                "speed_kmh": round(spd, 2),
                "heading_deg": round(heading, 1),
                "ax": round(ax, 4),
                "ay": round(ay, 4),
                "az": round(az, 4),
                "engine_state": engine_state,
                "edge_u": int(u),
                "edge_v": int(v),
                "semester": semester,
            })
            current_time += timedelta(seconds=GPS_SAMPLE_INTERVAL_S)

    return records


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def simulate(G: nx.MultiDiGraph) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns (vehicles, drivers, sem1_records, sem2_records).
    All records have provenance_hash stamped.
    """
    vehicles, drivers = _make_fleet()
    driver_map = {d["vehicle_id"]: d for d in drivers}

    sem1_records: list[dict] = []
    sem2_records: list[dict] = []

    sem1_base = datetime.fromisoformat(SEM1_START)
    sem2_base = datetime.fromisoformat(SEM2_START)

    # Peak-hour weights: more trips 7–9am and 4–7pm
    _hour_weights = [
        1,1,1,1,1,1,1, 4,4, 2,2,2,2,2, 2,2, 3,3,3, 2,1,1,1,1
    ]

    for vehicle in vehicles:
        vid = vehicle["vehicle_id"]
        driver = driver_map[vid]

        for sem, base_date, out_list, n_days in [
            (1, sem1_base, sem1_records, SIM_DAYS_SEM1),
            (2, sem2_base, sem2_records, SIM_DAYS_SEM2),
        ]:
            for day_offset in range(n_days):
                day = base_date + timedelta(days=day_offset)
                for _ in range(NUM_TRIPS_PER_VEHICLE_PER_DAY):
                    hour = random.choices(range(24), weights=_hour_weights)[0]
                    minute = random.randint(0, 59)
                    trip_start = day.replace(hour=hour, minute=minute, second=0)
                    trip_recs = _simulate_trip(G, vid, driver, trip_start, sem)
                    out_list.extend(trip_recs)

    sem1_records, sem1_root = stamp_batch(sem1_records)
    sem2_records, sem2_root = stamp_batch(sem2_records)

    print(f"[simulator] Sem1: {len(sem1_records):,} records | Merkle root: {sem1_root[:16]}…")
    print(f"[simulator] Sem2: {len(sem2_records):,} records | Merkle root: {sem2_root[:16]}…")

    return vehicles, drivers, sem1_records, sem2_records


def write_jsonl(records: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[simulator] Written {len(records):,} records -> {path}")


if __name__ == "__main__":
    G = load_campus_graph()
    vehicles, drivers, sem1, sem2 = simulate(G)
    write_jsonl(sem1, RAW_DIR / "sem1_telemetry.jsonl")
    write_jsonl(sem2, RAW_DIR / "sem2_telemetry.jsonl")

    import json as _json
    meta = {"vehicles": vehicles, "drivers": drivers}
    with open(RAW_DIR / "fleet_meta.json", "w") as f:
        _json.dump(meta, f, indent=2)
    print("[simulator] Fleet metadata saved.")
