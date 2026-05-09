"""
Reload Neo4j from already-enriched data on disk.
Run this after restarting Docker without re-simulating.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAW_DIR, PROCESSED_DIR
from phase1_ingestion.campus_graph import load_campus_graph
from phase1_ingestion.neo4j_loader import (
    get_driver, create_constraints, load_spatial_graph,
    load_fleet_entities, load_telemetry_events,
    load_provenance_anchor, clear_database,
)
from phase1_ingestion.provenance import build_chain

print("Loading campus graph from cache...")
G = load_campus_graph()
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("Loading fleet metadata...")
with open(RAW_DIR / "fleet_meta.json") as f:
    meta = json.load(f)
vehicles = meta["vehicles"]
drivers = meta["drivers"]

print("Loading enriched events from disk...")
enriched = []
with open(PROCESSED_DIR / "enriched_events.jsonl") as f:
    for line in f:
        enriched.append(json.loads(line))
sem1 = [r for r in enriched if r.get("semester") == 1]
sem2 = [r for r in enriched if r.get("semester") == 2]
print(f"  Sem1: {len(sem1):,} | Sem2: {len(sem2):,}")

print("Connecting to Neo4j...")
db = get_driver()
print("  Connected")

print("Clearing database...")
clear_database(db)

print("Creating constraints...")
create_constraints(db)

print("Loading spatial graph...")
load_spatial_graph(G, db)

print("Loading fleet entities...")
load_fleet_entities(vehicles, drivers, db)

print("Loading Semester 1 events...")
load_telemetry_events(sem1, db)
load_provenance_anchor(build_chain(sem1), 1, len(sem1), db)

print("Loading Semester 2 events...")
load_telemetry_events(sem2, db)
load_provenance_anchor(build_chain(sem2), 2, len(sem2), db)

db.close()
print("\nNeo4j reload complete!")
print("Launch dashboard: streamlit run phase4_dashboard/app.py")
