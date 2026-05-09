"""
MakFleet full pipeline runner — executes all phases in order.

Run this ONCE after Docker Neo4j is up:
    python run_pipeline.py

Then start the dashboard:
    streamlit run phase4_dashboard/app.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAW_DIR, PROCESSED_DIR, NEO4J_URI, NEO4J_AUTH

print("=" * 60)
print("  MakFleet Pipeline Runner")
print("=" * 60)


# ---------------------------------------------------------------------------
# Phase 1 — Campus graph + simulation
# ---------------------------------------------------------------------------

print("\n[Phase 1] Loading Makerere campus road network…")
from phase1_ingestion.campus_graph import load_campus_graph
G = load_campus_graph()
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\n[Phase 1] Simulating bodaboda telemetry…")
from phase1_ingestion.simulator import simulate, write_jsonl
vehicles, drivers, sem1_records, sem2_records = simulate(G)
write_jsonl(sem1_records, RAW_DIR / "sem1_telemetry.jsonl")
write_jsonl(sem2_records, RAW_DIR / "sem2_telemetry.jsonl")

meta = {"vehicles": vehicles, "drivers": drivers}
with open(RAW_DIR / "fleet_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Fleet meta saved: {len(vehicles)} vehicles, {len(drivers)} drivers")


# ---------------------------------------------------------------------------
# Phase 2 — Map-matching + semantic enrichment + privacy
# ---------------------------------------------------------------------------

print("\n[Phase 2] Map-matching Semester 1 records…")
from phase2_pipeline.map_matcher import match_all_vehicles
sem1_matched = match_all_vehicles(sem1_records, G)

print("\n[Phase 2] Map-matching Semester 2 records…")
sem2_matched = match_all_vehicles(sem2_records, G)

print("\n[Phase 2] Semantic enrichment…")
from phase2_pipeline.semantic_enricher import enrich_all
sem1_enriched = enrich_all(sem1_matched, G)
sem2_enriched = enrich_all(sem2_matched, G)

all_enriched = sem1_enriched + sem2_enriched
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
with open(PROCESSED_DIR / "enriched_events.jsonl", "w") as f:
    for r in all_enriched:
        f.write(json.dumps(r, default=str) + "\n")
print(f"  Enriched events written: {len(all_enriched):,}")


# ---------------------------------------------------------------------------
# Phase 1 — Neo4j loading (requires enriched data)
# ---------------------------------------------------------------------------

print("\n[Phase 1] Loading data into Neo4j…")
try:
    from phase1_ingestion.neo4j_loader import (
        get_driver, create_constraints, load_spatial_graph,
        load_fleet_entities, load_telemetry_events, load_provenance_anchor,
        clear_database,
    )
    from phase1_ingestion.provenance import build_chain

    db_driver = get_driver()
    print("  Connected to Neo4j")

    print("  Clearing existing data…")
    clear_database(db_driver)

    print("  Creating constraints…")
    create_constraints(db_driver)

    print("  Loading spatial graph…")
    load_spatial_graph(G, db_driver)

    print("  Loading fleet entities…")
    load_fleet_entities(vehicles, drivers, db_driver)

    print("  Loading Semester 1 events…")
    load_telemetry_events(sem1_enriched, db_driver)
    sem1_root = build_chain(sem1_enriched)
    load_provenance_anchor(sem1_root, 1, len(sem1_enriched), db_driver)

    print("  Loading Semester 2 events…")
    load_telemetry_events(sem2_enriched, db_driver)
    sem2_root = build_chain(sem2_enriched)
    load_provenance_anchor(sem2_root, 2, len(sem2_enriched), db_driver)

    db_driver.close()
    print("  Neo4j loading complete!")

except Exception as e:
    print(f"  WARNING: Neo4j loading failed: {e}")
    print("  Make sure Docker is running: docker-compose up -d")
    print("  Continuing — dashboard will work with reduced functionality.")


# ---------------------------------------------------------------------------
# Phase 3 — ST-GNN training (optional, dashboard works without it)
# ---------------------------------------------------------------------------

print("\n[Phase 3] Building PyG dataset and training ST-GNN…")
print("  (This may take 10–20 minutes on CPU. Skip with Ctrl+C — dashboard still works.)")
try:
    from phase3_model.graph_dataset import build_dataset, split_dataset
    from phase3_model.trainer import run_training

    sem1_ds = build_dataset(semester=1)
    sem2_ds = build_dataset(semester=2)

    if sem1_ds or sem2_ds:
        all_ds = sem1_ds + sem2_ds
        train_s, val_s, test_t, test_ood = split_dataset(all_ds)
        results = run_training(train_s, val_s, test_t, test_ood)
        print(f"  Training complete. Temporal OOD AUC: {results['temporal_ood']['auc']}")
    else:
        print("  No dataset available — skipping training")

except KeyboardInterrupt:
    print("  Training skipped (Ctrl+C). Dashboard will use rule-based scorer.")
except Exception as e:
    print(f"  Training error: {e}. Dashboard will use rule-based scorer.")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  Pipeline complete!")
print("  Start the dashboard with:")
print("    streamlit run phase4_dashboard/app.py")
print("=" * 60)
