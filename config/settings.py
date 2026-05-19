from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_CACHE_DIR = DATA_DIR / "graph_cache"
AUDIT_LOG = DATA_DIR / "audit_log.jsonl"
CHECKPOINT_PATH = DATA_DIR / "stgnn_checkpoint.pt"

# Makerere University campus bounding box (south, west, north, east)
MAKERERE_BBOX = (0.3340, 32.5680, 0.3420, 32.5760)

# Neo4j — reads from environment variables, falls back to local Docker defaults
import os
NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = (os.environ.get("NEO4J_USER") or
                  os.environ.get("NEO4J_USERNAME") or
                  "1a273157")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "makfleet2024")
NEO4J_AUTH     = (NEO4J_USER, NEO4J_PASSWORD)

# Simulation
NUM_VEHICLES = 15
NUM_TRIPS_PER_VEHICLE_PER_DAY = 8
SIM_DAYS_SEM1 = 90   # Jan 15 – Apr 15 2024
SIM_DAYS_SEM2 = 60   # Aug 01 – Oct 01 2024
SEM1_START = "2024-01-15"
SEM2_START = "2024-08-01"

# GPS noise model
GPS_NOISE_STD = 0.00005   # ~5 metres in degrees
GPS_SAMPLE_INTERVAL_S = 5  # seconds between telemetry ticks

# Accelerometer thresholds (m/s²)
ACCEL_HARSH_BRAKE_THRESHOLD = -3.5
ACCEL_RAPID_ACCEL_THRESHOLD = 2.8
IDLE_SPEED_THRESHOLD_KMH = 2.0

# Event injection rates for reckless drivers (risk_profile > 0.7)
RECKLESS_BRAKE_RATE = 0.05   # 5% of points
RECKLESS_ACCEL_RATE = 0.03

# Privacy
PRIVACY_K = 5          # k-anonymity
PRIVACY_EPSILON = 1.0  # differential privacy budget

# ST-GNN training
WINDOW_SIZE_MIN = 15   # minutes per graph snapshot
T_SNAPSHOTS = 4        # number of snapshots per sample
SPATIAL_OOD_LON = 32.574  # eastern campus boundary for spatial hold-out split
STGNN_HIDDEN = 64
STGNN_LAYERS = 2
STGNN_LR = 1e-3
STGNN_EPOCHS = 10
STGNN_POS_WEIGHT = 19.0   # class imbalance: ~5% anomalies

# Benchmark
BENCHMARK_RUNS = 10

# Makerere landmark coordinates (lat, lon) for synthetic spatial nodes
LANDMARKS = {
    "Main Gate":             (0.3344, 32.5681, "BodaBodaStop"),
    "Senate Building":       (0.3382, 32.5701, "ZebraCrossing"),
    "Faculty of Computing":  (0.3374, 32.5712, "BodaBodaStop"),
    "Main Library":          (0.3370, 32.5698, "ZebraCrossing"),
    "Sports Ground":         (0.3358, 32.5730, "BodaBodaStop"),
    "Lumumba Hall Gate":     (0.3349, 32.5720, "BodaBodaStop"),
    "Medical School":        (0.3393, 32.5695, "ZebraCrossing"),
}

UGANDAN_NAMES = [
    "Okello Joseph", "Nakamya Grace", "Mugisha David", "Namukasa Fatuma",
    "Ssebunya Ronald", "Namutebi Sarah", "Kiggundu Isaac", "Nalwoga Aisha",
    "Tumwine Patrick", "Nanteza Prossy", "Byaruhanga Dennis", "Nakato Juliet",
    "Ssemwogerere Alex", "Nabbosa Christine", "Muwonge Samuel",
]
