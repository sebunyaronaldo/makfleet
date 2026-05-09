"""
MakFleet Spatio-Temporal Knowledge Graph Ontology.

Three domains with strict ontological commitments:
  1. Spatial Entities   — campus topology
  2. Fleet Entities     — vehicles and drivers
  3. Temporal Event Entities — enriched telematics events

ContextualStop is a DISTINCT node label from TelematicsEvent.
MATCH (n:TelematicsEvent) will NEVER return a ContextualStop node.
To query both: MATCH (n) WHERE n:TelematicsEvent OR n:ContextualStop
"""

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------

class SpatialEntity:
    Intersection    = "Intersection"
    RoadSegment     = "RoadSegment"
    PedestrianZone  = "PedestrianZone"
    BodaBodaStop    = "BodaBodaStop"
    ZebraCrossing   = "ZebraCrossing"

class FleetEntity:
    Vehicle = "Vehicle"
    Driver  = "Driver"

class TemporalEventEntity:
    TelematicsEvent = "TelematicsEvent"   # all anomalous/normal events except safe stops
    ContextualStop  = "ContextualStop"    # safe_stop: ontologically distinct label
    TrajectorySegment = "TrajectorySegment"
    DemandWindow    = "DemandWindow"

# Convenience sets for Cypher WHERE clauses
ALL_EVENT_LABELS = {
    TemporalEventEntity.TelematicsEvent,
    TemporalEventEntity.ContextualStop,
}

ANOMALY_EVENT_TYPES = {
    "harsh_braking",
    "rapid_acceleration",
    "speeding",
}

NON_ANOMALY_EVENT_TYPES = {
    "safe_stop",       # → loaded as ContextualStop
    "idling",
    "normal_travel",
}

# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

class Rel:
    TRAVERSES   = "TRAVERSES"   # Vehicle → RoadSegment (with timestamp, speed)
    CAUSED_BY   = "CAUSED_BY"   # TelematicsEvent → Driver
    LOCATED_AT  = "LOCATED_AT"  # TelematicsEvent|ContextualStop → Intersection
    ADJACENT_TO = "ADJACENT_TO" # Intersection ↔ Intersection (bidirectional)
    DRIVES      = "DRIVES"      # Driver → Vehicle
    PRECEDES    = "PRECEDES"    # TelematicsEvent → TelematicsEvent (trajectory order)

# ---------------------------------------------------------------------------
# Severity scores per event type
# ---------------------------------------------------------------------------

SEVERITY = {
    "harsh_braking":      0.90,
    "speeding":           0.75,
    "rapid_acceleration": 0.65,
    "idling":             0.10,
    "safe_stop":          0.00,
    "normal_travel":      0.00,
}

IS_ANOMALY = {
    "harsh_braking":      True,
    "speeding":           True,
    "rapid_acceleration": True,
    "idling":             False,
    "safe_stop":          False,
    "normal_travel":      False,
}

# ---------------------------------------------------------------------------
# Road type encoding for ST-GNN edge features
# ---------------------------------------------------------------------------

ROAD_TYPE_ENCODING = {
    "residential":   0,
    "tertiary":      1,
    "secondary":     2,
    "primary":       3,
    "pedestrian":    4,
    "path":          5,
    "footway":       6,
    "unclassified":  7,
}
DEFAULT_ROAD_TYPE = 7
DEFAULT_SPEED_KMH = 30
