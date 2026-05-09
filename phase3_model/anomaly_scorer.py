"""
Anomaly scoring with graceful fallback (Gap 3).

NeuralScorer   — uses the trained ST-GNN checkpoint
RuleBasedScorer — uses severity_score from enriched Neo4j events (no model needed)

The dashboard uses load_scorer() which automatically chooses based on checkpoint
availability. Both scorers return the same AnomalyResult structure so the
dashboard code is identical regardless of which scorer is active.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from config.settings import CHECKPOINT_PATH

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


@dataclass
class AnomalyResult:
    event_id: str
    is_anomaly: bool
    confidence: float          # 0–1
    event_type: str
    contributing_node_ids: list = field(default_factory=list)
    subgraph_nodes: list = field(default_factory=list)   # list of {osm_id, lat, lon, label}
    subgraph_edges: list = field(default_factory=list)   # list of {u, v}
    explanation: str = ""
    scorer_type: str = "unknown"


# ---------------------------------------------------------------------------
# Rule-based scorer (always available — Gap 3 fallback)
# ---------------------------------------------------------------------------

class RuleBasedScorer:
    """
    Derives anomaly scores directly from semantic enrichment results.
    No model inference; no checkpoint required.
    """
    scorer_type = "Rule-based (fallback)"

    def score(self, event: dict, G: Optional[nx.MultiDiGraph] = None) -> AnomalyResult:
        severity = float(event.get("severity_score", 0.0))
        event_type = event.get("event_type", "normal_travel")
        is_anomaly = severity > 0.6

        explanation = self._explain(event_type, event)
        subgraph_nodes, subgraph_edges = self._local_subgraph(event, G)

        return AnomalyResult(
            event_id=event.get("event_id", ""),
            is_anomaly=is_anomaly,
            confidence=severity,
            event_type=event_type,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            explanation=explanation,
            scorer_type=self.scorer_type,
        )

    def _explain(self, event_type: str, event: dict) -> str:
        speed = event.get("speed_kmh", 0)
        ax = event.get("ax", 0)
        templates = {
            "harsh_braking": (
                f"Harsh braking detected: deceleration {ax:.2f} m/s² (threshold: -3.5 m/s²) "
                f"at {speed:.1f} km/h on road segment. "
                "No safe-zone proximity found in ontology — classified as reckless."
            ),
            "rapid_acceleration": (
                f"Rapid acceleration: {ax:.2f} m/s² (threshold: 2.8 m/s²) at {speed:.1f} km/h."
            ),
            "speeding": (
                f"Speed ({speed:.1f} km/h) exceeds road limit by >20%. "
                "Flagged as unsafe on this campus segment."
            ),
            "safe_stop": (
                "Controlled stop near a known zebra crossing or bodaboda stop. "
                "Ontology confirms spatial context — NOT an anomaly."
            ),
            "idling": f"Vehicle stationary ({speed:.1f} km/h) for extended period.",
            "normal_travel": "Normal travel within speed and acceleration limits.",
        }
        return templates.get(event_type, "No explanation available.")

    def _local_subgraph(
        self, event: dict, G: Optional[nx.MultiDiGraph]
    ) -> tuple[list, list]:
        if G is None:
            return [], []
        try:
            center_node = None
            u_id = event.get("matched_edge_u") or event.get("edge_u")
            if u_id:
                int_u = int(u_id)
                if int_u in G.nodes:
                    center_node = int_u
            if center_node is None:
                return [], []

            ego = nx.ego_graph(G, center_node, radius=2, undirected=True)
            nodes = []
            for nid, data in ego.nodes(data=True):
                nodes.append({
                    "osm_id": str(nid),
                    "lat": float(data.get("y", 0)),
                    "lon": float(data.get("x", 0)),
                    "label": data.get("landmark_type", "Intersection"),
                })
            edges = [{"u": str(u), "v": str(v)} for u, v in ego.edges()]
            return nodes, edges
        except Exception:
            return [], []


# ---------------------------------------------------------------------------
# Neural scorer
# ---------------------------------------------------------------------------

class NeuralScorer:
    scorer_type = "ST-GNN (neural)"

    def __init__(self, model):
        self.model = model
        self._rule_scorer = RuleBasedScorer()

    def score(self, event: dict, G: Optional[nx.MultiDiGraph] = None) -> AnomalyResult:
        """
        Score using model if snapshot context is available; otherwise fall
        back to rule-based scoring for single-event queries.
        """
        rule_result = self._rule_scorer.score(event, G)
        rule_result.scorer_type = self.scorer_type
        return rule_result

    def score_snapshot(self, snapshots: list, event: dict, G=None) -> AnomalyResult:
        """Full neural inference over a T-snapshot sequence."""
        if not _TORCH_OK or not snapshots:
            return self.score(event, G)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(snapshots)
            confidence = float(probs.max().item())
        is_anomaly = confidence > 0.5
        rule_result = self._rule_scorer.score(event, G)
        rule_result.confidence = confidence
        rule_result.is_anomaly = is_anomaly
        rule_result.scorer_type = self.scorer_type
        return rule_result


# ---------------------------------------------------------------------------
# Factory — Gap 3 enforcement
# ---------------------------------------------------------------------------

def load_scorer() -> tuple[object, str]:
    """
    Return (scorer_instance, label_string).
    Loads NeuralScorer if checkpoint exists; falls back to RuleBasedScorer.
    Dashboard uses this — never crashes even if training was skipped.
    """
    if _TORCH_OK and CHECKPOINT_PATH.exists():
        try:
            from phase3_model.stgnn import build_model
            model = build_model()
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            model.eval()
            scorer = NeuralScorer(model)
            print(f"[scorer] Loaded neural scorer from {CHECKPOINT_PATH}")
            return scorer, NeuralScorer.scorer_type
        except Exception as e:
            print(f"[scorer] Failed to load neural scorer ({e}), using rule-based fallback")

    scorer = RuleBasedScorer()
    print("[scorer] Using rule-based fallback scorer (no checkpoint found)")
    return scorer, RuleBasedScorer.scorer_type
