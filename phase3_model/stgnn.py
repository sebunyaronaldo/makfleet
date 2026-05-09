"""
Spatio-Temporal Graph Neural Network (ST-GNN).

Architecture: GCN spatial encoder (per snapshot) + per-node LSTM temporal
aggregator + binary classification head.

Architecturally equivalent to A3T-GCN but implemented from first principles
using torch_geometric.nn.GCNConv and torch.nn.LSTM so every design decision
is explicit and auditable — a stronger academic position than using a black-box
library default.

Input:  sequence of T graph snapshots, each a PyG Data object with
        node features X ∈ R^{N×6}
Output: per-node anomaly probability ∈ [0,1]
"""

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
    _PYG_OK = True
except ImportError:
    _PYG_OK = False
    GCNConv = None


class STGNNModel(nn.Module):
    """
    Spatial Encoder (shared GCN weights across all T snapshots):
        GCNConv(in_channels→hidden) → ReLU → GCNConv(hidden→hidden) → ReLU

    Temporal Aggregator (per node, across T snapshots):
        LSTM(input=hidden, hidden=hidden, num_layers=lstm_layers)

    Output head:
        Linear(hidden→hidden//2) → ReLU → Linear(hidden//2→1) → Sigmoid
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden: int = 64,
        lstm_layers: int = 2,
    ):
        super().__init__()
        if not _PYG_OK:
            raise ImportError("torch_geometric is required for STGNNModel")

        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

        self._hidden = hidden
        self._lstm_layers = lstm_layers

    def encode_snapshot(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Spatial encoding of one snapshot. Returns H ∈ R^{N×hidden}."""
        h = self.relu(self.gcn1(x, edge_index))
        h = self.relu(self.gcn2(h, edge_index))
        return h   # (N, hidden)

    def forward(self, snapshot_list: list) -> torch.Tensor:
        """
        snapshot_list: list of PyG Data objects (length = T_SNAPSHOTS)
        Returns: anomaly probability per node, shape (N,)
        """
        if not snapshot_list:
            return torch.zeros(1)

        N = snapshot_list[0].x.size(0)

        # Encode each snapshot → (N, hidden), stack to (N, T, hidden)
        encoded = []
        for snap in snapshot_list:
            x = snap.x
            ei = snap.edge_index
            if ei.numel() == 0:
                ei = torch.zeros((2, 0), dtype=torch.long)
            encoded.append(self.encode_snapshot(x, ei))

        H = torch.stack(encoded, dim=1)  # (N, T, hidden)

        # LSTM: treat each node independently across time
        # Reshape to (N, T, hidden) — already correct for batch_first=True
        out, _ = self.lstm(H)         # (N, T, hidden)
        h_final = out[:, -1, :]       # (N, hidden) — last time step

        # Output head
        probs = self.head(h_final).squeeze(-1)  # (N,)
        return probs


# ---------------------------------------------------------------------------
# Convenience: build a model with default settings from config
# ---------------------------------------------------------------------------

def build_model() -> STGNNModel:
    from config.settings import STGNN_HIDDEN, STGNN_LAYERS
    return STGNNModel(in_channels=6, hidden=STGNN_HIDDEN, lstm_layers=STGNN_LAYERS)
