"""
ST-GNN training loop with semester-based splits and OOD evaluation.

Causal validity: we strictly forbid random train/test splits. Training on
Semester 1 and evaluating on Semester 2 enforces temporal causality —
the model can never see future routing behaviour during training.

Additionally, a spatial hold-out split (eastern campus zone) evaluates
true OOD robustness against geographic distribution shift.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    _SK_OK = True
except ImportError:
    _SK_OK = False

from config.settings import (
    STGNN_LR, STGNN_EPOCHS, STGNN_POS_WEIGHT, CHECKPOINT_PATH,
)
from phase3_model.stgnn import build_model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _make_loss(pos_weight: float = STGNN_POS_WEIGHT):
    return nn.BCELoss(
        weight=None  # applied per-sample below via pos_weight scaling
    )


def _weighted_bce(probs: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weights = torch.where(targets > 0, torch.tensor(pos_weight), torch.tensor(1.0))
    eps = 1e-7
    loss = -(weights * (targets * torch.log(probs + eps) + (1 - targets) * torch.log(1 - probs + eps)))
    return loss.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(all_probs: list[float], all_labels: list[float]) -> dict:
    if not _SK_OK or len(all_probs) < 2:
        return {"auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    return {
        "auc": round(auc, 4),
        "f1": round(f1_score(all_labels, preds, zero_division=0), 4),
        "precision": round(precision_score(all_labels, preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, preds, zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_samples: list,
    val_samples: list,
    epochs: int = STGNN_EPOCHS,
) -> "STGNNModel":
    model = build_model()
    optimizer = Adam(model.parameters(), lr=STGNN_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    patience = 10
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for sample in train_samples:
            snaps = sample["snapshots"]
            label = sample["label"]
            if not snaps:
                continue
            optimizer.zero_grad()
            probs = model(snaps)
            # Sample-level label broadcast to all nodes
            targets = torch.full((probs.size(0),), label, dtype=torch.float)
            loss = _weighted_bce(probs, targets, STGNN_POS_WEIGHT)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate(model, val_samples, "val")
            val_auc = val_metrics["auc"]
            print(
                f"Epoch {epoch:3d}/{epochs} | loss={total_loss/max(len(train_samples),1):.4f} "
                f"| val AUC={val_auc:.4f}"
            )
            if val_auc > best_auc:
                best_auc = val_auc
                no_improve = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  ✓ Checkpoint saved (AUC={best_auc:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Load best checkpoint
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: "STGNNModel", samples: list, split_name: str = "test") -> dict:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for sample in samples:
            snaps = sample["snapshots"]
            label = sample["label"]
            if not snaps:
                continue
            probs = model(snaps)
            # Use max node anomaly probability as sample-level score
            sample_prob = float(probs.max().item())
            all_probs.append(sample_prob)
            all_labels.append(label)

    metrics = _compute_metrics(all_probs, all_labels)
    print(f"[{split_name}] AUC={metrics['auc']} F1={metrics['f1']} "
          f"P={metrics['precision']} R={metrics['recall']}")
    return metrics


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_training(train_s, val_s, test_temporal_s, test_ood_s) -> dict:
    print("=== ST-GNN Training ===")
    model = train(train_s, val_s)

    print("=== Evaluation ===")
    val_m = evaluate(model, val_s, "val")
    temporal_m = evaluate(model, test_temporal_s, "test_temporal_OOD")
    spatial_m = evaluate(model, test_ood_s, "test_spatial_OOD")

    results = {
        "val": val_m,
        "temporal_ood": temporal_m,
        "spatial_ood": spatial_m,
    }

    results_path = CHECKPOINT_PATH.parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[trainer] Results saved to {results_path}")
    return results


if __name__ == "__main__":
    from phase3_model.graph_dataset import build_dataset, split_dataset

    print("Building Semester 1 dataset…")
    sem1 = build_dataset(semester=1)
    print("Building Semester 2 dataset…")
    sem2 = build_dataset(semester=2)

    all_samples = sem1 + sem2
    train_s, val_s, test_t, test_ood = split_dataset(all_samples)
    run_training(train_s, val_s, test_t, test_ood)
