# MakFleet Notebooks

Two Jupyter/Colab notebooks for the MakFleet project.

## Notebooks

### 1. MakFleet_ST_GNN_Training_Colab.ipynb
**Purpose:** Train the Spatio-Temporal Graph Neural Network on GPU via Google Colab.

**How to use:**
1. Upload `data/pyg_train.pt`, `data/pyg_val.pt`, `data/pyg_test_temporal.pt`, `data/pyg_test_ood.pt` to Google Drive under `MakFleet/data/`
2. Open this notebook in Google Colab
3. Set Runtime → Change runtime type → **A100 GPU**
4. Run all cells
5. Download `stgnn_checkpoint_colab.pt` and place at `data/stgnn_checkpoint.pt`

**Expected training time:** ~2 minutes on A100 GPU (vs 20+ minutes on CPU)

**Results achieved:**
- Validation AUC: 0.9268
- Temporal OOD AUC: 0.9471
- Recall: 1.000 across all splits

---

### 2. MakFleet_EDA.ipynb
**Purpose:** Exploratory Data Analysis on the 826,260-record MakFleet simulated dataset.

**How to use:**
1. Ensure Neo4j is running locally (`docker-compose up -d`)
2. Ensure the pipeline has been run (`python run_pipeline.py`)
3. Run cells top to bottom

**Outputs (saved as PNG):**
- `eda_temporal_hourly.png` — Hourly demand and anomaly rate
- `eda_temporal_dow.png` — Day of week patterns
- `eda_anomaly_breakdown.png` — Event type distribution
- `eda_speed_distribution.png` — Speed by event type
- `eda_driver_risk.png` — Driver risk profiles
- `eda_spatial_hotspots.png` — Top 15 campus intersections
- `eda_semester_comparison.png` — Semester 1 vs 2 comparison
- `eda_sensor_validation.png` — Map-matching validation

**Key findings:**
- Peak demand: 08:00 (morning lecture rush)
- Most common anomaly: Speeding
- Anomaly rate: 8.8%
- Map-matching confidence: 0.906
