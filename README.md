# moe_pipeline

A Python package implementing a **Mixture-of-Experts (MoE)** pipeline for kinase drug activity prediction (pIC50 regression).  
Built for **BioE 242** to investigate whether MoE benefits more in low-data or high-data regimes.

---

## Research Question

> Does an MoE ensemble of heterogeneous expert models (XGBoost, Random Forest, LightGBM, D-MPNN, GIN) improve pIC50 regression accuracy compared to individual models, and does this advantage grow or shrink with training set size?

---

## Package Structure

```
moe_pipeline/
├── __init__.py       Public API: MoE, Pipeline, moe_production_pipeline, HP
├── __main__.py       Entry point — python -m moe_pipeline <command>
├── cli.py            Argparse CLI: production / train / predict subcommands
├── constants.py      N_DESC=200, K_OOF=5, NON_FEAT_COLS, HP (default hyperparameters)
├── data.py           Featurizer, add_murcko_scaffolds, split_dataset,
│                       load_processed_data, prepare_features, process_dataset
├── experts.py        ChempropDMPNN (D-MPNN), GIN, atom/bond feature functions
├── gate.py           GatingNetwork — MLP router with save/load
├── moe.py            MoE class — 5-expert manager with save/load
├── pipeline.py       Pipeline — K-fold CV + expert/gate grid search
└── production.py     moe_production_pipeline — full production run
```

### Module Descriptions

| File | Contents |
|------|----------|
| `constants.py` | All numeric constants and the full default hyperparameter grid (`HP` dict) |
| `data.py` | RDKit+Morgan featurization, Murcko scaffold computation, scaffold/random splitting, CSV loading |
| `experts.py` | `ChempropDMPNN` (Chemprop v2 D-MPNN) and `GIN` (PyG graph network) with MC-Dropout uncertainty |
| `gate.py` | `GatingNetwork` MLP trained with MSE + oracle supervision + optional entropy/load-balancing reg |
| `moe.py` | `MoE` — wraps 5 experts + gate; provides `train_experts`, `expert_predictions`, `evaluate`, `save`, `load` |
| `pipeline.py` | `Pipeline` — outer K-fold CV with inner OOF expert training; `run()` and `run_expert_gate_grid_search()` |
| `production.py` | `moe_production_pipeline()` — 10k vs. full-data comparison, delta metric plots, scaffold routing analysis |
| `cli.py` | Argparse-based CLI wiring `production`, `train`, `predict` subcommands |

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `rdkit`, `descriptastorus`, `torch`, `torch_geometric`, `chemprop`, `pytorch_lightning`, `xgboost`, `lightgbm`, `scikit-learn`, `joblib`, `matplotlib`, `pandas`, `numpy`.

---

## Quick Start

The package expects a **pre-processed CSV** (`chembl_kinase_processed.csv`) produced by `data.process_dataset()`. This file ships with the repository.

---

## CLI Commands

### `production` — Full production run (primary use case)

Runs the pipeline in both the low-data (10k scaffold-proportional subsample) and full-data regimes, compares all model types on the fixed test set, and saves three deliverables.

```bash
python -m moe_pipeline production \
  --input  chembl_kinase_processed.csv \
  --output-dir runs/production_run
```

**Outputs saved to `--output-dir`:**
- `delta_metrics.png` — bar charts showing ΔRMSE and ΔR² (full − 10k) for MoE, each expert, and the equal-weighted ensemble
- `best_config.json` — best expert+gate hyperparameter config for both data regimes, with validation and test metrics
- `scaffold_weights.png` — expert gate weight distributions across the top-10 most frequent test-set scaffolds (bar = mean, box plot = compound-level distribution)

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input / -i` | *(required)* | Path to processed CSV |
| `--output-dir / -o` | `runs/production_<timestamp>` | Output directory |
| `--train-frac` | `0.85` | Fraction of scaffolds assigned to train |
| `--split` | `scaffold` | Split strategy (`scaffold` or `random`) |
| `--seed` | *(from HP)* | Random seed (overrides `HP["seed"]`) |
| `--n-folds` | *(from HP)* | Outer CV folds (overrides `HP["n_folds"]`) |
| `--n-low` | `10000` | Low-data subsample size |
| `--hp-file` | — | JSON file with HP overrides |
| `--quiet` | — | Suppress progress output |

---

### `train` — Train and save a model

Trains the full pipeline on training data, grid-searches gate (and optionally expert) hyperparameters, evaluates on the held-out test set, and saves the model to disk for later inference.

```bash
# Gate-only grid search (faster)
python -m moe_pipeline train \
  --input     chembl_kinase_processed.csv \
  --model-dir models/my_run

# Full expert+gate grid search
python -m moe_pipeline train \
  --input      chembl_kinase_processed.csv \
  --model-dir  models/my_run \
  --expert-grid
```

**Saved artifacts (`--model-dir/`):**
```
experts/xgb.joblib, rf.joblib, lgbm.joblib
experts/chemprop.pt, gin.pt, gnn_configs.joblib
gate_net.pt, gate_scaler.joblib
desc_scaler.joblib
config.json
best_config.json     (hyperparams + test metrics)
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input / -i` | *(required)* | Path to processed CSV |
| `--model-dir / -m` | *(required)* | Directory to save model artifacts |
| `--train-frac` | `0.85` | Train/test split fraction |
| `--split` | `scaffold` | Split strategy |
| `--seed` | *(from HP)* | Random seed |
| `--n-folds` | *(from HP)* | Outer CV folds |
| `--gate-epochs` | *(from HP)* | Gate training epochs |
| `--gate-lr` | *(from HP)* | Gate learning rate |
| `--expert-grid` | — | Also search over expert model configurations |
| `--hp-file` | — | JSON file with HP overrides |
| `--quiet` | — | Suppress progress output |

---

### `predict` — Run inference on new compounds

Loads a saved model and predicts pIC50 for SMILES strings in a CSV.

```bash
python -m moe_pipeline predict \
  --model-dir models/my_run \
  --input     new_compounds.csv \
  --output    predictions.csv \
  --include-expert-preds \
  --include-weights
```

**Output CSV columns:**
- `predicted_pIC50` — MoE ensemble prediction
- `pred_xgb, pred_rf, pred_lgbm, pred_chemprop, pred_gin` — per-expert (with `--include-expert-preds`)
- `weight_xgb, weight_rf, ...` — gate routing weights (with `--include-weights`)

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir / -m` | *(required)* | Saved model directory |
| `--input / -i` | *(required)* | CSV with SMILES column |
| `--output / -o` | *(required)* | Output CSV path |
| `--smiles-col` | `smiles` | Name of SMILES column |
| `--include-expert-preds` | — | Add per-expert prediction columns |
| `--include-weights` | — | Add gate routing weight columns |
| `--quiet` | — | Suppress progress output |

---

## Programmatic API

```python
import pandas as pd
from moe_pipeline import MoE, Pipeline, moe_production_pipeline, HP
from moe_pipeline.data import load_processed_data, split_dataset, prepare_features

# Load and split data
df = load_processed_data("chembl_kinase_processed.csv")
train_df, test_df = split_dataset(df, split="scaffold", train_frac=0.85)
X_train, y_train, smi_train, X_test, y_test, smi_test = prepare_features(train_df, test_df)

# Run gate-only grid search
pipeline = Pipeline(seed=HP["seed"], n_folds=HP["n_folds"], verbose=True)
results  = pipeline.run(X_train, y_train, smi_train)
final_moe  = results["final_moe"]
best_config = results["best_config"]

# Evaluate on test set
from moe_pipeline.constants import N_DESC
X_test_desc   = X_test[:, :N_DESC]
X_test_scaled = np.concatenate(
    [final_moe.scaler.transform(X_test_desc), X_test[:, N_DESC:]], axis=1
)
metrics = final_moe.evaluate(X_test_desc, X_test_scaled, smi_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.4f}   R²: {metrics['r2']:.4f}")

# Full production run
prod_results = moe_production_pipeline(
    train_df=train_df,
    test_df=test_df,
    path="runs/production",
    HP=HP,
    n_low=10_000,
    verbose=True,
)
```

---

## Pipeline Architecture

### Expert Models

| Expert | Type | Input |
|--------|------|-------|
| XGBoost | Gradient boosting | Tabular (200 descriptors + 2048 FP bits) |
| RandomForest | Tree ensemble | Tabular |
| LightGBM | Gradient boosting | Tabular |
| ChempropDMPNN | D-MPNN graph network | SMILES → PyG graphs (33-D atom, 6-D bond) |
| GIN | Graph Isomorphism Network | SMILES → PyG graphs (4 layers, hidden=256) |

### Gating Network

MLP router that weights expert outputs:
- **Input (210-D):** `[raw descriptors (200) | expert predictions (5) | expert uncertainties (5)]`
- **Architecture:** `Linear(210) → ReLU(100) → ReLU(80) → Softmax(5)`
- **Loss:** `L_MSE + oracle_reg × L_oracle − entropy_reg × H(w) + load_balancing × L_lb`
- **Routing modes:** soft, top-1, top-2, top-3

### Training Flow (OOF Stacking)

1. **Outer K-fold CV** (K=4): 85% pool / 15% val per fold
2. **Inner 5-fold OOF** on pool: honest expert predictions (no data leakage)
3. **Hyperparameter search** over gate configs: `l2_regs × oracle_regs × loss_configs × routing_schemes`
4. **Optional expert config search** over 4 preset configurations (A/B/C/D corners)
5. **Final MoE** retrained on 100% of data with best config

### Default Hyperparameter Grid (`HP`)

#### Expert configs (4 total)

The 4 configs are corners of a 2×2 grid: (tabular-regularized) × (GNN-regularized).

| Config | n_estimators | learning_rate | lgbm_reg_lambda | rf_max_features | dmpnn_dropout | gin_num_layers |
|--------|-------------|---------------|-----------------|-----------------|---------------|----------------|
| A — baseline | 200 | 0.05 | 0.0 | 1.0 | 0.0 | 4 |
| B — tabular-tuned | 500 | 0.02 | 1.0 | "sqrt" | 0.0 | 4 |
| C — GNN-tuned | 200 | 0.05 | 0.0 | 1.0 | 0.2 | 3 |
| D — all-tuned | 500 | 0.02 | 1.0 | "sqrt" | 0.2 | 3 |

#### Gate configs (288 per expert config)

| Dimension | Values | Count |
|-----------|--------|-------|
| `l2_regs` (weight decay on gate MLP) | 0.0, 0.001, 0.01, 0.1 | 4 |
| `oracle_regs` (cross-entropy toward best expert) | 0.0, 0.01, 0.1 | 3 |
| `loss_configs` (entropy reg + load balancing) | baseline, entropy=1e-4, 1e-3, 1e-2, 0.1, load-balancing | 6 |
| `routing_schemes` (top-k sparsity) | soft (all experts), top-1, top-2, top-3 | 4 |
| **Total** | | **288** |

#### Total evaluations

```
288 gate × 4 expert configs × 4 folds = 4,608 per data regime
                                       × 2 regimes (10k + full) = 9,216 total
```

Each evaluation trains a gate network (300 epochs) and scores it on the validation fold. Expert models are trained once per expert config per fold via inner OOF, then reused across all 288 gate configs for that fold — so you're not retraining 9,216 full expert sets, just 4 expert configs × 4 folds × 2 regimes = 32.
