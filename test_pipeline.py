#!/usr/bin/env python3
"""
test_pipeline.py — End-to-end smoke test for moe_pipeline.

Exercises every major code path on a small data subset with minimal GNN epochs
so the full run completes in a few minutes rather than hours.

Usage:
    python test_pipeline.py
"""

import sys
import time
import tempfile
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_failures = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def check(label, fn):
    try:
        t0 = time.time()
        result = fn()
        print(f"  [PASS] {label}  ({time.time()-t0:.1f}s)")
        return result
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        _failures.append(label)
        return None


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

section("1 — Imports")

moe_mod     = check("moe_pipeline package", lambda: __import__("moe_pipeline"))
MoE         = check("moe_pipeline.MoE",      lambda: __import__("moe_pipeline", fromlist=["MoE"]).MoE)
Pipeline    = check("moe_pipeline.Pipeline",  lambda: __import__("moe_pipeline", fromlist=["Pipeline"]).Pipeline)
HP          = check("moe_pipeline.HP",        lambda: __import__("moe_pipeline", fromlist=["HP"]).HP)

from moe_pipeline.data      import load_processed_data, split_dataset, prepare_features, Featurizer
from moe_pipeline.experts   import ChempropDMPNN, GIN
from moe_pipeline.gate      import GatingNetwork
from moe_pipeline.moe       import MoE
from moe_pipeline.pipeline  import Pipeline
from moe_pipeline.constants import HP, N_DESC

# ---------------------------------------------------------------------------
# Monkey-patch GNN epochs so the test finishes fast.
# Production code is unchanged — only the default epoch counts are overridden
# for this process via staticmethod replacement.
# ---------------------------------------------------------------------------

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def _fast_experts(seed=42):
    seeds = [seed, seed+1, seed+2, seed+3, seed+4]
    return [
        XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                     verbosity=0, random_state=seeds[0]),
        RandomForestRegressor(n_estimators=50, max_depth=None,
                              min_samples_leaf=3, n_jobs=-1, random_state=seeds[1]),
        LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=6,
                      n_jobs=-1, verbose=-1, random_state=seeds[2]),
        ChempropDMPNN(epochs=2, hidden_size=300, depth=3, seed=seeds[3]),
        GIN(hidden_channels=64, num_layers=2, dropout=0.3,
            weight_decay=1e-4, epochs=2, seed=seeds[4]),
    ]

def _fast_experts_from_config(config, seed=42):
    seeds = [seed, seed+1, seed+2, seed+3, seed+4]
    n_est = config.get("n_estimators", 50)
    lr    = config.get("learning_rate", 0.05)
    return [
        XGBRegressor(n_estimators=n_est, learning_rate=lr, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                     verbosity=0, random_state=seeds[0]),
        RandomForestRegressor(n_estimators=50, max_depth=None,
                              min_samples_leaf=3,
                              max_features=config.get("rf_max_features", 1.0),
                              n_jobs=-1, random_state=seeds[1]),
        LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=6,
                      reg_lambda=config.get("lgbm_reg_lambda", 0.0),
                      n_jobs=-1, verbose=-1, random_state=seeds[2]),
        ChempropDMPNN(epochs=2, hidden_size=300, depth=3,
                      dropout=config.get("dmpnn_dropout", 0.0), seed=seeds[3]),
        GIN(hidden_channels=64, num_layers=config.get("gin_num_layers", 2),
            dropout=0.3, weight_decay=1e-4, epochs=2, seed=seeds[4]),
    ]

MoE._make_experts             = staticmethod(_fast_experts)
MoE._make_experts_from_config = staticmethod(_fast_experts_from_config)


# ---------------------------------------------------------------------------
# 2. Data pipeline
# ---------------------------------------------------------------------------

section("2 — Data pipeline")

df = check("load_processed_data", lambda: load_processed_data("chembl_kinase_processed.csv"))
if df is None:
    print("Cannot continue without data. Exiting.")
    sys.exit(1)

SMALL = df.sample(600, random_state=42)

train_df, test_df = check(
    "split_dataset (scaffold)",
    lambda: split_dataset(SMALL, split="scaffold", train_frac=0.8, seed=42),
)

data = check(
    "prepare_features",
    lambda: prepare_features(train_df, test_df),
)
X_train, y_train, smi_train, X_test, y_test, smi_test = data

print(f"         train {X_train.shape}  test {X_test.shape}")

check(
    "Featurizer.featurize_single",
    lambda: Featurizer().featurize_single(smi_train[0]),
)


# ---------------------------------------------------------------------------
# 3. Individual experts
# ---------------------------------------------------------------------------

section("3 — Individual experts (tiny data)")

TINY_N = 100
X_tiny, y_tiny, smi_tiny = X_train[:TINY_N], y_train[:TINY_N], smi_train[:TINY_N]

from sklearn.preprocessing import StandardScaler
_sc = StandardScaler()
X_tiny_desc   = X_tiny[:, :N_DESC].astype(np.float64)
X_tiny_scaled = np.concatenate([_sc.fit_transform(X_tiny_desc), X_tiny[:, N_DESC:].astype(np.float64)], axis=1)

def _train_predict(expert, X_scaled, y, smiles):
    MoE._fit_one(expert, X_scaled, y, smiles)
    p = MoE._pred_one(expert, X_scaled, smiles)
    u = MoE._uncertainty_one(expert, X_scaled, smiles)
    assert p.shape == (len(y),), f"wrong pred shape {p.shape}"
    assert u.shape == (len(y),), f"wrong unc shape {u.shape}"

check("XGBoost fit/predict/uncertainty",
      lambda: _train_predict(XGBRegressor(n_estimators=10, verbosity=0, random_state=0),
                             X_tiny_scaled, y_tiny, smi_tiny))
check("RandomForest fit/predict/uncertainty",
      lambda: _train_predict(RandomForestRegressor(n_estimators=10, random_state=0),
                             X_tiny_scaled, y_tiny, smi_tiny))
check("LightGBM fit/predict/uncertainty",
      lambda: _train_predict(LGBMRegressor(n_estimators=10, verbose=-1, random_state=0),
                             X_tiny_scaled, y_tiny, smi_tiny))
check("ChempropDMPNN fit/predict/uncertainty",
      lambda: _train_predict(ChempropDMPNN(epochs=2, seed=0),
                             X_tiny_scaled, y_tiny, smi_tiny))
check("GIN fit/predict/uncertainty",
      lambda: _train_predict(GIN(hidden_channels=64, num_layers=2, epochs=2, seed=0),
                             X_tiny_scaled, y_tiny, smi_tiny))


# ---------------------------------------------------------------------------
# 4. MoE: train all experts + gate
# ---------------------------------------------------------------------------

section("4 — MoE: full training + evaluation")

moe = MoE(seed=42)

X_train_desc   = X_train[:, :N_DESC].astype(np.float64)
X_train_scaled = moe._scale(X_train.astype(np.float64), fit=True)
X_test_desc    = X_test[:, :N_DESC].astype(np.float64)
X_test_scaled  = moe._scale(X_test.astype(np.float64), fit=False)

check("MoE.train_experts", lambda: moe.train_experts(X_train_scaled, y_train, smi_train))

E, U = check("MoE.expert_predictions",
             lambda: moe.expert_predictions(X_train_scaled, smi_train))

check("MoE.train_gating_network",
      lambda: moe.train_gating_network(
          X_train_desc, E, U, y_train,
          gate_epochs=5, gate_lr=1e-3,
      ))

metrics = check("MoE.evaluate",
                lambda: moe.evaluate(X_test_desc, X_test_scaled, smi_test, y_test))
if metrics:
    print(f"         MoE  RMSE {metrics['rmse']:.3f}  R² {metrics['r2']:.3f}")
    print(f"         per-expert RMSE: "
          + "  ".join(f"{n}={v:.3f}" for n, v in zip(MoE.EXPERT_NAMES, metrics["per_expert_rmse"])))


# ---------------------------------------------------------------------------
# 5. MoE: save / load / predict
# ---------------------------------------------------------------------------

section("5 — MoE: save / load / predict")

with tempfile.TemporaryDirectory() as tmpdir:
    check("MoE.save", lambda: moe.save(tmpdir))

    moe2 = check("MoE.load", lambda: MoE.load(tmpdir))

    if moe2 is not None:
        check("MoE.load → evaluate",
              lambda: moe2.evaluate(X_test_desc, X_test_scaled, smi_test, y_test))


# ---------------------------------------------------------------------------
# 6. Pipeline: gate-only grid search (run)
# ---------------------------------------------------------------------------

section("6 — Pipeline.run (gate-only, 1-config grid, 2 folds)")

pipeline_gate = Pipeline(
    seed=42,
    n_folds=2,
    gate_epochs=5,
    l2_regs=[0.0],
    loss_configs=[{"entropy_reg": 0.0, "load_balancing": False}],
    routing_schemes=[{"top_k": None}],
    verbose=False,
)

gate_results = check(
    "Pipeline.run",
    lambda: pipeline_gate.run(X_train, y_train, smi_train),
)
if gate_results:
    print(f"         val RMSE {gate_results['mean_rmse']:.3f}  R² {gate_results['mean_r2']:.3f}")


# ---------------------------------------------------------------------------
# 7. Pipeline: expert + gate grid search
# ---------------------------------------------------------------------------

section("7 — Pipeline.run_expert_gate_grid_search (1 expert config, 1 gate config, 2 folds)")

pipeline_full = Pipeline(
    seed=42,
    n_folds=2,
    gate_epochs=5,
    l2_regs=[0.0],
    loss_configs=[{"entropy_reg": 0.0, "load_balancing": False}],
    routing_schemes=[{"top_k": None}],
    verbose=False,
)

full_results = check(
    "Pipeline.run_expert_gate_grid_search",
    lambda: pipeline_full.run_expert_gate_grid_search(
        X_train, y_train, smi_train,
        expert_configs=[HP["expert_configs"][0]],
        oracle_regs=[0.0],
    ),
)
if full_results:
    print(f"         val RMSE {full_results['mean_rmse']:.3f}  R² {full_results['mean_r2']:.3f}")
    print(f"         best config: {full_results['best_config']}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Summary")

total   = 7   # approximate number of sections
n_fail  = len(_failures)

if not _failures:
    print("\n  All checks passed. The pipeline is ready for a production run.\n")
else:
    print(f"\n  {n_fail} check(s) FAILED:")
    for f in _failures:
        print(f"    - {f}")
    print()
    sys.exit(1)
