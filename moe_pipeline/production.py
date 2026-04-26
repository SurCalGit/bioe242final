import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score

from .constants import N_DESC, NON_FEAT_COLS, HP as DEFAULT_HP
from .moe import MoE
from .pipeline import Pipeline


def moe_production_pipeline(train_df, test_df, path, HP=None, n_low=10_000, verbose=True):
    """
    --- Main Steps --- 
    1. Sample a 10k subset of training data (each SMILES proportional to scaffold size) and run the full
       expert-plus-gate grid search on both subsets. Evaluate all model types on the test
       set and plot change in RMSE and R² between models trained on the two datasets. 
    2. Save the best gate configuration for each data regime. 
    3. Plot per-scaffold expert weight distributions for the top-10 most frequent
       scaffolds in the test set, showing mean bars with box-plot overlays.

    Parameters:
        path : str
            Output directory. Created if it does not exist.
        HP : dict
            Hyperparameter dictionary.
        verbose : bool
            Print progress if True (default)

    Returns:
        dict: low_data_metrics, full_data_metrics, delta_rmse, delta_r2,
                    best_config_low, best_config_full
    """

    import matplotlib
    matplotlib.use("Agg")
    if HP is None:
        hp = {**DEFAULT_HP}
    else:
        hp = {**DEFAULT_HP, **HP}
    HP = hp  # use resolved copy

    os.makedirs(path, exist_ok=True)

    EXPERT_NAMES  = MoE.EXPERT_NAMES # ["xgb","rf","lgbm","chemprop","gin"]
    MODEL_LABELS  = ["MoE", "XGB", "RF", "LGBM", "Chemprop", "GIN", "Ensemble"]
    NON_FEAT_COLS = ["smiles", "pIC50", "target", "scaffold", "scaffold_idx"]

    # Helper methods 
    
    def _log(msg):
        if verbose:
            print(msg, flush=True)

    def _sample_scaffold_proportional(train_df, n, seed):
        """
        Sample n rows from train_df so each scaffold contributes approximately
        the same fraction as it does in the full training set.
        """
        rng = np.random.default_rng(seed)
        scaffold_sizes = train_df.groupby("scaffold").size()
        total = len(train_df)
        scaffolds = list(scaffold_sizes.index)

        # Proportional allocation with rounding correction
        allocated = {s: max(0, round(n * scaffold_sizes[s] / total)) for s in scaffolds}
        diff = n - sum(allocated.values())
        if diff != 0:
            sign = int(np.sign(diff))
            sorted_scafs = sorted(scaffolds, key=lambda s: scaffold_sizes[s], reverse=(sign > 0))
            for s in sorted_scafs:
                if diff == 0:
                    break
                allocated[s] += sign
                diff -= sign

        sampled_idx = []
        for scaf, k in allocated.items():
            k = min(int(k), scaffold_sizes[scaf])
            if k <= 0:
                continue
            scaf_row_idx = train_df.index[train_df["scaffold"] == scaf].tolist()
            chosen = rng.choice(scaf_row_idx, size=k, replace=False)
            sampled_idx.extend(chosen.tolist())

        return train_df.loc[sampled_idx].reset_index(drop=True)

    def _build_xy(df):
        """Extract X (numpy, float64, sanitized), y, smiles from a kinase DataFrame."""
        fc = [c for c in df.columns if c not in NON_FEAT_COLS]
        X = df[fc].astype(np.float64).values
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return X, df["pIC50"].values, df["smiles"].tolist()

    def _run_pipeline(X_raw, y, smiles, label):
        """Instantiate a Pipeline and run the full expert plus gate grid search."""
        _log(f"\n{'='*62}\n  Pipeline: {label}  ({len(y):,} samples)\n{'='*62}")
        pipe = Pipeline(
            seed=HP["seed"],
            n_folds=HP["n_folds"],
            gate_hidden_sizes=HP["gate_hidden_sizes"],
            gate_lr=HP["gate_lr"],
            gate_epochs=HP["gate_epochs"],
            gate_oracle_reg=HP["gate_oracle_reg"],
            l2_regs=HP["l2_regs"],
            loss_configs=HP["loss_configs"],
            routing_schemes=HP["routing_schemes"],
            verbose=verbose,
        )
        results = pipe.run_expert_gate_grid_search(
            X_raw, y, smiles,
            expert_configs=HP["expert_configs"],
            oracle_regs=HP["oracle_regs"],
        )
        return pipe, results

    def _evaluate_on_test(final_moe, best_config, X_test_raw, y_test_arr, smiles_test_lst):
        """
        Evaluate the final MoE (and its constituent experts + equal-weighted ensemble)
        on the test set.

        Returns
        -------
            metrics : dict — keys: moe_rmse, moe_r2, <expert>_rmse, <expert>_r2,
                               ensemble_rmse, ensemble_r2
            E : ndarray (n_test, 5) — expert predictions
            weights : ndarray (n_test, 5) — gate routing weights
        """
        top_k = best_config.get("top_k", None)

        X_desc = X_test_raw[:, :N_DESC].astype(np.float64)
        X_scaled = np.concatenate(
            [final_moe.scaler.transform(X_desc),
             X_test_raw[:, N_DESC:].astype(np.float64)], axis=1
        )

        E, U = final_moe.expert_predictions(X_scaled, smiles_test_lst)
        X_gate = np.concatenate([X_desc, E, U], axis=1)
        weights  = final_moe.gate.predict_weights(X_gate, top_k=top_k)
        moe_pred = (weights * E).sum(axis=1)
        ens_pred = E.mean(axis=1)

        metrics = {
            "moe_rmse": float(np.sqrt(mean_squared_error(y_test_arr, moe_pred))),
            "moe_r2": float(r2_score(y_test_arr, moe_pred)),
            "ensemble_rmse": float(np.sqrt(mean_squared_error(y_test_arr, ens_pred))),
            "ensemble_r2": float(r2_score(y_test_arr, ens_pred)),
        }
        for k_idx, name in enumerate(EXPERT_NAMES):
            metrics[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(y_test_arr, E[:, k_idx])))
            metrics[f"{name}_r2"] = float(r2_score(y_test_arr, E[:, k_idx]))

        return metrics, E, weights

    # STEP 1: construct datasets 

    N_LOW = n_low
    _log(f"\n[Step 1] Sampling {N_LOW:,} scaffold-proportional compounds from training set...")
    low_df = _sample_scaffold_proportional(train_df, N_LOW, seed=HP["seed"])
    X_low, y_low, smi_low = _build_xy(low_df)
    _log(f"  10k subset : {len(y_low):,} rows | {low_df['scaffold'].nunique()} unique scaffolds")

    X_full, y_full, smi_full = _build_xy(train_df)
    _log(f"  Full train : {len(y_full):,} rows | {train_df['scaffold'].nunique()} unique scaffolds")

    X_test_raw, y_test_arr, smi_test = _build_xy(test_df)

    # STEP 2: Run pipelines 

    _, results_low  = _run_pipeline(X_low,  y_low,  smi_low,  "10k subsample")
    _, results_full = _run_pipeline(X_full, y_full, smi_full, "Full training set")

    # STEP 3: Evaluate on test set 

    _log("\n[Step 3] Evaluating both final MoEs on the held-out test set...")

    metrics_low,  _, W_low  = _evaluate_on_test(
        results_low["final_moe"],  results_low["best_config"],  X_test_raw, y_test_arr, smi_test
    )
    metrics_full, _, W_full = _evaluate_on_test(
        results_full["final_moe"], results_full["best_config"], X_test_raw, y_test_arr, smi_test
    )

    _log("\n  === Test metrics (10k subsample) ===")
    for k, v in sorted(metrics_low.items()):
        _log(f"    {k:<22}: {v:.4f}")
    _log("\n  === Test metrics (full training) ===")
    for k, v in sorted(metrics_full.items()):
        _log(f"    {k:<22}: {v:.4f}")

    ## Delta metrics (full − 10k) 

    rmse_keys = ["moe_rmse"] + [f"{n}_rmse" for n in EXPERT_NAMES] + ["ensemble_rmse"]
    r2_keys   = ["moe_r2"]   + [f"{n}_r2"   for n in EXPERT_NAMES] + ["ensemble_r2"]

    delta_rmse = [metrics_full[k] - metrics_low[k] for k in rmse_keys]
    delta_r2   = [metrics_full[k] - metrics_low[k] for k in r2_keys]

    ## Delta bar charts

    MODEL_COLORS  = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#00BCD4", "#795548"]
    x     = np.arange(len(MODEL_LABELS))
    bar_w = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_specs = [
        (axes[0], delta_rmse, "ΔRMSE (Full − 10k)",
         "Change in Test RMSE: Full vs. 10k\n(negative = more data helped)"),
        (axes[1], delta_r2,   "ΔR² (Full − 10k)",
         "Change in Test R²: Full vs. 10k\n(positive = more data helped)"),
    ]
    for ax, delta, ylabel, title in plot_specs:
        bars = ax.bar(x, delta, width=bar_w, color=MODEL_COLORS, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        span = max(abs(max(delta)), abs(min(delta))) or 1.0
        for bar, val in zip(bars, delta):
            va    = "bottom" if val >= 0 else "top"
            nudge = span * 0.03 * (1 if val >= 0 else -1)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + nudge,
                f"{val:+.3f}", ha="center", va=va, fontsize=8, fontweight="bold",
            )

    fig.tight_layout()
    delta_fig_path = os.path.join(path, "delta_metrics.png")
    fig.savefig(delta_fig_path, dpi=150, bbox_inches="tight")
    _log(f"[Saved] {delta_fig_path}")

    # STEP 4: Save best MoE configuration

    _log("\n[Step 2] Recording best MoE configurations...")

    config_out = {
        "n_low":  N_LOW,
        "n_full": int(len(y_full)),
        "low_data": {
            "best_config":  results_low["best_config"],
            "val_rmse":     results_low["mean_rmse"],
            "val_r2":       results_low["mean_r2"],
            "test_metrics": metrics_low,
        },
        "full_data": {
            "best_config":  results_full["best_config"],
            "val_rmse":     results_full["mean_rmse"],
            "val_r2":       results_full["mean_r2"],
            "test_metrics": metrics_full,
        },
        "delta_rmse": dict(zip(MODEL_LABELS, [round(v, 6) for v in delta_rmse])),
        "delta_r2":   dict(zip(MODEL_LABELS, [round(v, 6) for v in delta_r2])),
    }

    config_path = os.path.join(path, "best_config.json")
    with open(config_path, "w") as fh:
        json.dump(config_out, fh, indent=2)
    _log(f"[Saved] {config_path}")

    # STEP 5: Per-scaffold expert weight bar + box plot 

    _log("\n[Step 3] Computing per-scaffold expert gate weights on test set...")

    ## Use gate weights from the full-data final MoE (W_full: n_test × 5)
    test_scaffolds = test_df["scaffold"].values

    ## Top-10 test-set scaffolds by compound count (skip NaN / empty strings)
    valid_mask   = np.array([s is not None and pd.notna(s) and s != "" for s in test_scaffolds])
    scaf_counter = Counter(test_scaffolds[valid_mask])
    top10_scaffolds = [scaf for scaf, _ in scaf_counter.most_common(10)]

    EXPERT_COLORS   = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
    EXPERT_DISPLAY  = ["XGB", "RF", "LGBM", "Chemprop", "GIN"]

    n_scaf = len(top10_scaffolds)
    n_exp  = len(EXPERT_NAMES)

    ##Layout: 5 bars per scaffold group, groups separated by a small gap
    bar_w_s    = 0.14
    group_gap  = 0.25
    group_width = n_exp * bar_w_s + group_gap

    fig, ax = plt.subplots(figsize=(max(14, n_scaf * 2.2), 6))
    group_centers = []

    for s_idx, scaf in enumerate(top10_scaffolds):
        scaf_mask  = np.array([s == scaf for s in test_scaffolds])
        n_cpds     = int(scaf_mask.sum())
        ### Center of this group (midpoint of 5 bars)
        grp_center = s_idx * group_width + (n_exp - 1) * bar_w_s / 2
        group_centers.append(grp_center)

        for e_idx, (ename, ecolor, edisplay) in enumerate(
            zip(EXPERT_NAMES, EXPERT_COLORS, EXPERT_DISPLAY)
        ):
            x_pos = s_idx * group_width + e_idx * bar_w_s
            w_vec = W_full[scaf_mask, e_idx]       ### gate weights for this expert on this scaffold

            ### Mean bar
            ax.bar(
                x_pos, w_vec.mean(), width=bar_w_s * 0.85,
                color=ecolor, alpha=0.70,
                label=edisplay if s_idx == 0 else "_nolegend_",
                edgecolor="white", linewidth=0.4,
            )

            ### Distribution box plot overlaid on bar
            ax.boxplot(
                w_vec,
                positions=[x_pos],
                widths=bar_w_s * 0.55,
                patch_artist=False,
                manage_ticks=False,
                medianprops=dict(color=ecolor, linewidth=2.0),
                whiskerprops=dict(color=ecolor, linewidth=1.1, alpha=0.85),
                capprops=dict(color=ecolor, linewidth=1.1, alpha=0.85),
                boxprops=dict(color=ecolor, linewidth=1.1),
                flierprops=dict(marker=".", markersize=3, color=ecolor, alpha=0.5),
            )

        ### Compound count annotation below the group label
        ax.text(
            grp_center, -0.055, f"n={n_cpds}",
            ha="center", va="top", fontsize=7.5, color="#444444",
            transform=ax.get_xaxis_transform(),
        )

    ## x-tick labels: scaffold SMILES truncated to 28 chars
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [s[:28] + "\u2026" if len(s) > 28 else s for s in top10_scaffolds],
        rotation=45, ha="right", fontsize=8,
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Gate Weight", fontsize=12)
    ax.set_title(
        "Expert Gate Weight Distribution by Scaffold\n"
        "(Top-10 Most Frequent Test-Set Scaffolds  |  bar = mean, box = distribution)",
        fontsize=12,
    )
    ax.legend(title="Expert", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    scaffold_fig_path = os.path.join(path, "scaffold_weights.png")
    fig.savefig(scaffold_fig_path, dpi=150, bbox_inches="tight")
    _log(f"[Saved] {scaffold_fig_path}")

    # SUMMARY 

    _log(f"\n{'='*62}")
    _log(f"  Production run complete. Outputs saved to: {path}")
    _log(f"    delta_metrics.png   — ΔRMSE / ΔR² bar charts")
    _log(f"    best_config.json    — best gate + expert configurations")
    _log(f"    scaffold_weights.png — per-scaffold routing weight analysis")
    _log(f"{'='*62}")

    return {
        "low_data_metrics":  metrics_low,
        "full_data_metrics": metrics_full,
        "delta_rmse": dict(zip(MODEL_LABELS, delta_rmse)),
        "delta_r2":  dict(zip(MODEL_LABELS, delta_r2)),
        "best_config_low": results_low["best_config"],
        "best_config_full": results_full["best_config"],
    }
