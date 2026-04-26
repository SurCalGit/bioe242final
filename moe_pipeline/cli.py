"""
Command-line interface for moe_pipeline.

Subcommands
-----------
production  Run the full low-data vs. high-data production comparison.
train       Train the pipeline on a processed CSV and save the final model.
predict     Load a saved model and run inference on new SMILES.
"""

import argparse
import datetime
import json
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe_pipeline",
        description="Kinase pIC50 MoE Pipeline",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    _add_production_parser(sub)
    _add_train_parser(sub)
    _add_predict_parser(sub)

    return parser


def _add_common_data_args(p: argparse.ArgumentParser):
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to the pre-processed kinase CSV (chembl_kinase_processed.csv).",
    )
    p.add_argument(
        "--train-frac", type=float, default=0.85, metavar="FRAC",
        help="Fraction of scaffolds assigned to training set (default: 0.85).",
    )
    p.add_argument(
        "--split", choices=["scaffold", "random"], default="scaffold",
        help="Train/test split strategy (default: scaffold).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Overrides HP['seed'] when set.",
    )


def _add_production_parser(sub):
    p = sub.add_parser(
        "production",
        help="Low-data vs. high-data regime comparison (main production run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_data_args(p)
    p.add_argument(
        "--output-dir", "-o", default=None, metavar="DIR",
        help=(
            "Directory for all outputs. "
            "Defaults to runs/production_<YYYYMMDD_HHMMSS>."
        ),
    )
    p.add_argument(
        "--n-folds", type=int, default=None, metavar="K",
        help="Number of outer CV folds. Overrides HP['n_folds'] when set.",
    )
    p.add_argument(
        "--n-low", type=int, default=10_000, metavar="N",
        help="Size of the scaffold-proportional low-data subsample (default: 10000).",
    )
    p.add_argument(
        "--hp-file", default=None, metavar="JSON",
        help="Optional JSON file whose keys override the default HP dictionary.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")


def _add_train_parser(sub):
    p = sub.add_parser(
        "train",
        help="Train the pipeline on all training data and save the final model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_data_args(p)
    p.add_argument(
        "--model-dir", "-m", required=True, metavar="DIR",
        help="Directory to save the trained model artifacts.",
    )
    p.add_argument(
        "--n-folds", type=int, default=None, metavar="K",
        help="Number of outer CV folds. Overrides HP['n_folds'] when set.",
    )
    p.add_argument(
        "--gate-epochs", type=int, default=None, metavar="N",
        help="Gate training epochs. Overrides HP['gate_epochs'] when set.",
    )
    p.add_argument(
        "--gate-lr", type=float, default=None, metavar="LR",
        help="Gate learning rate. Overrides HP['gate_lr'] when set.",
    )
    p.add_argument(
        "--expert-grid", action="store_true",
        help=(
            "Run the full expert+gate grid search (run_expert_gate_grid_search). "
            "Without this flag, only the gate is searched (pipeline.run)."
        ),
    )
    p.add_argument(
        "--hp-file", default=None, metavar="JSON",
        help="Optional JSON file whose keys override the default HP dictionary.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")


def _add_predict_parser(sub):
    p = sub.add_parser(
        "predict",
        help="Load a saved model and predict pIC50 for new SMILES.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-dir", "-m", required=True, metavar="DIR",
        help="Model directory saved by the 'train' subcommand.",
    )
    p.add_argument(
        "--input", "-i", required=True, metavar="CSV",
        help="Input CSV file with a column of SMILES strings.",
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="CSV",
        help="Output CSV file for predictions.",
    )
    p.add_argument(
        "--smiles-col", default="smiles", metavar="COL",
        help="Name of the SMILES column in the input CSV (default: smiles).",
    )
    p.add_argument(
        "--include-expert-preds", action="store_true",
        help="Add per-expert prediction columns to the output CSV.",
    )
    p.add_argument(
        "--include-weights", action="store_true",
        help="Add gate routing weight columns to the output CSV.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _resolve_hp(args, hp_file_attr="hp_file", seed_attr="seed", n_folds_attr="n_folds",
                gate_epochs_attr=None, gate_lr_attr=None):
    """Merge CLI overrides into a copy of the default HP dict."""
    from .constants import HP as DEFAULT_HP
    hp = {**DEFAULT_HP}

    # Optional JSON file override
    hp_file = getattr(args, hp_file_attr, None)
    if hp_file:
        with open(hp_file) as fh:
            hp.update(json.load(fh))

    # Explicit CLI overrides
    seed = getattr(args, seed_attr, None)
    if seed is not None:
        hp["seed"] = seed

    n_folds = getattr(args, n_folds_attr, None)
    if n_folds is not None:
        hp["n_folds"] = n_folds

    if gate_epochs_attr:
        gate_epochs = getattr(args, gate_epochs_attr, None)
        if gate_epochs is not None:
            hp["gate_epochs"] = gate_epochs

    if gate_lr_attr:
        gate_lr = getattr(args, gate_lr_attr, None)
        if gate_lr is not None:
            hp["gate_lr"] = gate_lr

    return hp


def cmd_production(args):
    from .data import load_processed_data, split_dataset
    from .production import moe_production_pipeline

    verbose = not args.quiet

    if verbose:
        print(f"[production] Loading data from {args.input}")
    df = load_processed_data(args.input)

    seed_for_split = args.seed if args.seed is not None else 100
    if verbose:
        print(f"[production] Splitting data (split={args.split}, train_frac={args.train_frac})")
    train_df, test_df = split_dataset(
        df, split=args.split, train_frac=args.train_frac, seed=seed_for_split
    )
    if verbose:
        print(f"[production] Train: {len(train_df):,}   Test: {len(test_df):,}")

    hp = _resolve_hp(args, n_folds_attr="n_folds")

    out_dir = args.output_dir or os.path.join(
        "runs", f"production_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if verbose:
        print(f"[production] Output directory: {out_dir}")
        n_gate  = (len(hp["l2_regs"]) * len(hp["oracle_regs"])
                   * len(hp["loss_configs"]) * len(hp["routing_schemes"]))
        n_exp   = len(hp["expert_configs"])
        n_total = n_gate * n_exp * hp["n_folds"]
        print(f"[production] Grid: {n_gate} gate × {n_exp} expert configs × "
              f"{hp['n_folds']} folds = {n_total:,} evaluations (×2 data regimes)")

    results = moe_production_pipeline(
        train_df=train_df,
        test_df=test_df,
        path=out_dir,
        HP=hp,
        n_low=args.n_low,
        verbose=verbose,
    )

    if verbose:
        print("\n[production] === Summary ===")
        print(f"  Low-data  test RMSE : {results['low_data_metrics']['moe_rmse']:.4f}  "
              f"R²: {results['low_data_metrics']['moe_r2']:.4f}")
        print(f"  Full-data test RMSE : {results['full_data_metrics']['moe_rmse']:.4f}  "
              f"R²: {results['full_data_metrics']['moe_r2']:.4f}")
        print(f"  Saved to: {out_dir}")


def cmd_train(args):
    from .data import load_processed_data, split_dataset, prepare_features
    from .pipeline import Pipeline

    verbose = not args.quiet

    if verbose:
        print(f"[train] Loading data from {args.input}")
    df = load_processed_data(args.input)

    seed_for_split = args.seed if args.seed is not None else 100
    train_df, test_df = split_dataset(
        df, split=args.split, train_frac=args.train_frac, seed=seed_for_split
    )
    if verbose:
        print(f"[train] Train: {len(train_df):,}   Test: {len(test_df):,}")

    X_train, y_train, smi_train, X_test, y_test, smi_test = prepare_features(
        train_df, test_df
    )

    hp = _resolve_hp(
        args,
        n_folds_attr="n_folds",
        gate_epochs_attr="gate_epochs",
        gate_lr_attr="gate_lr",
    )

    pipeline = Pipeline(
        seed=hp["seed"],
        n_folds=hp["n_folds"],
        gate_hidden_sizes=hp["gate_hidden_sizes"],
        gate_lr=hp["gate_lr"],
        gate_epochs=hp["gate_epochs"],
        gate_oracle_reg=hp["gate_oracle_reg"],
        l2_regs=hp["l2_regs"],
        loss_configs=hp["loss_configs"],
        routing_schemes=hp["routing_schemes"],
        verbose=verbose,
    )

    if args.expert_grid:
        if verbose:
            print("[train] Running full expert+gate grid search...")
        results = pipeline.run_expert_gate_grid_search(
            X_train, y_train, smi_train,
            expert_configs=hp["expert_configs"],
            oracle_regs=hp["oracle_regs"],
        )
    else:
        if verbose:
            print("[train] Running gate-only grid search...")
        results = pipeline.run(X_train, y_train, smi_train)

    final_moe   = results["final_moe"]
    best_config = results["best_config"]

    # Evaluate on held-out test set
    if verbose:
        print("\n[train] Evaluating on test set...")
    from .constants import N_DESC
    top_k = best_config.get("top_k", None)
    X_test_desc   = X_test[:, :N_DESC].astype(np.float64)
    X_test_scaled = np.concatenate(
        [final_moe.scaler.transform(X_test_desc), X_test[:, N_DESC:].astype(np.float64)], axis=1
    )
    test_metrics = final_moe.evaluate(
        X_test_desc, X_test_scaled, smi_test, y_test, top_k=top_k
    )

    print(f"\n[train] Test RMSE : {test_metrics['rmse']:.4f}")
    print(f"[train] Test R²   : {test_metrics['r2']:.4f}")
    print(f"[train] Best config: {best_config}")

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    final_moe.save(args.model_dir)

    # Save best config and test metrics alongside the model
    summary = {
        "best_config":   best_config,
        "val_rmse":      results["mean_rmse"],
        "val_r2":        results["mean_r2"],
        "test_rmse":     test_metrics["rmse"],
        "test_r2":       test_metrics["r2"],
        "per_expert_rmse": dict(
            zip(final_moe.EXPERT_NAMES, test_metrics["per_expert_rmse"])
        ),
        "ensemble_rmse": test_metrics["ensemble_rmse"],
    }
    with open(os.path.join(args.model_dir, "best_config.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print(f"[train] Model saved to: {args.model_dir}")


def cmd_predict(args):
    import pandas as pd
    from .moe import MoE
    from .data import Featurizer
    from .constants import N_DESC

    verbose = not args.quiet

    if verbose:
        print(f"[predict] Loading model from {args.model_dir}")
    moe = MoE.load(args.model_dir)

    if verbose:
        print(f"[predict] Loading compounds from {args.input}")
    input_df = pd.read_csv(args.input)

    if args.smiles_col not in input_df.columns:
        print(
            f"ERROR: column '{args.smiles_col}' not found. "
            f"Available columns: {list(input_df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    smiles_list = input_df[args.smiles_col].tolist()

    if verbose:
        print(f"[predict] Featurizing {len(smiles_list):,} compounds...")
    featurizer = Featurizer()
    X_raw = np.vstack([featurizer.featurize_single(s) for s in smiles_list])
    np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    X_scaled = np.concatenate(
        [moe.scaler.transform(X_raw[:, :N_DESC].astype(np.float64)),
         X_raw[:, N_DESC:].astype(np.float64)], axis=1
    )

    if verbose:
        print("[predict] Running inference...")
    E, U     = moe.expert_predictions(X_scaled, smiles_list)
    X_gate   = np.concatenate([X_raw[:, :N_DESC].astype(np.float64), E, U], axis=1)
    weights  = moe.gate.predict_weights(X_gate)
    moe_pred = (weights * E).sum(axis=1)

    # Build output DataFrame
    out_df = input_df.copy()
    out_df["predicted_pIC50"] = moe_pred

    if args.include_expert_preds:
        for k, name in enumerate(moe.EXPERT_NAMES):
            out_df[f"pred_{name}"] = E[:, k]

    if args.include_weights:
        for k, name in enumerate(moe.EXPERT_NAMES):
            out_df[f"weight_{name}"] = weights[:, k]

    out_df.to_csv(args.output, index=False)

    if verbose:
        print(f"[predict] Predictions saved to: {args.output}  ({len(out_df):,} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "production":
        cmd_production(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
