import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from .constants import N_DESC, K_OOF
from .moe import MoE
from .gate import GatingNetwork


class Pipeline:
    """
    Single scaffold-based train/val split training pipeline.

    The pool (training) data is further split with K_OOF inner folds so that
    experts are trained on K_OOF-1 sections and the gating network is trained
    using predictions on the held-out section (OOF data), preventing leakage
    between expert training and gate training.
    """

    def __init__(
        self,
        seed: int = 42,
        val_frac: float = 0.15,
        val_split: str = "scaffold",
        verbose: bool = True,
        gate_hidden_sizes: list = None,    # default: [100, 80]
        gate_lr: float = 1e-3,
        gate_epochs: int = 300,
        gate_oracle_reg: float = 0.01,
        l2_regs: list = None,              # default: [0.0, 0.001, 0.01]
        loss_configs: list = None,         # default: 6 configs
        routing_schemes: list = None,      # default: [None, 1, 2, 3] top-k options
        # Kept for backward compatibility but no longer used
        n_folds: int = None,
    ):
        self.seed       = seed
        self.val_frac   = val_frac
        self.val_split  = val_split
        self.verbose    = verbose
        self.final_moe  = None

        self.gate_hidden_sizes = gate_hidden_sizes if gate_hidden_sizes is not None else [100, 80]
        self.gate_lr         = gate_lr
        self.gate_epochs     = gate_epochs
        self.gate_oracle_reg = gate_oracle_reg

        self.l2_regs = l2_regs if l2_regs is not None else [0.0, 0.001, 0.01, 0.1]
        self.loss_configs = loss_configs if loss_configs is not None else [
            dict(entropy_reg=0.0,  load_balancing=False),
            dict(entropy_reg=1e-4, load_balancing=False),
            dict(entropy_reg=1e-3, load_balancing=False),
            dict(entropy_reg=1e-2, load_balancing=False),
            dict(entropy_reg=0.1,  load_balancing=False),
            dict(entropy_reg=0.0,  load_balancing=True),
        ]
        self.routing_schemes = routing_schemes if routing_schemes is not None else [
            dict(top_k=None),
            dict(top_k=1),
            dict(top_k=2),
            dict(top_k=3),
        ]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _scaffold_split(self, smiles: list, val_frac: float, seed: int):
        """
        Bemis-Murcko scaffold split. Returns (pool_idx, val_idx) as numpy arrays.
        Scaffolds are shuffled then greedily assigned to pool until (1-val_frac)
        is reached; remaining scaffolds go to val.
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaf_to_idx: dict = {}
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaf = ""
            else:
                scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            scaf_to_idx.setdefault(scaf, []).append(i)

        rng = np.random.default_rng(seed)
        unique = list(scaf_to_idx.keys())
        rng.shuffle(unique)

        n = len(smiles)
        train_frac = 1.0 - val_frac
        pool_idx, val_idx = [], []
        count = 0
        for scaf in unique[:-1]:
            if count / n < train_frac:
                pool_idx.extend(scaf_to_idx[scaf])
                count += len(scaf_to_idx[scaf])
            else:
                val_idx.extend(scaf_to_idx[scaf])
        val_idx.extend(scaf_to_idx[unique[-1]])

        return np.array(pool_idx), np.array(val_idx)

    def _run_inner_oof(
        self,
        X_pool_scaled: np.ndarray,
        y_pool: np.ndarray,
        smiles_pool: list,
        seed: int,
        expert_config: dict = None,
    ):
        """K_OOF-fold inner OOF on pool data."""
        n = len(y_pool)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        E_oof = np.zeros((n, len(MoE.EXPERT_NAMES)))
        U_oof = np.zeros((n, len(MoE.EXPERT_NAMES)))

        for k in range(K_OOF):
            val_start = round(n * k / K_OOF)
            val_end   = round(n * (k + 1) / K_OOF)
            iv_idx = perm[val_start:val_end]
            it_idx = np.concatenate([perm[:val_start], perm[val_end:]])

            X_it   = X_pool_scaled[it_idx]
            y_it   = y_pool[it_idx]
            smi_it = [smiles_pool[i] for i in it_idx]
            smi_iv = [smiles_pool[i] for i in iv_idx]

            self._log(f"    [OOF {k+1}/{K_OOF}] training experts ({len(it_idx):,} samples)...")
            experts_k = (
                MoE._make_experts_from_config(expert_config, seed=seed + k * 97 + 13)
                if expert_config is not None
                else MoE._make_experts(seed=seed + k * 97 + 13)
            )
            for name, exp in zip(MoE.EXPERT_NAMES, experts_k):
                self._log(f"      {name}...")
                MoE._fit_one(exp, X_it, y_it, smi_it)

            for j, exp in enumerate(experts_k):
                E_oof[iv_idx, j] = MoE._pred_one(exp, X_pool_scaled[iv_idx], smi_iv)
                U_oof[iv_idx, j] = MoE._uncertainty_one(exp, X_pool_scaled[iv_idx], smi_iv)

        return E_oof, U_oof

    def _eval_val(
        self,
        gate: GatingNetwork,
        X_val_gate: np.ndarray,
        E_val: np.ndarray,
        y_val: np.ndarray,
        top_k: int = None,
    ):
        """Return (rmse, r2) for the validation set given a trained gate."""
        w     = gate.predict_weights(X_val_gate, top_k=top_k)
        preds = (w * E_val).sum(axis=1)
        rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
        r2    = float(r2_score(y_val, preds))
        return rmse, r2

    def run(self, X_raw: np.ndarray, y: np.ndarray, smiles: list):
        """
        Single scaffold-based train/val split pipeline with gate hyperparameter search.
        The final MoE is trained on the full dataset using the best gate config found.
        """
        np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # Single scaffold split
        pool_idx, val_idx = self._scaffold_split(smiles, self.val_frac, self.seed)
        self._log(
            f"[Split] pool={len(pool_idx):,}  val={len(val_idx):,}  "
            f"(scaffold split, val_frac={self.val_frac})"
        )

        X_pool_raw = X_raw[pool_idx]
        y_pool     = y[pool_idx]
        smi_pool   = [smiles[i] for i in pool_idx]

        X_val_raw = X_raw[val_idx]
        y_val     = y[val_idx]
        smi_val   = [smiles[i] for i in val_idx]

        # Scaler fitted on pool descriptors only
        scaler    = StandardScaler()
        desc_sc   = scaler.fit_transform(X_pool_raw[:, :N_DESC].astype(np.float64))
        X_pool_scaled = np.concatenate([desc_sc, X_pool_raw[:, N_DESC:]], axis=1)

        desc_val     = scaler.transform(X_val_raw[:, :N_DESC].astype(np.float64))
        X_val_scaled = np.concatenate([desc_val, X_val_raw[:, N_DESC:]], axis=1)

        # Inner OOF on pool -> honest expert predictions for gate training
        self._log("  Running inner OOF on pool...")
        E_oof, U_oof = self._run_inner_oof(X_pool_scaled, y_pool, smi_pool, seed=self.seed)

        # Pool experts predict on val set (for gate validation)
        self._log("  Training pool experts (for val prediction)...")
        pool_moe        = MoE(seed=self.seed + 7)
        pool_moe.scaler = scaler
        pool_moe.train_experts(X_pool_scaled, y_pool, smi_pool)
        E_val, U_val = pool_moe.expert_predictions(X_val_scaled, smi_val)

        X_gate_pool = np.concatenate([X_pool_raw[:, :N_DESC], E_oof,  U_oof],  axis=1)
        X_gate_val  = np.concatenate([X_val_raw[:, :N_DESC],  E_val,  U_val],  axis=1)
        self._log(f"  Gate input shape: {X_gate_pool.shape} (pool), {X_gate_val.shape} (val)")

        # Gate hyperparameter search
        hyperparam_grid = [
            dict(weight_decay=l2, **loss, **routing)
            for l2, loss, routing in itertools.product(
                self.l2_regs, self.loss_configs, self.routing_schemes
            )
        ]
        self._log(f"  Searching {len(hyperparam_grid)} gate configs...")
        best_rmse   = float("inf")
        best_r2     = float("-inf")
        best_config = None

        for cfg in hyperparam_grid:
            gate = GatingNetwork(
                n_features=X_gate_pool.shape[1],
                n_experts=len(MoE.EXPERT_NAMES),
                hidden_sizes=self.gate_hidden_sizes,
                lr=self.gate_lr,
                epochs=self.gate_epochs,
                weight_decay=cfg["weight_decay"],
                entropy_reg=cfg["entropy_reg"],
                load_balancing=cfg["load_balancing"],
                oracle_reg=self.gate_oracle_reg,
            )
            gate.fit(X_gate_pool, E_oof, y_pool)
            rmse, r2 = self._eval_val(gate, X_gate_val, E_val, y_val, top_k=cfg["top_k"])
            if rmse < best_rmse:
                best_rmse   = rmse
                best_r2     = r2
                best_config = cfg

        self._log(f"  Best val RMSE={best_rmse:.4f}  R²={best_r2:.4f}")
        self._log(f"  Best config: {best_config}")

        # Train final MoE on full dataset
        self._log("\n[Final MoE] Training on full dataset...")

        full_scaler  = StandardScaler()
        desc_all     = full_scaler.fit_transform(X_raw[:, :N_DESC].astype(np.float64))
        X_all_scaled = np.concatenate([desc_all, X_raw[:, N_DESC:]], axis=1)

        self._log("  Running OOF on full data...")
        E_oof_full, U_oof_full = self._run_inner_oof(
            X_all_scaled, y, smiles, seed=self.seed + 9999
        )

        self._log("  Training final experts on full data...")
        self.final_moe        = MoE(seed=self.seed)
        self.final_moe.scaler = full_scaler
        self.final_moe.train_experts(X_all_scaled, y, smiles)

        _, U_full   = self.final_moe.expert_predictions(X_all_scaled, smiles)
        X_gate_full = np.concatenate([X_raw[:, :N_DESC], E_oof_full, U_full], axis=1)

        self._log("  Training final gate...")
        self.final_moe.gate = GatingNetwork(
            n_features=X_gate_full.shape[1],
            n_experts=len(MoE.EXPERT_NAMES),
            hidden_sizes=self.gate_hidden_sizes,
            lr=self.gate_lr,
            epochs=self.gate_epochs,
            weight_decay=best_config.get("weight_decay", 0.0),
            entropy_reg=best_config.get("entropy_reg", 0.0),
            load_balancing=best_config.get("load_balancing", False),
            oracle_reg=self.gate_oracle_reg,
        )
        self.final_moe.gate.fit(X_gate_full, E_oof_full, y)
        self._log("[Final MoE] Done.")

        return dict(
            mean_rmse=float(best_rmse),
            std_rmse=0.0,
            mean_r2=float(best_r2),
            std_r2=0.0,
            fold_results=[dict(fold=1, val_rmse=best_rmse, val_r2=best_r2, best_config=best_config)],
            best_config=best_config,
            final_moe=self.final_moe,
        )

    def run_expert_gate_grid_search(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        smiles: list,
        expert_configs: list = None,
        oracle_regs: list = None,
    ) -> dict:
        """
        Joint grid search over expert hyperparameters and gate hyperparameters,
        evaluated on a single scaffold-based train/val split.

        Expert config grid (default — 4 configs, corners of tabular-tuned x GNN-tuned):
            A — baseline  : n_estimators=200, lr=0.05, lgbm_lambda=0.0, rf_feat=1.0,   dropout=0.0, gin_layers=4
            B — tabular   : n_estimators=500, lr=0.02, lgbm_lambda=1.0, rf_feat="sqrt", dropout=0.0, gin_layers=4
            C — GNN       : n_estimators=200, lr=0.05, lgbm_lambda=0.0, rf_feat=1.0,   dropout=0.2, gin_layers=3
            D — all tuned : n_estimators=500, lr=0.02, lgbm_lambda=1.0, rf_feat="sqrt", dropout=0.2, gin_layers=3

        Gate grid = self.l2_regs x oracle_regs x self.loss_configs x self.routing_schemes.

        Parameters
        ----------
        X_raw          : (n, n_features) raw feature matrix
        y              : (n,) targets
        smiles         : list of SMILES strings
        expert_configs : list of expert config dicts (see MoE._make_experts_from_config).
                         Defaults to the 4-config A/B/C/D grid.
        oracle_regs    : oracle supervision coefficients to search. Default [0.0, 0.01, 0.1].

        Returns
        -------
        Same dict format as run():
            mean_rmse, std_rmse, mean_r2, std_r2 — for the best (expert, gate) combo
            fold_results  — single-entry list with val metrics for the best combo
            best_config   — merged dict of best expert + gate hyperparameters
            final_moe     — MoE trained on the full dataset with best config
        """
        if expert_configs is None:
            expert_configs = []
        if oracle_regs is None:
            oracle_regs = [0.0, 0.01, 0.1]

        np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # Single scaffold split
        pool_idx, val_idx = self._scaffold_split(smiles, self.val_frac, self.seed)
        self._log(
            f"[Split] pool={len(pool_idx):,}  val={len(val_idx):,}  "
            f"(scaffold split, val_frac={self.val_frac})"
        )

        X_pool_raw = X_raw[pool_idx]
        y_pool     = y[pool_idx]
        smi_pool   = [smiles[i] for i in pool_idx]
        X_val_raw  = X_raw[val_idx]
        y_val      = y[val_idx]
        smi_val    = [smiles[i] for i in val_idx]

        # Scaler fitted once on pool; shared across all expert configs
        scaler        = StandardScaler()
        desc_sc       = scaler.fit_transform(X_pool_raw[:, :N_DESC].astype(np.float64))
        X_pool_scaled = np.concatenate([desc_sc, X_pool_raw[:, N_DESC:]], axis=1)
        desc_val      = scaler.transform(X_val_raw[:, :N_DESC].astype(np.float64))
        X_val_scaled  = np.concatenate([desc_val, X_val_raw[:, N_DESC:]], axis=1)

        # Gate config grid — oracle_reg is searched
        gate_grid = [
            dict(weight_decay=l2, oracle_reg=oreg, **loss, **routing)
            for l2, oreg, loss, routing in itertools.product(
                self.l2_regs, oracle_regs, self.loss_configs, self.routing_schemes
            )
        ]

        n_exp  = len(expert_configs)
        n_gate = len(gate_grid)
        self._log(
            f"[run_expert_gate_grid_search] "
            f"{n_exp} expert configs x {n_gate} gate configs = {n_exp * n_gate:,} gate evals"
        )

        # rmse_matrix[ecfg_idx, gcfg_idx]
        rmse_matrix = np.full((n_exp, n_gate), np.inf)
        r2_matrix   = np.full((n_exp, n_gate), -np.inf)

        for ecfg_idx, ecfg in enumerate(expert_configs):
            self._log(f"\n  [Expert config {ecfg_idx+1}/{n_exp}] {ecfg}")

            # Inner OOF -> honest expert predictions for gate training
            self._log("    Running inner OOF...")
            oof_seed = self.seed + ecfg_idx * 100
            E_oof, U_oof = self._run_inner_oof(
                X_pool_scaled, y_pool, smi_pool, seed=oof_seed, expert_config=ecfg
            )

            # Pool experts predict on val set
            self._log("    Training pool experts (for val prediction)...")
            pool_moe        = MoE(seed=self.seed + ecfg_idx * 50 + 7)
            pool_moe.scaler = scaler
            pool_moe.train_experts_from_config(ecfg, X_pool_scaled, y_pool, smi_pool)
            E_val, U_val = pool_moe.expert_predictions(X_val_scaled, smi_val)

            X_gate_pool = np.concatenate([X_pool_raw[:, :N_DESC], E_oof, U_oof], axis=1)
            X_gate_val  = np.concatenate([X_val_raw[:, :N_DESC],  E_val, U_val], axis=1)

            # Sweep all gate configs
            self._log(f"    Evaluating {n_gate} gate configs...")
            for gcfg_idx, gcfg in enumerate(gate_grid):
                gate = GatingNetwork(
                    n_features=X_gate_pool.shape[1],
                    n_experts=len(MoE.EXPERT_NAMES),
                    hidden_sizes=self.gate_hidden_sizes,
                    lr=self.gate_lr,
                    epochs=self.gate_epochs,
                    weight_decay=gcfg["weight_decay"],
                    entropy_reg=gcfg["entropy_reg"],
                    load_balancing=gcfg["load_balancing"],
                    oracle_reg=gcfg["oracle_reg"],
                )
                gate.fit(X_gate_pool, E_oof, y_pool)
                rmse, r2 = self._eval_val(
                    gate, X_gate_val, E_val, y_val, top_k=gcfg.get("top_k")
                )
                rmse_matrix[ecfg_idx, gcfg_idx] = rmse
                r2_matrix[ecfg_idx,   gcfg_idx] = r2

        # Select best (expert_config, gate_config) by val RMSE
        best_flat = int(np.argmin(rmse_matrix))
        best_ecfg_idx, best_gcfg_idx = np.unravel_index(best_flat, rmse_matrix.shape)

        best_expert_config = expert_configs[best_ecfg_idx]
        best_gate_config   = gate_grid[best_gcfg_idx]
        best_combined      = {**best_expert_config, **best_gate_config}

        best_rmse = float(rmse_matrix[best_ecfg_idx, best_gcfg_idx])
        best_r2   = float(r2_matrix[best_ecfg_idx,   best_gcfg_idx])

        self._log("\n[Grid Search Summary]")
        self._log(f"  Best val RMSE : {best_rmse:.4f}")
        self._log(f"  Best val R²   : {best_r2:.4f}")
        self._log(f"  Best expert config (idx {best_ecfg_idx}): {best_expert_config}")
        self._log(f"  Best gate config   (idx {best_gcfg_idx}): {best_gate_config}")

        # Train final MoE on full dataset with best config
        self._log("\n[Final MoE] Training on full dataset...")

        full_scaler  = StandardScaler()
        desc_all     = full_scaler.fit_transform(X_raw[:, :N_DESC].astype(np.float64))
        X_all_scaled = np.concatenate([desc_all, X_raw[:, N_DESC:]], axis=1)

        self._log("  Running OOF on full data...")
        E_oof_full, U_oof_full = self._run_inner_oof(
            X_all_scaled, y, smiles, seed=self.seed + 9999, expert_config=best_expert_config
        )

        self._log("  Training final experts on full data...")
        self.final_moe        = MoE(seed=self.seed)
        self.final_moe.scaler = full_scaler
        self.final_moe.train_experts_from_config(best_expert_config, X_all_scaled, y, smiles)

        _, U_full   = self.final_moe.expert_predictions(X_all_scaled, smiles)
        X_gate_full = np.concatenate([X_raw[:, :N_DESC], E_oof_full, U_full], axis=1)

        self._log("  Training final gate...")
        self.final_moe.gate = GatingNetwork(
            n_features=X_gate_full.shape[1],
            n_experts=len(MoE.EXPERT_NAMES),
            hidden_sizes=self.gate_hidden_sizes,
            lr=self.gate_lr,
            epochs=self.gate_epochs,
            weight_decay=best_gate_config.get("weight_decay", 0.0),
            entropy_reg=best_gate_config.get("entropy_reg", 0.0),
            load_balancing=best_gate_config.get("load_balancing", False),
            oracle_reg=best_gate_config.get("oracle_reg", self.gate_oracle_reg),
        )
        self.final_moe.gate.fit(X_gate_full, E_oof_full, y)
        self._log("[Final MoE] Done.")

        return dict(
            mean_rmse=best_rmse,
            std_rmse=0.0,
            mean_r2=best_r2,
            std_r2=0.0,
            fold_results=[dict(fold=1, val_rmse=best_rmse, val_r2=best_r2, best_config=best_combined)],
            best_config=best_combined,
            final_moe=self.final_moe,
        )
