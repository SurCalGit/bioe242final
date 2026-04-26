import os
import json
import numpy as np
import joblib
import torch
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

from .constants import N_DESC
from .experts import ChempropDMPNN, GIN
from .gate import GatingNetwork


class MoE:
    """
    Mixture-of-Experts model combining 5 heterogeneous experts via a learned gating network.

    Experts: XGBoost, RandomForest, LightGBM, ChempropDMPNN, GIN
    Gate:    MLP router trained with OOF expert predictions (no leakage).
    """

    EXPERT_NAMES = ["xgb", "rf", "lgbm", "chemprop", "gin"]

    def __init__(self, seed: int = 42):
        self.seed    = seed
        self.experts = None
        self.gate    = None
        self.scaler  = None    # StandardScaler for descriptors; set externally or by _scale()

    # ------------------------------------------------------------------
    # Expert factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def _make_experts(seed: int = 42):
        """Return 5 fresh, unfitted expert instances with default hyperparameters."""
        seeds = [seed, seed + 1, seed + 2, seed + 3, seed + 4]
        return [
            XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0,
                random_state=seeds[0],
            ),
            RandomForestRegressor(
                n_estimators=200, max_depth=None, min_samples_leaf=3,
                n_jobs=-1, random_state=seeds[1],
            ),
            LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                n_jobs=-1, verbose=-1, random_state=seeds[2],
            ),
            ChempropDMPNN(epochs=50, hidden_size=300, depth=3, seed=seeds[3]),
            GIN(hidden_channels=256, num_layers=4, dropout=0.3,
                weight_decay=1e-4, epochs=100, seed=seeds[4]),
        ]

    @staticmethod
    def _make_experts_from_config(config: dict, seed: int = 42):
        """
        Return 5 fresh, unfitted experts using the given config dict.

        Config keys (all optional; missing keys fall back to _make_experts defaults):
            n_estimators    : XGBoost + LightGBM tree count       (default 200)
            learning_rate   : XGBoost + LightGBM learning rate    (default 0.05)
            lgbm_reg_lambda : LightGBM L2 regularisation          (default 0.0)
            rf_max_features : RandomForest max_features            (default 1.0)
            dmpnn_dropout   : ChempropDMPNN FFN dropout rate      (default 0.0)
            gin_num_layers  : GIN number of GINConv layers         (default 4)
        """
        n_est = config.get("n_estimators", 200)
        lr    = config.get("learning_rate", 0.05)
        seeds = [seed, seed + 1, seed + 2, seed + 3, seed + 4]
        return [
            XGBRegressor(
                n_estimators=n_est, learning_rate=lr, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0,
                random_state=seeds[0],
            ),
            RandomForestRegressor(
                n_estimators=200, max_depth=None, min_samples_leaf=3,
                max_features=config.get("rf_max_features", 1.0),
                n_jobs=-1, random_state=seeds[1],
            ),
            LGBMRegressor(
                n_estimators=n_est, learning_rate=lr, max_depth=6,
                reg_lambda=config.get("lgbm_reg_lambda", 0.0),
                n_jobs=-1, verbose=-1, random_state=seeds[2],
            ),
            ChempropDMPNN(
                epochs=50, hidden_size=300, depth=3,
                dropout=config.get("dmpnn_dropout", 0.0), seed=seeds[3],
            ),
            GIN(
                hidden_channels=256,
                num_layers=config.get("gin_num_layers", 4),
                dropout=0.3, weight_decay=1e-4, epochs=100, seed=seeds[4],
            ),
        ]

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_experts(self, X_scaled: np.ndarray, y: np.ndarray, smiles: list):
        """Train all 5 experts with default hyperparameters."""
        self.experts = self._make_experts(self.seed)
        for name, expert in zip(self.EXPERT_NAMES, self.experts):
            print(f"  [MoE] Training {name}...", end=" ", flush=True)
            self._fit_one(expert, X_scaled, y, smiles)
            print("done")

    def train_experts_from_config(
        self, config: dict, X_scaled: np.ndarray, y: np.ndarray, smiles: list
    ) -> None:
        """Create and train 5 experts using the given expert config dict."""
        self.experts = self._make_experts_from_config(config, seed=self.seed)
        for name, expert in zip(self.EXPERT_NAMES, self.experts):
            print(f"  [MoE] Training {name}...", end=" ", flush=True)
            self._fit_one(expert, X_scaled, y, smiles)
            print("done")

    @staticmethod
    def _fit_one(
        expert, X_scaled: np.ndarray, y: np.ndarray, smiles: list, sample_weight=None
    ) -> None:
        if getattr(expert, "is_gnn", False):
            expert.fit(smiles, y, sample_weight=sample_weight)
        else:
            if sample_weight is not None:
                try:
                    expert.fit(X_scaled, y, sample_weight=sample_weight)
                except TypeError:
                    expert.fit(X_scaled, y)
            else:
                expert.fit(X_scaled, y)

    @staticmethod
    def _pred_one(expert, X_scaled: np.ndarray, smiles: list) -> np.ndarray:
        if getattr(expert, "is_gnn", False):
            return expert.predict(smiles)
        return expert.predict(X_scaled)

    @staticmethod
    def _uncertainty_one(expert, X_scaled: np.ndarray, smiles: list) -> np.ndarray:
        if getattr(expert, "is_gnn", False):
            return expert.predict_uncertainty(smiles).astype(np.float64)

        if isinstance(expert, RandomForestRegressor):
            tree_preds = np.column_stack([t.predict(X_scaled) for t in expert.estimators_])
            return tree_preds.std(axis=1)

        if isinstance(expert, XGBRegressor):
            n_trees   = expert.n_estimators
            pred_full = expert.predict(X_scaled)
            dmat      = xgb.DMatrix(X_scaled)
            pred_half = expert.get_booster().predict(
                dmat, iteration_range=(0, max(1, n_trees // 2))
            )
            return np.abs(pred_full - pred_half)

        if isinstance(expert, LGBMRegressor):
            n_iter    = expert.n_estimators
            pred_full = expert.predict(X_scaled)
            pred_half = expert.predict(X_scaled, num_iteration=max(1, n_iter // 2))
            return np.abs(pred_full - pred_half)

        return np.zeros(len(X_scaled), dtype=np.float64)

    def _scale(self, X_raw: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply StandardScaler to the descriptor columns only."""
        if fit:
            self.scaler = StandardScaler()
            desc_sc = self.scaler.fit_transform(X_raw[:, :N_DESC])
        else:
            desc_sc = self.scaler.transform(X_raw[:, :N_DESC])
        return np.concatenate([desc_sc, X_raw[:, N_DESC:]], axis=1)

    # ------------------------------------------------------------------
    # Expert predictions
    # ------------------------------------------------------------------

    def expert_predictions(self, X_scaled: np.ndarray, smiles: list):
        """Return (E, U) arrays of shape (n, 5): predictions and uncertainties."""
        if self.experts is None:
            raise RuntimeError("Call train_experts() first.")
        preds = [self._pred_one(e, X_scaled, smiles) for e in self.experts]
        uncs  = [self._uncertainty_one(e, X_scaled, smiles) for e in self.experts]
        return np.column_stack(preds), np.column_stack(uncs)

    # ------------------------------------------------------------------
    # Gate training
    # ------------------------------------------------------------------

    def train_gating_network(
        self,
        X_raw_desc:      np.ndarray,
        E:               np.ndarray,
        U:               np.ndarray,
        y:               np.ndarray,
        config:          dict  = None,
        gate_hidden_sizes: list  = None,
        gate_lr:         float = 1e-3,
        gate_epochs:     int   = 300,
        gate_oracle_reg: float = 0.01,
    ) -> None:
        """Train the gate on [raw_desc | expert_preds | uncertainties]."""
        if config is None:
            config = dict(weight_decay=0.0, entropy_reg=0.0, load_balancing=False, top_k=None)

        hidden_sizes = gate_hidden_sizes if gate_hidden_sizes is not None else [100, 80]
        X_gate = np.concatenate([X_raw_desc, E, U], axis=1)

        self.gate = GatingNetwork(
            n_features=X_gate.shape[1],
            n_experts=len(self.EXPERT_NAMES),
            hidden_sizes=hidden_sizes,
            lr=gate_lr,
            epochs=gate_epochs,
            weight_decay=config.get("weight_decay", 0.0),
            entropy_reg=config.get("entropy_reg", 0.0),
            load_balancing=config.get("load_balancing", False),
            oracle_reg=gate_oracle_reg,
        )
        self.gate.fit(X_gate, E, y)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_raw_desc: np.ndarray,
        X_scaled:   np.ndarray,
        smiles:     list,
        y_true:     np.ndarray,
        top_k:      int = None,
    ) -> dict:
        """
        Compute MoE predictions and return RMSE, R², per-expert RMSE, and ensemble RMSE.
        """
        if self.experts is None or self.gate is None:
            raise RuntimeError("Train experts and gate before evaluating.")

        E, U      = self.expert_predictions(X_scaled, smiles)
        X_gate    = np.concatenate([X_raw_desc, E, U], axis=1)
        weights   = self.gate.predict_weights(X_gate, top_k=top_k)
        moe_preds = (weights * E).sum(axis=1)

        per_expert_rmse = [
            float(np.sqrt(mean_squared_error(y_true, E[:, k])))
            for k in range(E.shape[1])
        ]
        ensemble_preds = E.mean(axis=1)

        return dict(
            rmse=float(np.sqrt(mean_squared_error(y_true, moe_preds))),
            r2=float(r2_score(y_true, moe_preds)),
            per_expert_rmse=per_expert_rmse,
            ensemble_rmse=float(np.sqrt(mean_squared_error(y_true, ensemble_preds))),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str) -> None:
        """
        Save all experts, gate, descriptor scaler, and metadata to model_dir.

        Layout
        ------
        model_dir/
          experts/xgb.joblib
          experts/rf.joblib
          experts/lgbm.joblib
          experts/chemprop.pt
          experts/gin.pt
          gate_net.pt
          gate_scaler.joblib
          desc_scaler.joblib
          config.json
        """
        if self.experts is None or self.gate is None:
            raise RuntimeError("Train MoE before saving.")

        experts_dir = os.path.join(model_dir, "experts")
        os.makedirs(experts_dir, exist_ok=True)

        # Tabular experts
        for name, expert in zip(self.EXPERT_NAMES[:3], self.experts[:3]):
            joblib.dump(expert, os.path.join(experts_dir, f"{name}.joblib"))

        # GNN experts — save state dicts
        chemprop_exp = self.experts[3]
        torch.save(
            chemprop_exp._model_state,
            os.path.join(experts_dir, "chemprop.pt"),
        )
        gin_exp = self.experts[4]
        torch.save(
            gin_exp._model.state_dict(),
            os.path.join(experts_dir, "gin.pt"),
        )
        # Persist GNN configs alongside weights so load() can reconstruct them
        joblib.dump(
            {
                "chemprop": {k: getattr(chemprop_exp, k) for k in
                             ["epochs", "batch_size", "hidden_size", "depth",
                              "ffn_hidden", "ffn_layers", "lr", "dropout", "seed"]},
                "gin": {k: getattr(gin_exp, k) for k in
                        ["hidden_channels", "num_layers", "dropout", "epochs",
                         "lr", "weight_decay", "batch_size", "seed"]},
            },
            os.path.join(experts_dir, "gnn_configs.joblib"),
        )

        # Descriptor scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(model_dir, "desc_scaler.joblib"))

        # Gate
        self.gate.save(model_dir)

        # Config JSON
        cfg = {
            "seed":         self.seed,
            "n_desc":       N_DESC,
            "expert_names": self.EXPERT_NAMES,
            "gate_hidden_sizes": self.gate.hidden_sizes,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "MoE":
        """Reconstruct a fully trained MoE from a directory saved by save()."""
        with open(os.path.join(model_dir, "config.json")) as fh:
            cfg = json.load(fh)

        moe = cls(seed=cfg["seed"])

        experts_dir = os.path.join(model_dir, "experts")
        gnn_cfgs    = joblib.load(os.path.join(experts_dir, "gnn_configs.joblib"))

        # Tabular experts
        tabular = [
            joblib.load(os.path.join(experts_dir, f"{n}.joblib"))
            for n in ["xgb", "rf", "lgbm"]
        ]

        # ChempropDMPNN
        cc = gnn_cfgs["chemprop"]
        chemprop_exp = ChempropDMPNN(**cc)
        # Rebuild model architecture and load weights
        from .experts import ChempropDMPNN as _CD
        import chemprop.models as cmodels
        from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
        mp  = BondMessagePassing(d_h=cc["hidden_size"], depth=cc["depth"])
        agg = MeanAggregation()
        ffn = RegressionFFN(
            n_tasks=1, input_dim=cc["hidden_size"],
            hidden_dim=cc["ffn_hidden"], n_layers=cc["ffn_layers"], dropout=cc["dropout"],
        )
        chemprop_model = cmodels.MPNN(
            message_passing=mp, agg=agg, predictor=ffn,
            init_lr=cc["lr"], max_lr=cc["lr"] * 10, final_lr=cc["lr"],
        )
        state = torch.load(
            os.path.join(experts_dir, "chemprop.pt"), map_location="cpu", weights_only=True
        )
        chemprop_model.load_state_dict(state)
        chemprop_model.eval()
        chemprop_exp._model       = chemprop_model
        chemprop_exp._model_state = state

        # GIN
        gc = gnn_cfgs["gin"]
        gin_exp = GIN(**gc)
        gin_exp._device = torch.device("cpu")
        from .experts import _GINNet
        gin_net = _GINNet.build(GIN.ATOM_FDIM, gc["hidden_channels"], gc["num_layers"], gc["dropout"])
        gin_net.load_state_dict(
            torch.load(os.path.join(experts_dir, "gin.pt"), map_location="cpu", weights_only=True)
        )
        gin_net.eval()
        gin_exp._model = gin_net

        moe.experts = [*tabular, chemprop_exp, gin_exp]

        # Descriptor scaler
        desc_scaler_path = os.path.join(model_dir, "desc_scaler.joblib")
        if os.path.exists(desc_scaler_path):
            moe.scaler = joblib.load(desc_scaler_path)

        # Gate
        gate = GatingNetwork(
            n_features=N_DESC + len(cls.EXPERT_NAMES) * 2,
            n_experts=len(cls.EXPERT_NAMES),
            hidden_sizes=cfg["gate_hidden_sizes"],
        )
        gate.load(model_dir)
        moe.gate = gate

        return moe
