import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class GatingNetwork:
    """
    MLP gating network for soft or sparse expert routing.

    Input: [raw descriptors (200) | expert preds (5) | expert uncertainties (5)] = 210-D
    Output: per-expert routing weights via softmax.

    Loss:  L_MSE
         + oracle_reg  * L_oracle      (cross-entropy toward per-sample best expert)
         - entropy_reg * H(weights)    (entropy bonus for diverse routing)
         + load_balancing penalty      (Switch-Transformer style)
    """

    def __init__(
        self,
        n_features:     int,
        n_experts:      int,
        hidden_sizes:   list  = None,
        lr:             float = 1e-3,
        epochs:         int   = 300,
        weight_decay:   float = 0.0,
        entropy_reg:    float = 0.0,
        load_balancing: bool  = False,
        oracle_reg:     float = 0.01,
        temperature:    float = 1.0,
    ):
        self.n_features     = n_features
        self.n_experts      = n_experts
        self.hidden_sizes   = list(hidden_sizes) if hidden_sizes is not None else [100, 80]
        self.lr             = lr
        self.epochs         = epochs
        self.weight_decay   = weight_decay
        self.entropy_reg    = entropy_reg
        self.load_balancing = load_balancing
        self.oracle_reg     = oracle_reg
        self.temperature    = max(temperature, 1e-6)
        self._scaler: StandardScaler = None
        self._net = None

    def _build(self) -> nn.Module:
        layers  = []
        in_size = self.n_features
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, self.n_experts))
        return nn.Sequential(*layers)

    def _train(self, X_scaled: np.ndarray, expert_preds: np.ndarray, y: np.ndarray):
        X_t = torch.tensor(X_scaled,     dtype=torch.float32)
        E_t = torch.tensor(expert_preds, dtype=torch.float32)
        y_t = torch.tensor(y,            dtype=torch.float32)

        if self._net is None:
            self._net = self._build()

        K         = self.n_experts
        T         = self.temperature
        optimizer = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits  = self._net(X_t) / T
            weights = torch.softmax(logits, dim=1)
            preds   = (weights * E_t).sum(dim=1)
            loss    = nn.functional.mse_loss(preds, y_t)

            if self.entropy_reg > 0.0:
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
                loss    = loss - self.entropy_reg * entropy

            if self.load_balancing:
                mean_w    = weights.mean(dim=0)
                load_loss = 0.01 * K * (mean_w ** 2).sum()
                loss      = loss + load_loss

            if self.oracle_reg > 0.0:
                with torch.no_grad():
                    errs     = (E_t - y_t.unsqueeze(1)) ** 2
                    oracle_w = torch.softmax(-errs, dim=1)
                oracle_loss = -(oracle_w * torch.log(weights + 1e-8)).sum(dim=1).mean()
                loss        = loss + self.oracle_reg * oracle_loss

            loss.backward()
            optimizer.step()

        self._net.eval()

    def fit(self, X_raw: np.ndarray, expert_preds: np.ndarray, y: np.ndarray):
        """Fit internal scaler then train the gate MLP from scratch."""
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X_raw)
        self._net    = None
        self._train(X_scaled, expert_preds, y)

    def predict_weights(self, X_raw: np.ndarray, top_k: int = None) -> np.ndarray:
        """Return (n, n_experts) routing weights, optionally sparsified to top-k."""
        X_scaled = self._scaler.transform(X_raw)
        X_t      = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits  = self._net(X_t) / self.temperature
            weights = torch.softmax(logits, dim=1).numpy()

        if top_k is not None:
            mask     = np.zeros_like(weights)
            topk_idx = np.argpartition(weights, -top_k, axis=1)[:, -top_k:]
            np.put_along_axis(mask, topk_idx, 1.0, axis=1)
            weights  = weights * mask
            row_sums = weights.sum(axis=1, keepdims=True)
            weights  = weights / np.maximum(row_sums, 1e-8)

        return weights

    def save(self, model_dir: str):
        """Save gate MLP weights and internal scaler to model_dir."""
        import joblib
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self._net.state_dict(), os.path.join(model_dir, "gate_net.pt"))
        joblib.dump(self._scaler, os.path.join(model_dir, "gate_scaler.joblib"))

    def load(self, model_dir: str):
        """Load gate MLP and scaler from model_dir (in-place)."""
        import joblib
        self._scaler = joblib.load(os.path.join(model_dir, "gate_scaler.joblib"))
        self._net    = self._build()
        state        = torch.load(
            os.path.join(model_dir, "gate_net.pt"), map_location="cpu", weights_only=True
        )
        self._net.load_state_dict(state)
        self._net.eval()
        return self
