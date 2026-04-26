"""
GNN expert models and shared atom feature functions.

Chemprop (v2) and PyTorch Geometric imports are deferred to method bodies
so the package can be imported even when those dependencies are unavailable.
"""

import math
import numpy as np
import tempfile

from rdkit import Chem
from rdkit.Chem import rdchem


# ---------------------------------------------------------------------------
# Atom feature function (shared by GIN via _smiles_to_pyg)
# ---------------------------------------------------------------------------

def _atom_features(atom) -> list:
    ATOM_TYPES   = [6, 7, 8, 16, 9, 17, 35, 53, 15]
    DEGREE_MAX   = 6
    NUM_HS_MAX   = 4
    HYBRID_TYPES = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]

    def _one_hot(val, lst):
        return [int(val == x) for x in lst] + [int(val not in lst)]

    anum = atom.GetAtomicNum()
    deg  = min(atom.GetDegree(), DEGREE_MAX)
    nhs  = min(atom.GetTotalNumHs(), NUM_HS_MAX)
    fc   = max(-3, min(3, atom.GetFormalCharge())) / 3.0
    arom = int(atom.GetIsAromatic())
    ring = int(atom.IsInRing())
    hyb  = atom.GetHybridization()

    return (
        _one_hot(anum, ATOM_TYPES)
        + _one_hot(deg,  list(range(DEGREE_MAX + 1)))
        + _one_hot(nhs,  list(range(NUM_HS_MAX + 1)))
        + [fc, arom, ring]
        + _one_hot(hyb, HYBRID_TYPES)
    )


def _smiles_to_pyg(smiles_list, y_list=None):
    """Convert SMILES to PyG Data objects. Invalid SMILES become single-node graphs."""
    import torch
    from torch_geometric.data import Data

    N_FEAT = 33
    results = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            x          = torch.zeros((1, N_FEAT), dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            x = torch.tensor(
                [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
            )
            edges = []
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges += [[u, v], [v, u]]
            edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
                if edges else torch.zeros((2, 0), dtype=torch.long)
            )
        d = Data(x=x, edge_index=edge_index)
        if y_list is not None:
            d.y = torch.tensor([float(y_list[i])], dtype=torch.float)
        results.append(d)
    return results


class _GINNet:
    """Factory for the GIN network architecture. All torch/PyG imports are lazy."""

    @staticmethod
    def build(in_channels, hidden_channels, num_layers, dropout):
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GINConv, global_mean_pool

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs   = nn.ModuleList()
                self.bns     = nn.ModuleList()
                self.dropout = nn.Dropout(dropout)
                in_c = in_channels
                for _ in range(num_layers):
                    mlp = nn.Sequential(
                        nn.Linear(in_c, hidden_channels),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                    self.convs.append(GINConv(mlp))
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
                    in_c = hidden_channels
                self.head = nn.Linear(hidden_channels, 1)

            def forward(self, x, edge_index, batch):
                for conv, bn in zip(self.convs, self.bns):
                    x = conv(x, edge_index)
                    x = bn(x)
                    x = F.relu(x)
                    x = self.dropout(x)
                x = global_mean_pool(x, batch)
                return self.head(x).squeeze(-1)

        return _Net()


# ---------------------------------------------------------------------------
# ChempropDMPNN
# ---------------------------------------------------------------------------

class ChempropDMPNN:
    """Directed Message-Passing Neural Network (D-MPNN) via Chemprop v2."""

    ATOM_FDIM = 33
    BOND_FDIM = 6

    def __init__(
        self,
        epochs:      int   = 50,
        batch_size:  int   = 64,
        hidden_size: int   = 300,
        depth:       int   = 3,
        ffn_hidden:  int   = 300,
        ffn_layers:  int   = 2,
        lr:          float = 1e-4,
        dropout:     float = 0.0,
        seed:        int   = 42,
    ):
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.hidden_size = hidden_size
        self.depth       = depth
        self.ffn_hidden  = ffn_hidden
        self.ffn_layers  = ffn_layers
        self.lr          = lr
        self.dropout     = dropout
        self.seed        = seed
        self.is_gnn      = True
        self._model       = None
        self._model_state = None

    def fit(self, smiles, y, sample_weight=None):
        import torch
        import lightning.pytorch as pl
        import chemprop.data as cdata
        import chemprop.models as cmodels
        from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

        torch.manual_seed(self.seed)

        y_arr   = np.asarray(y, dtype=np.float32)
        weights = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None

        datapoints = []
        for i, smi in enumerate(smiles):
            w = float(weights[i]) if weights is not None else 1.0
            datapoints.append(cdata.MoleculeDatapoint.from_smi(smi, np.array([y_arr[i]]), weight=w))

        dataset = cdata.MoleculeDataset(datapoints)
        loader  = cdata.build_dataloader(dataset, batch_size=self.batch_size, shuffle=True)

        mp  = BondMessagePassing(d_h=self.hidden_size, depth=self.depth)
        agg = MeanAggregation()
        ffn = RegressionFFN(
            n_tasks=1,
            input_dim=self.hidden_size,
            hidden_dim=self.ffn_hidden,
            n_layers=self.ffn_layers,
            dropout=self.dropout,
        )
        model = cmodels.MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            init_lr=self.lr,
            max_lr=self.lr * 10,
            final_lr=self.lr,
        )

        # Pre-compute max_steps so Lightning does NOT iterate the dataloader
        # upfront to estimate stepping batches (avoids the slow first-pass hang
        # on large datasets caused by lazy molecular-graph featurization).
        n_batches = math.ceil(len(smiles) / self.batch_size)
        max_steps = self.epochs * n_batches

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                max_steps=max_steps,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                default_root_dir=tmpdir,
            )
            trainer.fit(model, loader)

        self._model       = model
        self._model_state = model.state_dict()
        return self

    def predict(self, smiles) -> np.ndarray:
        import torch
        import lightning.pytorch as pl
        import chemprop.data as cdata

        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        datapoints = [cdata.MoleculeDatapoint.from_smi(smi) for smi in smiles]
        dataset    = cdata.MoleculeDataset(datapoints)
        loader     = cdata.build_dataloader(dataset, batch_size=self.batch_size, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = pl.Trainer(
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                default_root_dir=tmpdir,
            )
            raw = trainer.predict(self._model, loader)

        return np.concatenate([t.numpy() for t in raw]).ravel()

    def predict_uncertainty(self, smiles, T: int = 20) -> np.ndarray:
        import torch
        import chemprop.data as cdata

        if self._model is None:
            raise RuntimeError("Call fit() before predict_uncertainty().")

        datapoints = [cdata.MoleculeDatapoint.from_smi(smi) for smi in smiles]
        dataset    = cdata.MoleculeDataset(datapoints)
        loader     = cdata.build_dataloader(dataset, batch_size=self.batch_size, shuffle=False)

        dropout_states: dict = {}
        for name, m in self._model.named_modules():
            if isinstance(m, torch.nn.Dropout):
                dropout_states[name] = m.training
                m.train()

        all_preds = []
        with torch.no_grad():
            for _ in range(T):
                preds_t = []
                for bi, batch in enumerate(loader):
                    try:
                        out = self._model.predict_step(batch, bi)
                    except Exception:
                        for name, m in self._model.named_modules():
                            if name in dropout_states:
                                m.training = dropout_states[name]
                        self._model.eval()
                        return np.zeros(len(smiles), dtype=np.float32)
                    if isinstance(out, torch.Tensor):
                        preds_t.append(out.detach().numpy().ravel())
                    else:
                        preds_t.append(
                            np.concatenate([t.detach().numpy().ravel() for t in out])
                        )
                all_preds.append(np.concatenate(preds_t))

        for name, m in self._model.named_modules():
            if name in dropout_states:
                m.training = dropout_states[name]
        self._model.eval()

        return np.std(np.stack(all_preds), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# GIN (Graph Isomorphism Network)
# ---------------------------------------------------------------------------

class GIN:
    """Graph Isomorphism Network using PyTorch Geometric."""

    ATOM_FDIM = 33

    def __init__(
        self,
        hidden_channels: int   = 256,
        num_layers:      int   = 4,
        dropout:         float = 0.3,
        epochs:          int   = 100,
        lr:              float = 1e-3,
        weight_decay:    float = 1e-4,
        batch_size:      int   = 64,
        seed:            int   = 42,
    ):
        self.hidden_channels = hidden_channels
        self.num_layers      = num_layers
        self.dropout         = dropout
        self.epochs          = epochs
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.batch_size      = batch_size
        self.seed            = seed
        self.is_gnn          = True
        self._model  = None
        self._device = None

    def fit(self, smiles, y, sample_weight=None):
        import torch
        import torch.optim as optim
        from torch_geometric.loader import DataLoader

        torch.manual_seed(self.seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_arr   = np.asarray(y, dtype=np.float32)
        weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(len(y_arr), dtype=np.float32)
        )

        graph_list = _smiles_to_pyg(smiles, y_arr)
        for g, w in zip(graph_list, weights):
            g.weight = torch.tensor([w], dtype=torch.float)

        loader = DataLoader(graph_list, batch_size=self.batch_size, shuffle=True)

        self._model = _GINNet.build(
            self.ATOM_FDIM, self.hidden_channels, self.num_layers, self.dropout
        ).to(self._device)
        optimizer = optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self._model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                out   = self._model(batch.x, batch.edge_index, batch.batch)
                w_b   = batch.weight.squeeze(-1)
                loss  = (w_b * (out - batch.y.squeeze(-1)) ** 2).mean()
                loss.backward()
                optimizer.step()

        self._model.eval()
        return self

    def predict(self, smiles) -> np.ndarray:
        import torch
        from torch_geometric.loader import DataLoader

        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        loader = DataLoader(
            _smiles_to_pyg(smiles), batch_size=self.batch_size, shuffle=False
        )
        preds = []
        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                out   = self._model(batch.x, batch.edge_index, batch.batch)
                preds.append(out.cpu().numpy())

        return np.concatenate(preds)

    def predict_uncertainty(self, smiles, T: int = 20) -> np.ndarray:
        import torch
        from torch_geometric.loader import DataLoader

        if self._model is None:
            raise RuntimeError("Call fit() before predict_uncertainty().")

        loader    = DataLoader(
            _smiles_to_pyg(smiles), batch_size=self.batch_size, shuffle=False
        )
        all_preds = []
        self._model.train()
        with torch.no_grad():
            for _ in range(T):
                preds_t = []
                for batch in loader:
                    batch = batch.to(self._device)
                    out   = self._model(batch.x, batch.edge_index, batch.batch)
                    preds_t.append(out.cpu().numpy())
                all_preds.append(np.concatenate(preds_t))

        self._model.eval()
        return np.std(np.stack(all_preds), axis=0).astype(np.float32)
