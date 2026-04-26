import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .constants import NON_FEAT_COLS, N_DESC

try:
    from descriptastorus.descriptors import rdDescriptors as _rdDescriptors
    _HAS_DESCRIPTASTORUS = True
except ImportError:
    _HAS_DESCRIPTASTORUS = False


class Featurizer:
    """Converts SMILES to 200 RDKit descriptors + 2048-bit Morgan fingerprint vector."""

    def __init__(self, fp_radius: int = 3, fp_size: int = 2048):
        self.fp_radius = fp_radius
        self.fp_size   = fp_size

        if not _HAS_DESCRIPTASTORUS:
            raise ImportError(
                "descriptastorus is required for RDKit2D descriptors. "
                "Install it with: pip install descriptastorus"
            )
        self._desc_gen = _rdDescriptors.RDKit2D()
        self.descriptor_names = [x[0] for x in self._desc_gen.GetColumns()][1:]

    def smiles_to_fp(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.fp_size, dtype=np.int8)
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fp_radius, fpSize=self.fp_size
        )
        arr = np.zeros(self.fp_size, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        return arr

    def smiles_to_descriptors(self, smiles: str) -> np.ndarray:
        result = self._desc_gen.process(smiles)
        n = len(self.descriptor_names)
        if result is None or result[0] == 0:
            return np.zeros(n, dtype=np.float64)
        arr = np.array(result[1:], dtype=np.float64)
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return arr

    def featurize_single(self, smiles: str) -> np.ndarray:
        return np.concatenate([
            self.smiles_to_descriptors(smiles),
            self.smiles_to_fp(smiles).astype(np.float64),
        ])

    def get_feature_names_single(self) -> list:
        return list(self.descriptor_names) + [f"fp{i}" for i in range(self.fp_size)]


def add_murcko_scaffolds(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Append a 'scaffold' column with canonical Bemis-Murcko SMILES."""
    def _scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "PARSE_ERROR"
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(_scaffold)
    return df


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Featurize raw SMILES with RDKit descriptors + Morgan FPs, add Murcko scaffolds,
    and one-hot encode the target column.

    Expected input columns: smiles, pIC50, target
    Output: DataFrame with 200 RDKit descriptor cols, 2048 Morgan FP bit cols,
            scaffold, scaffold_idx, per-target one-hot columns.
    """
    featurizer = Featurizer(fp_radius=3, fp_size=2048)
    feat_arr   = np.vstack([featurizer.featurize_single(s) for s in df["smiles"]])
    feat_df    = pd.DataFrame(
        feat_arr, columns=featurizer.get_feature_names_single(), index=df.index
    )

    out = add_murcko_scaffolds(df.copy(), smiles_col="smiles")
    out["scaffold_idx"] = out["scaffold"].map(
        {s: i for i, s in enumerate(out["scaffold"].unique())}
    )
    target_ohe = pd.get_dummies(out["target"], prefix="target")

    return pd.concat(
        [out[["smiles", "pIC50", "target", "scaffold", "scaffold_idx"]], feat_df, target_ohe],
        axis=1,
    ).reset_index(drop=True)


def split_dataset(
    df:         pd.DataFrame,
    split:      str   = "scaffold",
    train_frac: float = 0.85,
    seed:       int   = 100,
):
    """
    Return (train_df, test_df).

    split='scaffold' : scaffolds assigned greedily to train until train_frac is met.
    split='random'   : rows sampled at exactly train_frac.
    """
    rng = np.random.default_rng(seed)
    if split == "scaffold":
        unique = np.array(df["scaffold"].unique().copy())
        rng.shuffle(unique)
        sizes = df.groupby("scaffold").size()
        chosen, count = set(), 0
        for s in unique[:-1]:
            if count / len(df) < train_frac and s is not np.nan:
                chosen.add(s)
                count += sizes[s]
        mask = df["scaffold"].isin(chosen).values
    else:
        idx  = rng.permutation(len(df))
        mask = np.zeros(len(df), dtype=bool)
        mask[idx[: int(round(len(df) * train_frac))]] = True

    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def load_processed_data(path: str) -> pd.DataFrame:
    """Load a pre-featurized kinase CSV produced by process_dataset()."""
    return pd.read_csv(path, index_col=0)


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Extract feature matrices, labels, and SMILES lists from train/test DataFrames.
    Applies nan/inf sanitization in-place.

    Returns
    -------
    X_train, y_train, smiles_train, X_test, y_test, smiles_test
    """
    feat_cols = [c for c in train_df.columns if c not in NON_FEAT_COLS]

    X_train      = train_df[feat_cols].astype(np.float64).values
    y_train      = train_df["pIC50"].values
    smiles_train = train_df["smiles"].tolist()

    X_test      = test_df[feat_cols].astype(np.float64).values
    y_test      = test_df["pIC50"].values
    smiles_test = test_df["smiles"].tolist()

    np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return X_train, y_train, smiles_train, X_test, y_test, smiles_test
