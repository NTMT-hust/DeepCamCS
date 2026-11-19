"""Load the omics tables and the labels (aligned by sample ID), and make the cross-validation splits.

Splits, per seed: stratified K-fold (outer); each fold holds out 1/K of the samples as the TEST set. The rest
is split once, stratified, into TRAIN and VALIDATION (``val_fraction``): with K = 5 and 0.2 that is
64% / 16% / 20%. Every sample is tested exactly once per seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .config import DataConfig


@dataclass
class OmicsMatrix:
    name: str
    feature_names: np.ndarray      # [n_features]
    values: np.ndarray             # [n_features, n_samples] float32


@dataclass
class MultiOmicsData:
    sample_ids: np.ndarray         # [n_samples]
    labels: np.ndarray             # [n_samples] int, index into class_names
    class_names: list              # sorted label names
    omics: list                    # [OmicsMatrix], order = image channel order

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def omics_names(self) -> list:
        return [o.name for o in self.omics]

    def matrices(self, idx: np.ndarray | None = None) -> list:
        """Per omics: [n_features, n_selected_samples]."""
        return [o.values if idx is None else o.values[:, idx] for o in self.omics]


def read_omics_table(path: Path) -> pd.DataFrame:
    """One omics CSV (features x samples); duplicated feature IDs get a ``__1``, ``__2`` ... suffix."""
    df = pd.read_csv(path, index_col=0)
    df.index = _unique(df.index.astype(str).tolist())
    df.columns = df.columns.astype(str)
    return df


def load_multiomics(cfg: DataConfig, log=print) -> MultiOmicsData:
    """Read every omics CSV and the label CSV; keep the samples present in all of them, in label-file
    order. NaN values are replaced by 0."""
    base = Path(cfg.data_dir)
    labels_path = base / cfg.labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path} (see data/README.md)")
    lab = pd.read_csv(labels_path)
    for col in (cfg.label_id_col, cfg.label_col):
        if col not in lab.columns:
            raise KeyError(f"{labels_path} has no column {col!r}; found {list(lab.columns)}")
    lab = lab[[cfg.label_id_col, cfg.label_col]].dropna()
    lab[cfg.label_id_col] = lab[cfg.label_id_col].astype(str)
    if lab[cfg.label_id_col].duplicated().any():
        raise ValueError(f"duplicate sample IDs in {labels_path}")
    lab = lab.set_index(cfg.label_id_col)

    tables = {name: read_omics_table(base / fname) for name, fname in cfg.omics.items()}
    common = [s for s in lab.index if all(s in t.columns for t in tables.values())]
    if not common:
        raise ValueError("no sample is present in the labels file and in every omics file")
    if len(common) < len(lab):
        log(f"[data] {len(lab) - len(common)} labelled samples missing from >= 1 omics file were dropped")

    label_text = lab.loc[common, cfg.label_col].astype(str).to_numpy()
    class_names = sorted(set(label_text))
    y = np.array([class_names.index(c) for c in label_text], dtype=np.int64)

    omics = []
    for name, df in tables.items():
        values = df[common].to_numpy(dtype=np.float32)
        n_nan = int(np.isnan(values).sum())
        if n_nan:
            log(f"[data] {name}: {n_nan} NaN values replaced by 0")
            values = np.nan_to_num(values, nan=0.0)
        omics.append(OmicsMatrix(name, df.index.to_numpy(dtype=str), values))
        log(f"[data] {name:<12} {values.shape[0]:>6} features x {values.shape[1]} samples")
    log("[data] classes: " + ", ".join(f"{c}={n}" for c, n in zip(class_names, np.bincount(y))))
    return MultiOmicsData(np.array(common, dtype=str), y, class_names, omics)


def _unique(names: list) -> list:
    seen: dict = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


@dataclass
class FoldSplit:
    seed: int
    fold: int              # 0-based
    train: np.ndarray      # sample indices, sorted
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict:
        return {"seed": self.seed, "fold": self.fold + 1, "train": self.train.tolist(), "val": self.val.tolist(),
                "test": self.test.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FoldSplit":
        return cls(int(d["seed"]), int(d["fold"]) - 1, *(np.asarray(d[k], dtype=np.int64) for k in ("train", "val", "test")))


def make_cv_splits(y: np.ndarray, seed: int, n_folds: int = 5, val_fraction: float = 0.2) -> list:
    y = np.asarray(y)
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for fold, (trval, test) in enumerate(outer.split(np.zeros(len(y)), y)):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr, va = next(inner.split(np.zeros(len(trval)), y[trval]))
        splits.append(FoldSplit(seed, fold, np.sort(trval[tr]), np.sort(trval[va]), np.sort(test)))
    for sp in splits:
        a, b, c = set(sp.train), set(sp.val), set(sp.test)
        assert not (a & b or a & c or b & c) and len(a | b | c) == len(y), "train/val/test must partition the data"
    tested = np.concatenate([sp.test for sp in splits])
    assert len(np.unique(tested)) == len(y) == len(tested), "every sample must be tested exactly once"
    return splits
