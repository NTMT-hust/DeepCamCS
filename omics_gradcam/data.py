"""Stage 1a - load the omics tables and the labels, aligned by sample ID."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def omics_names(self) -> list:
        return [o.name for o in self.omics]

    def matrices(self, idx: np.ndarray | None = None) -> list:
        """Per omics: [n_features, n_selected_samples]."""
        return [o.values if idx is None else o.values[:, idx] for o in self.omics]


def load_multiomics(cfg: DataConfig, log=print) -> MultiOmicsData:
    """Read every omics CSV (features x samples) and the label CSV; keep the samples present in all
    of them, in label-file order. NaN values are replaced by 0."""
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

    tables = {}
    for name, fname in cfg.omics.items():
        df = pd.read_csv(base / fname, index_col=0)
        df.index = _unique(df.index.astype(str).tolist())
        df.columns = df.columns.astype(str)
        tables[name] = df

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
    """Make duplicated feature IDs unique: the repeats get a ``__1``, ``__2`` ... suffix."""
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
