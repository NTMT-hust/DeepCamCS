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
    """Read omics and labels, accepting common filename and label-table variants.

    Configured names take precedence. If one is absent, known aliases are tried. Label tables may
    contain a single class column or one-hot class columns, with common sample-ID column names.
    """
    base = Path(cfg.data_dir)
    labels_path = resolve_file(base, cfg.labels_file, ("labels.csv", "label.csv", "labels.txt", "label.txt"))
    lab = _read_labels(labels_path, cfg.label_id_col, cfg.label_col)

    aliases = {
        "mrna": ("mRNA_filtered.csv", "mRNA.csv", "mrna.csv", "gene_expression.csv", "expression.csv"),
        "methylation": ("methylation_filtered.csv", "Methylation_filtered.csv", "methylation.csv", "DNA_methylation.csv"),
        "cnv": ("CNA_filtered.csv", "CNV_filtered.csv", "cnv.csv", "CNA.csv", "copy_number.csv"),
    }
    tables = {}
    for name, fname in cfg.omics.items():
        candidates = aliases.get(name.lower(), (fname,))
        path = resolve_file(base, fname, candidates)
        tables[name] = read_omics_table(path)
    common = [s for s in lab.index if all(s in t.columns for t in tables.values())]
    if not common:
        raise ValueError("no sample is present in the labels file and in every omics file")
    if len(common) < len(lab):
        log(f"[data] {len(lab) - len(common)} labelled samples missing from >= 1 omics file were dropped")

    label_text = lab.loc[common].astype(str).to_numpy()
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


def resolve_file(base: Path, preferred: str, aliases: tuple[str, ...]) -> Path:
    """Find a configured file first, then case-insensitive known aliases."""
    preferred_path = base / preferred
    if preferred_path.is_file():
        return preferred_path
    by_lower = {p.name.lower(): p for p in base.iterdir() if p.is_file()}
    for name in (preferred, *aliases):
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    tried = ", ".join(dict.fromkeys((preferred, *aliases)))
    raise FileNotFoundError(f"could not find {preferred!r} in {base}; tried: {tried}")


def _read_labels(path: Path, id_col: str, label_col: str) -> pd.Series:
    """Read either a sample/class table or a sample plus one-hot class table."""
    lab = pd.read_csv(path)
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in lab.columns}
    id_candidates = (id_col, "patient_id", "sample_id", "sample", "sampleid", "barcode", "id")
    found_id = next((normalized.get(str(c).lower()) for c in id_candidates
                     if normalized.get(str(c).lower()) is not None), None)
    if found_id is None:
        # A common alternative stores sample IDs in the first unnamed/index column.
        unnamed = [c for c in lab.columns if str(c).lower().startswith("unnamed")]
        if unnamed:
            found_id = unnamed[0]
    if found_id is None:
        raise KeyError(f"{path} needs a sample ID column (e.g. {id_col!r}, patient_id, or sample_id); found {list(lab.columns)}")

    label_candidates = (label_col, "PAM50", "PAM50_subtype", "molecular_subtype", "subtype", "class", "label", "target")
    target = next((normalized[key] for c in label_candidates
                   if (key := str(c).strip().lower().replace(" ", "_")) in normalized), None)
    if target is not None:
        out = lab[[found_id, target]].dropna().copy()
        out.columns = ["sample_id", "label"]
    else:
        # One-hot labels: all remaining columns must be binary and each row must select one class.
        class_cols = [c for c in lab.columns if c != found_id]
        numeric = lab[class_cols].apply(pd.to_numeric, errors="coerce")
        if not class_cols or numeric.isna().any().any() or not numeric.isin([0, 1]).all().all():
            raise KeyError(f"{path} has no recognized target column ({label_col!r}, PAM50, subtype, class, label) "
                           f"and is not a one-hot label table; found {list(lab.columns)}")
        active = numeric.to_numpy().sum(axis=1)
        if not np.all(active == 1):
            raise ValueError(f"one-hot labels in {path} must have exactly one active class per sample")
        out = pd.DataFrame({"sample_id": lab[found_id], "label": np.asarray(class_cols)[numeric.to_numpy().argmax(axis=1)]}).dropna()
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    if out["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample IDs in {path}")
    return out.set_index("sample_id")["label"]


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
