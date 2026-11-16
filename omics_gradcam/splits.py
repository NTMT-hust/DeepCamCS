"""Stage 1b - repeated stratified cross-validation splits.

Per seed: stratified K-fold (outer); each fold holds out 1/K of the samples as the TEST set. The rest
is split once, stratified, into TRAIN and VALIDATION (``val_fraction``): with K = 5 and 0.2 that is
64% / 16% / 20%. Every sample is tested exactly once per seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class FoldSplit:
    seed: int
    fold: int              # 0-based
    train: np.ndarray      # sample indices, sorted
    val: np.ndarray
    test: np.ndarray

    def as_dict(self, sample_ids: np.ndarray) -> dict:
        return {"seed": self.seed, "fold": self.fold + 1, "train": sample_ids[self.train].tolist(),
                "val": sample_ids[self.val].tolist(), "test": sample_ids[self.test].tolist()}


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
