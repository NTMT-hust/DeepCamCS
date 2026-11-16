"""Stage 5 - post-hoc probability calibration on the validation split.

    cal = build_calibrator("beta").fit(val_logits, val_labels)
    probs = cal.predict_proba(logits)
    cal.tune_thresholds(cal.predict_proba(val_logits), val_labels)     # optional
    preds = cal.predict(probs)                                          # argmax(p - thresholds)

``beta``         one-vs-rest Beta calibration (Kull et al. 2017) + renormalisation
``temperature``  one temperature fitted by NLL (Guo et al. 2017)
``none``         softmax
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression


def _softmax(logits: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64), axis=1)


class Calibrator:
    """Base class: softmax probabilities and optional per-class decision thresholds."""
    thresholds: np.ndarray | None = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "Calibrator":
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(logits)

    def tune_thresholds(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per class, the one-vs-rest threshold in [0.05, 0.95) that maximises F1."""
        candidates = np.arange(0.05, 0.95, 0.01)
        th = np.full(probs.shape[1], 0.5)
        for c in range(probs.shape[1]):
            y = labels == c
            pred = probs[:, c][None, :] >= candidates[:, None]               # [T, N]
            tp = (pred & y).sum(1)
            fp = (pred & ~y).sum(1)
            fn = (~pred & y).sum(1)
            denom = 2 * tp + fp + fn
            f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
            best = int(np.argmax(f1))
            if f1[best] > 0:
                th[c] = float(candidates[best])
        self.thresholds = th
        return th

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return probs.argmax(1) if self.thresholds is None else (probs - self.thresholds[None, :]).argmax(1)


class BetaCalibrator(Calibrator):
    """Per class: logistic regression on [log p_c, log(1 - p_c)], then renormalise."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.models: list = []

    @staticmethod
    def _features(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.column_stack([np.log(p), np.log1p(-p)])

    def fit(self, logits, labels):
        probs = _softmax(logits)
        self.models = []
        for c in range(probs.shape[1]):
            y = (labels == c).astype(int)
            if y.min() == y.max():                     # class absent from the validation split
                self.models.append(None)
                continue
            clf = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, class_weight="balanced")
            self.models.append(clf.fit(self._features(probs[:, c]), y))
        return self

    def predict_proba(self, logits):
        probs = _softmax(logits)
        out = np.empty_like(probs)
        for c, clf in enumerate(self.models):
            out[:, c] = probs[:, c] if clf is None else clf.predict_proba(self._features(probs[:, c]))[:, 1]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-9)


class TemperatureScaler(Calibrator):
    def __init__(self, max_iter: int = 100, lr: float = 0.01):
        self.max_iter, self.lr = max_iter, lr
        self.T = 1.0

    def fit(self, logits, labels):
        lg = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
        lb = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)       # T = exp(log_t) > 0
        opt = torch.optim.LBFGS([log_t], lr=self.lr, max_iter=self.max_iter)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), lb)
            loss.backward()
            return loss
        opt.step(closure)
        self.T = float(log_t.exp().item())
        return self

    def predict_proba(self, logits):
        return _softmax(np.asarray(logits) / self.T)


def build_calibrator(method: str) -> Calibrator:
    return {"beta": BetaCalibrator, "temperature": TemperatureScaler, "none": Calibrator}[method]()
