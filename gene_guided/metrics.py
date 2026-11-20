"""Classification metrics (macro averages; sensitivity / specificity one-vs-rest, then macro)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

# Main metrics, then imbalance / calibration checks (NLL, Brier, ECE: lower is better)
EVAL_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "roc_auc", "pr_auc",
                "sensitivity", "specificity"]
EXTRA_METRICS = ["balanced_accuracy", "f1_weighted", "kappa", "nll", "brier", "ece"]
KEY_METRICS = EVAL_METRICS + EXTRA_METRICS
PERCENT_METRICS = {"accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                   "sensitivity", "specificity"}
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)", "mcc": "MCC", "roc_auc": "ROC-AUC (OvR, macro)", "pr_auc": "PR-AUC (macro)",
    "sensitivity": "Sensitivity (macro)", "specificity": "Specificity (macro)",
    "balanced_accuracy": "Balanced accuracy", "f1_weighted": "F1 (weighted)", "kappa": "Cohen's kappa",
    "nll": "NLL", "brier": "Brier", "ece": "ECE",
}


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    K = len(class_names)
    labels = list(range(K))
    m: dict = {"n": int(len(y_true)), "accuracy": accuracy_score(y_true, y_pred),
               "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        m[f"precision_{avg}"], m[f"recall_{avg}"], m[f"f1_{avg}"] = p, r, f
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["kappa"] = cohen_kappa_score(y_true, y_pred)

    present = np.unique(y_true)
    try:
        if K == 2:
            m["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            m["pr_auc"] = average_precision_score(y_true, y_prob[:, 1])
        else:
            all_present = len(present) == K
            m["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels) \
                if all_present else float("nan")
            m["pr_auc"] = float(np.mean([average_precision_score(y_true == c, y_prob[:, c]) for c in present]))
    except ValueError:
        m["roc_auc"] = m["pr_auc"] = float("nan")

    m["nll"] = float(-np.log(np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean())
    m["brier"] = float(((y_prob - np.eye(K)[y_true]) ** 2).sum(1).mean())
    m["ece"] = expected_calibration_error(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    fp = cm.sum(0) - np.diag(cm)
    spec = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1), 0.0)
    m["sensitivity"], m["specificity"] = float(np.mean(r)), float(np.mean(spec))
    m["confusion_matrix"] = cm.tolist()
    m["per_class"] = {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                             "specificity": float(spec[i]), "support": int(s[i])}
                      for i, name in enumerate(class_names)}
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def fmt(m: dict) -> str:
    return (f"acc={m['accuracy']*100:.2f}% F1m={m['f1_macro']*100:.2f}% MCC={m['mcc']:.4f} "
            f"AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} sens={m['sensitivity']*100:.2f}% "
            f"spec={m['specificity']*100:.2f}% NLL={m['nll']:.4f} ECE={m['ece']:.4f}")


def format_confusion(cm, class_names) -> str:
    cm = np.asarray(cm)
    w = max(8, max(len(c) for c in class_names) + 1)
    lines = ["true\\pred".ljust(w) + "".join(c.rjust(w) for c in class_names)]
    lines += [c.ljust(w) + "".join(str(v).rjust(w) for v in cm[i]) for i, c in enumerate(class_names)]
    return "\n".join(lines)
