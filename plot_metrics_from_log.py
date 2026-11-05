import argparse
from collections import defaultdict
import os
import re
import matplotlib.pyplot as plt
import numpy as np


FOLD_HEADER_RE = re.compile(r"={10,}\s*FOLD\s+(\d+)/(\d+)\s*={10,}")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
EPOCH_METRIC_RE = re.compile(
    r"(Train|Val|Test)\s*-\s*Loss:\s*([0-9.]+)\s*\|\s*Acc:\s*([0-9.]+)%\s*\|\s*AUC:\s*([0-9.]+)\s*\|\s*F1:\s*([0-9.]+)%"
)
FOLD_TEST_RE = re.compile(
    r"\[Fold\s+(\d+)\]\s*TEST\s*[—-]\s*Acc:\s*([0-9.]+)%\s*\|\s*AUC:\s*([0-9.]+)\s*\|\s*F1:\s*([0-9.]+)%"
)
DETAILED_HEADER_RE = re.compile(r"FOLD\s+(\d+)\s*-\s*DETAILED METRICS")

DETAILED_PATTERNS = {
    "accuracy": re.compile(r"Accuracy:\s*([0-9.]+)%"),
    "precision_macro": re.compile(r"Precision \(Macro\):\s*([0-9.]+)%"),
    "precision_weighted": re.compile(r"Precision \(Weight\):\s*([0-9.]+)%"),
    "recall_macro": re.compile(r"Recall \(Macro\):\s*([0-9.]+)%"),
    "recall_weighted": re.compile(r"Recall \(Weight\):\s*([0-9.]+)%"),
    "f1_macro": re.compile(r"F1-Score \(Macro\):\s*([0-9.]+)%"),
    "f1_weighted": re.compile(r"F1-Score \(Weight\):\s*([0-9.]+)%"),
    "mcc": re.compile(r"MCC:\s*([+-]?[0-9.]+)"),
    "roc_auc": re.compile(r"ROC-AUC(?:\s*\(OvR\))?:\s*([0-9.]+)"),
    "pr_auc": re.compile(r"PR-AUC:\s*([0-9.]+)"),
    "sensitivity": re.compile(r"Sensitivity:\s*([0-9.]+)%"),
    "specificity": re.compile(r"Specificity:\s*([0-9.]+)%"),
}


def parse_log_file(log_path):
    """Parse epoch histories, held-out test metrics, and detailed validation metrics from a pipeline log."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    # Normalize carriage returns from tqdm progress bars into newlines
    lines = raw_text.replace("\r", "\n").splitlines()

    folds_history = defaultdict(lambda: defaultdict(list))
    fold_test_summary = {}
    fold_val_detailed = defaultdict(dict)

    current_fold = None
    in_detailed_fold = None

    for line in lines:
        fold_match = FOLD_HEADER_RE.search(line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            in_detailed_fold = None
            continue

        det_match = DETAILED_HEADER_RE.search(line)
        if det_match:
            in_detailed_fold = int(det_match.group(1))
            continue

        test_match = FOLD_TEST_RE.search(line)
        if test_match:
            fold_idx = int(test_match.group(1))
            fold_test_summary[fold_idx] = {
                "accuracy": float(test_match.group(2)),
                "roc_auc": float(test_match.group(3)),
                "f1_macro": float(test_match.group(4)),
            }
            continue

        if current_fold is not None:
            ep_metric = EPOCH_METRIC_RE.search(line)
            if ep_metric:
                split = ep_metric.group(1).lower()
                folds_history[current_fold][f"{split}_loss"].append(float(ep_metric.group(2)))
                folds_history[current_fold][f"{split}_acc"].append(float(ep_metric.group(3)))
                folds_history[current_fold][f"{split}_auc"].append(float(ep_metric.group(4)))
                folds_history[current_fold][f"{split}_f1"].append(float(ep_metric.group(5)))
                continue

        if in_detailed_fold is not None:
            for key, pattern in DETAILED_PATTERNS.items():
                m = pattern.search(line)
                if m and key not in fold_val_detailed[in_detailed_fold]:
                    fold_val_detailed[in_detailed_fold][key] = float(m.group(1))

    return dict(folds_history), fold_test_summary, dict(fold_val_detailed)


def plot_epoch_curves(folds_history, save_path, per_fold_save_path=None):
    """Plot Train, Validation, and Test together in the same chart for each metric."""
    splits = [
        ("train", "Train", "#1f77b4"),
        ("val", "Validation", "#ff7f0e"),
        ("test", "Test", "#2ca02c"),
    ]
    metrics = [
        ("loss", "Loss"),
        ("acc", "Accuracy (%)"),
        ("auc", "ROC-AUC"),
        ("f1", "Macro F1-Score (%)"),
    ]
    sorted_folds = sorted(folds_history.keys())

    # 1) Combined 2x2 chart (1 subplot per metric, Train/Val/Test overlaid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (metric_key, metric_label) in zip(axes.flat, metrics):
        for split_key, split_label, color in splits:
            series_key = f"{split_key}_{metric_key}"
            fold_series = [
                folds_history[f].get(series_key, [])
                for f in sorted_folds
                if folds_history[f].get(series_key)
            ]
            if not fold_series:
                continue

            if len(fold_series) == 1:
                vals = fold_series[0]
                epochs = np.arange(1, len(vals) + 1)
                ax.plot(epochs, vals, color=color, linewidth=2.0, label=split_label)
            else:
                # Plot individual fold trajectories lightly and the mean curve boldly
                for idx, vals in enumerate(fold_series):
                    epochs = np.arange(1, len(vals) + 1)
                    ax.plot(epochs, vals, color=color, alpha=0.22, linewidth=1.0)

                max_len = max(len(s) for s in fold_series)
                padded = np.full((len(fold_series), max_len), np.nan)
                for idx, vals in enumerate(fold_series):
                    padded[idx, : len(vals)] = vals
                mean_vals = np.nanmean(padded, axis=0)
                epochs = np.arange(1, max_len + 1)
                ax.plot(
                    epochs,
                    mean_vals,
                    color=color,
                    linewidth=2.2,
                    label=f"{split_label} (Mean across {len(fold_series)} folds)",
                )

        ax.set_title(f"{metric_label} — Train vs. Validation vs. Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Train, Validation & Test Metrics Over Epochs", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved combined Train/Val/Test metric curves to: {save_path}")

    # 2) Per-fold grid where each fold has Train, Val, and Test in the same chart per metric
    if per_fold_save_path and sorted_folds:
        n_folds = len(sorted_folds)
        fig_f, axes_f = plt.subplots(
            n_folds, 4, figsize=(22, 4.2 * n_folds), squeeze=False
        )
        for row, fold in enumerate(sorted_folds):
            for col, (metric_key, metric_label) in enumerate(metrics):
                ax = axes_f[row, col]
                for split_key, split_label, color in splits:
                    vals = folds_history[fold].get(f"{split_key}_{metric_key}", [])
                    if vals:
                        epochs = np.arange(1, len(vals) + 1)
                        ax.plot(epochs, vals, color=color, linewidth=1.6, alpha=0.85, label=split_label)
                ax.set_title(f"Fold {fold} — {metric_label}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric_label)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

        fig_f.suptitle("Per-Fold Train, Validation & Test Curves", fontsize=16)
        fig_f.tight_layout()
        fig_f.savefig(per_fold_save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_f)
        print(f"✓ Saved per-fold Train/Val/Test curves to: {per_fold_save_path}")


def plot_fold_summaries(fold_val_detailed, fold_test_summary, save_path):
    """Plot bar charts of best validation and held-out test metrics per fold."""
    all_folds = sorted(set(fold_val_detailed.keys()) | set(fold_test_summary.keys()))
    if not all_folds:
        print("ℹ️ No completed fold summary blocks found yet in log (training may still be in progress).")
        return

    metric_specs = [
        ("val", "accuracy", "Best Val Accuracy (%)"),
        ("val", "f1_macro", "Best Val Macro F1 (%)"),
        ("val", "roc_auc", "Best Val ROC-AUC (OvR)"),
        ("val", "pr_auc", "Best Val PR-AUC"),
        ("val", "sensitivity", "Best Val Sensitivity (%)"),
        ("val", "specificity", "Best Val Specificity (%)"),
        ("test", "accuracy", "Held-Out Test Accuracy (%)"),
        ("test", "f1_macro", "Held-Out Test Macro F1 (%)"),
        ("test", "roc_auc", "Held-Out Test ROC-AUC"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fold_labels = [f"Fold {f}" for f in all_folds]

    for ax, (source, key, title) in zip(axes.flat, metric_specs):
        source_dict = fold_val_detailed if source == "val" else fold_test_summary
        values = [source_dict.get(f, {}).get(key, np.nan) for f in all_folds]
        color = "teal" if source == "val" else "steelblue"

        bars = ax.bar(fold_labels, values, color=color, alpha=0.85)
        finite_vals = [v for v in values if np.isfinite(v)]
        if finite_vals:
            mean_val = float(np.mean(finite_vals))
            ax.axhline(mean_val, color="crimson", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.2f}")
            ax.legend(fontsize=8, loc="lower right")

        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                fmt = f"{val:.4f}" if key in {"roc_auc", "pr_auc", "mcc"} else f"{val:.2f}"
                ax.annotate(
                    fmt,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("Fold-Level Best Validation & Held-Out Test Metrics from Log", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved fold summary plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse a saved DeepCamCS pipeline log and plot training/validation/test metrics.")
    parser.add_argument(
        "--log_file",
        type=str,
        default="OutputData/pipeline_20261105_105502.txt",
        help="Path to the pipeline log file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="OutputData",
        help="Directory to save the generated metric plots.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_stem = os.path.splitext(os.path.basename(args.log_file))[0]

    folds_history, fold_test_summary, fold_val_detailed = parse_log_file(args.log_file)
    print(f"Parsed '{args.log_file}':")
    for fold in sorted(folds_history.keys()):
        n_ep = len(folds_history[fold].get("train_loss", []))
        print(f"  Fold {fold}: {n_ep} epoch(s) logged")

    curves_path = os.path.join(args.output_dir, f"{log_stem}_epoch_curves.png")
    per_fold_curves_path = os.path.join(args.output_dir, f"{log_stem}_epoch_curves_by_fold.png")
    summary_path = os.path.join(args.output_dir, f"{log_stem}_fold_metrics.png")

    plot_epoch_curves(folds_history, curves_path, per_fold_curves_path)
    plot_fold_summaries(fold_val_detailed, fold_test_summary, summary_path)


if __name__ == "__main__":
    main()

