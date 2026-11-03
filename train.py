import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ProcessHeatMapResult import (
    calculate_mean,
    visualize_mean_heatmap,
    find_genes_attribute_scores,
)
from StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stratified K-Fold Cross-Validation with Guided Learning & Grad-CAM"
    )

    # Dataset & Paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"H:\My Drive\bioinfor_training\28729127\MLOmics\Main_Dataset\Classification_datasets\GS-BRCA\Image\Aligned\dataset",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--coord_base",
        type=str,
        default="/kaggle/input/brca-cacnerbenchmark-withgenenamemapping/Image/Aligned_withmapping",
        help="Base directory containing omics coordinate CSV files (CNV, Methylation, mRNA).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Explanation",
        help="Directory to save explanation heatmaps and attribute scores.",
    )
    parser.add_argument(
        "--results_plot",
        type=str,
        default="imbalanced_kfold_comprehensive_results.png",
        help="Path to save the comprehensive training results plot.",
    )
    parser.add_argument(
        "--test_metrics_plot",
        type=str,
        default="test_metrics_by_fold.png",
        help="Path to save held-out test metrics by fold.",
    )
    parser.add_argument(
        "--test_results_json",
        type=str,
        default="fold_test_results.txt",
        help="Path to save the test results JSON.",
    )

    # Cross-Validation
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for Stratified K-Fold Cross Validation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=70,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Hyperparameters & Optimization
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=1e-3,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss.",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=5,
        help="Weight lambda1 for guided learning loss.",
    )
    # Display option
    parser.add_argument(
        "--show_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the plot window with plt.show() after training.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = StratifiedKFoldCrossValidation(
        dataset_path=args.dataset_path,
        k_folds=args.k_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda2=args.lambda2,
        dropout_rate=args.dropout_rate,
        focal_gamma=args.focal_gamma,
        random_seed=args.random_seed,
        lambda1=args.lambda1,
    )

    fold_results, _, _, class_names, _, all_ig_records, fold_test_results = model.run()

    print(f'\n{"="*60}')
    print("FINAL SUMMARY")
    print(f'{"="*60}')
    print(f"Classes: {class_names}")
    print(f"Number of folds: {len(fold_results)}")

    # Visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    for i, result in enumerate(fold_results):
        history = result["history"]

        axes[0, 0].plot(history["train_loss"], label=f"Fold {i+1}", alpha=0.7)
        axes[0, 1].plot(history["val_loss"], label=f"Fold {i+1}", alpha=0.7)
        axes[0, 2].plot(history["train_f1"], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 0].plot(history["train_acc"], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 1].plot(history["val_acc"], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 2].plot(history["val_f1"], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 0].plot(history["train_auc"], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 1].plot(history["val_auc"], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 2].plot(history["val_sens"], label=f"Fold {i+1}", alpha=0.7)

    axes[0, 0].set_title("Training Loss")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 2].set_title("Training F1-Score")
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 2].set_title("Validation F1-Score")
    axes[2, 0].set_title("Training AUC")
    axes[2, 1].set_title("Validation AUC")
    axes[2, 2].set_title("Validation Sensitivity")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.results_plot, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization saved as '{args.results_plot}'")

    # Test scores are fold-level values, so keep them separate from
    # the epoch-wise training and validation curves above.
    metric_specs = [
        ("accuracy", "Test Accuracy (%)"),
        ("precision_macro", "Test Macro Precision (%)"),
        ("recall_macro", "Test Macro Recall (%)"),
        ("f1_macro", "Test Macro F1 (%)"),
        ("roc_auc", "Test ROC AUC"),
        ("pr_auc", "Test PR AUC"),
        ("sensitivity", "Test Sensitivity (%)"),
        ("specificity", "Test Specificity (%)"),
        ("mcc", "Test MCC"),
    ]
    test_metrics_by_fold = [
        result["test_metrics"]["test_metrics"] for result in fold_results
    ]
    test_fig, test_axes = plt.subplots(3, 3, figsize=(18, 12))
    fold_labels = [f"Fold {i + 1}" for i in range(len(test_metrics_by_fold))]

    for ax, (metric_key, title) in zip(test_axes.flat, metric_specs):
        if metric_key == "roc_auc":
            values = [
                metrics.get("roc_auc", metrics.get("roc_auc_ovr", np.nan))
                for metrics in test_metrics_by_fold
            ]
        else:
            values = [metrics.get(metric_key, np.nan) for metrics in test_metrics_by_fold]

        bars = ax.bar(fold_labels, values, color="steelblue")
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                label = f"{value:.3f}" if metric_key in {"roc_auc", "pr_auc", "mcc"} else f"{value:.1f}"
                ax.annotate(
                    label,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    test_fig.suptitle("Held-Out Test Metrics by Fold", fontsize=16)
    test_fig.tight_layout()
    test_fig.savefig(args.test_metrics_plot, dpi=300, bbox_inches="tight")
    print(f"Test metrics plot saved as '{args.test_metrics_plot}'")

    if args.show_plot:
        plt.show()

    with open(args.test_results_json, "w", encoding="utf-8") as f:
        json.dump(
            fold_test_results,
            f,
            indent=4,
            default=lambda value: (
                value.tolist() if isinstance(value, np.ndarray)
                else value.item() if isinstance(value, np.generic)
                else str(value)
            ),
        )
    print(f"✓ Test results saved as '{args.test_results_json}'")

    # ── Explanation & Gene Attribute Score Pipeline ──────────────────
    if len(all_ig_records) > 0:
        print("\n" + "=" * 60)
        print("GENERATING INTEGRATED GRADIENTS ATTRIBUTE SCORES & GENE MAPPINGS")
        print("=" * 60)

        out = args.output_dir
        os.makedirs(out, exist_ok=True)

        coord_base = args.coord_base
        coord_files = {
            "CNV": os.path.join(coord_base, "gene_coordinates_CNV.csv"),
            "Methylation": os.path.join(coord_base, "gene_coordinates_Methylation.csv"),
            "mRNA": os.path.join(coord_base, "gene_coordinates_mRNA.csv"),
        }

        channels = {"mRNA": 0, "Methylation": 1, "CNV": 2}
        records_by_modality = {
            modality: calculate_mean([(attr[channel], cls) for attr, cls in all_ig_records])
            for modality, channel in channels.items()
        }

        all_attribution_dfs = []
        classes = sorted({cls for _, cls in all_ig_records})
        for cls in classes:
            print(f"\nProcessing IG maps for class '{cls}'...")
            class_gene_dfs = []

            for omics_type, coord_path in coord_files.items():
                if not os.path.exists(coord_path):
                    print(f"  ⚠️ Skipping {omics_type}: coordinate file not found at {coord_path}")
                    continue

                pixel_attr = records_by_modality[omics_type].get(cls)
                if pixel_attr is None:
                    continue

                visualize_mean_heatmap(
                    pixel_attr,
                    os.path.join(out, f"ig_mean_heatmap_{omics_type}_class_{cls}.png"),
                )
                gene_df = find_genes_attribute_scores(
                    attr_map=pixel_attr,
                    coord_filepath=coord_path,
                    omics_type=omics_type,
                    predicted_label=cls,
                )
                gene_df.to_csv(os.path.join(out, f"ig_{omics_type.lower()}_class_{cls}.csv"), index=False)
                class_gene_dfs.append(gene_df)

            if class_gene_dfs:
                cls_combined_df = pd.concat(class_gene_dfs, ignore_index=True)
                cls_combined_df.to_csv(os.path.join(out, f"predicted_label_{cls}_attribute_scores.csv"), index=False)
                all_attribution_dfs.append(cls_combined_df)

        if all_attribution_dfs:
            master_df = pd.concat(all_attribution_dfs, ignore_index=True)
            master_df.to_csv(os.path.join(out, "all_classes_attribute_scores.csv"), index=False)
            print(f"\n✅ All gene attribute scores saved to '{os.path.join(out, 'all_classes_attribute_scores.csv')}'")
    else:
        print("\n⚠️ No Integrated Gradients records were generated.")


if __name__ == "__main__":
    main()
