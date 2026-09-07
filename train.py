import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ProcessHeatMapResult import (
    calculate_mean,
    calculate_pixel_attribute_score,
    visualize_mean_heatmap,
    find_critical_pixel,
    save_to_csv,
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
        "--test_results_json",
        type=str,
        default="fold_test_results.txt",
        help="Path to save the test results JSON.",
    )

    # Model Architecture & Cross-Validation
    parser.add_argument(
        "--model_name",
        type=str,
        default="EfficientNetB1Classifier",
        help="Name of model classifier architecture.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for Stratified K-Fold Cross Validation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=0,
        help="Number of initial epochs to freeze backbone weights.",
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
        default=0.0003,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.6,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor.",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.1,
        help="Weight lambda1 for guided learning loss.",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=10.0,
        help="Weight lambda2 for guided learning loss.",
    )

    # Boolean flags
    parser.add_argument(
        "--use_class_aware_aug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable class-aware data augmentation.",
    )
    parser.add_argument(
        "--use_weighted_sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable weighted random sampling for imbalanced classes.",
    )
    parser.add_argument(
        "--use_temperature_scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable temperature scaling calibration.",
    )
    parser.add_argument(
        "--calculate_cluster_metrics_flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable cluster metrics calculation.",
    )
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
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        k_folds=args.k_folds,
        num_epochs=args.num_epochs,
        freeze_epochs=args.freeze_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_class_aware_aug=args.use_class_aware_aug,
        use_weighted_sampling=args.use_weighted_sampling,
        use_temperature_scaling=args.use_temperature_scaling,
        calculate_cluster_metrics_flag=args.calculate_cluster_metrics_flag,
        random_seed=args.random_seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )

    fold_results, fold_models, ensemble_metrics, class_names, calibrators, all_heatmaps, fold_test_results = model.run()

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

    if args.show_plot:
        plt.show()

    with open(args.test_results_json, "w", encoding="utf-8") as f:
        json.dump(fold_test_results, f, indent=4)
    print(f"✓ Test results saved as '{args.test_results_json}'")

    # ── Explanation & Gene Attribute Score Pipeline ──────────────────
    if len(all_heatmaps) > 0:
        print("\n" + "=" * 60)
        print("GENERATING GRAD-CAM ATTRIBUTE SCORES & GENE MAPPINGS")
        print("=" * 60)

        mean_heatmaps = calculate_mean(all_heatmaps)
        out = args.output_dir
        os.makedirs(out, exist_ok=True)

        coord_base = args.coord_base
        coord_files = {
            "CNV": os.path.join(coord_base, "gene_coordinates_CNV.csv"),
            "Methylation": os.path.join(coord_base, "gene_coordinates_Methylation.csv"),
            "mRNA": os.path.join(coord_base, "gene_coordinates_mRNA.csv"),
        }

        all_attribution_dfs = []

        for cls, mean_heatmap in mean_heatmaps.items():
            print(f"\nProcessing class '{cls}' explanation map...")

            # 1. Calculate per-pixel attribute score
            pixel_attr = calculate_pixel_attribute_score(mean_heatmap, method="raw")

            # 2. Visualize heatmap
            visualize_mean_heatmap(pixel_attr, os.path.join(out, f"mean_heatmap_class_{cls}.png"))

            # 3. Critical pixels (top 10%)
            threshold = np.percentile(pixel_attr, 90)
            pixels = find_critical_pixel(pixel_attr, threshold)
            save_to_csv(pixels, os.path.join(out, f"critical_pixels_class_{cls}.csv"), headers=["x", "y", "attribute_score"])

            # 4. Map pixel attribute scores to genes for each omics modality
            class_gene_dfs = []
            for omics_type, coord_path in coord_files.items():
                if os.path.exists(coord_path):
                    gene_df = find_genes_attribute_scores(
                        attr_map=pixel_attr,
                        coord_filepath=coord_path,
                        omics_type=omics_type,
                        predicted_label=cls,
                    )
                    gene_df.to_csv(os.path.join(out, f"{omics_type.lower()}_class_{cls}.csv"), index=False)
                    class_gene_dfs.append(gene_df)
                else:
                    print(f"  ⚠️ Skipping {omics_type}: coordinate file not found at {coord_path}")

            # 5. Combine omics for this class into format ready for ChooseGenes.py
            if class_gene_dfs:
                cls_combined_df = pd.concat(class_gene_dfs, ignore_index=True)
                cls_combined_df.to_csv(os.path.join(out, f"predicted_label_{cls}_attribute_scores.csv"), index=False)
                all_attribution_dfs.append(cls_combined_df)

        # 6. Save master attribute file across all classes
        if all_attribution_dfs:
            master_df = pd.concat(all_attribution_dfs, ignore_index=True)
            master_df.to_csv(os.path.join(out, "all_classes_attribute_scores.csv"), index=False)
            print(f"\n✅ All gene attribute scores saved to '{os.path.join(out, 'all_classes_attribute_scores.csv')}'")
    else:
        print("\n⚠️ No heatmaps collected during training to generate explanations.")


if __name__ == "__main__":
    main()
