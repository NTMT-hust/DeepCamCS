import argparse
import glob
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Select top K positive and negative attributed genes per subtype.")
    parser.add_argument("--base_dir", type=str, default="Explanation", help="Directory containing attribution CSV files.")
    parser.add_argument("--k", type=int, default=400, help="Number of top genes to select.")
    parser.add_argument("--output_dir", type=str, default="selected_genes_400", help="Directory to save selected gene CSVs.")
    return parser.parse_args()


def select_top_genes(base_dir="Explanation", k=400, output_dir="selected_genes_400"):
    os.makedirs(output_dir, exist_ok=True)

    # Search for all attribution CSVs in base directory or fold subdirectories
    patterns = [
        os.path.join(base_dir, "predicted_label_*_attribute_scores.csv"),
        os.path.join(base_dir, "all_classes_attribute_scores.csv"),
        os.path.join(base_dir, "Fold_*", "*.csv"),
        os.path.join(base_dir, "*.csv"),
    ]

    files = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            files.extend(matched)
            break

    if not files:
        print(f"⚠️ No CSV files found matching patterns in '{base_dir}'.")
        return

    print(f"Loading {len(files)} attribution file(s)...")
    dfs = [pd.read_csv(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)

    required_cols = ["gene_name", "predicted_label", "omics_type", "attribute_score"]
    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"Missing required column: '{col}' in input files.")

    # Median score grouped by predicted_label, omics_type, and gene_name
    mean_df = (
        full_df.groupby(["predicted_label", "omics_type", "gene_name"])["attribute_score"]
        .median()
        .reset_index()
    )

    classes = sorted(mean_df["predicted_label"].unique())
    for cls in classes:
        print(f"Processing subtype label '{cls}'...")
        class_df = mean_df[mean_df["predicted_label"] == cls]

        pos_list = []
        neg_list = []

        for omics_type in class_df["omics_type"].unique():
            omics_df = class_df[class_df["omics_type"] == omics_type]

            top_pos = (
                omics_df[omics_df["attribute_score"] > 0]
                .sort_values("attribute_score", ascending=False)
                .head(k)
            )
            top_neg = (
                omics_df[omics_df["attribute_score"] < 0]
                .sort_values("attribute_score", ascending=True)
                .head(k)
            )

            pos_list.append(top_pos)
            neg_list.append(top_neg)

        if pos_list:
            final_pos = pd.concat(pos_list, ignore_index=True)
            pos_path = os.path.join(output_dir, f"predicted_label_{cls}_top_{k}_positive.csv")
            final_pos.to_csv(pos_path, index=False)

        if neg_list:
            final_neg = pd.concat(neg_list, ignore_index=True)
            neg_path = os.path.join(output_dir, f"predicted_label_{cls}_top_{k}_negative.csv")
            final_neg.to_csv(neg_path, index=False)

        print(f"  ✓ Saved top {k} positive & negative files for label '{cls}'")

    print(f"\n🎉 Completed selecting top genes. Outputs saved to '{output_dir}'.")


if __name__ == "__main__":
    args = parse_args()
    select_top_genes(base_dir=args.base_dir, k=args.k, output_dir=args.output_dir)