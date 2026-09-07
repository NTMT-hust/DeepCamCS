import argparse
import glob
import os
import pandas as pd

PAM50 = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA",
    "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF",
    "CEP55", "CXXC5", "DCN", "EGFR", "ERBB2", "ESR1",
    "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2",
    "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2",
    "MYC", "NAT1", "NDC80", "NUF2", "ORC6", "PGR",
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6",
    "TMEM45B", "TYMS", "UBE2C", "UBE2T",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge selected genes from multi-omics modalities and evaluate PAM50 overlap.")
    parser.add_argument("--folder", type=str, default="selected_genes_400", help="Folder containing top selected gene CSVs.")
    parser.add_argument("--output_dir", type=str, default="Top", help="Folder to save merged unique gene CSV.")
    return parser.parse_args()


def merge_and_analyze(folder="selected_genes_400", output_dir="Top"):
    pattern = os.path.join(folder, "predicted_label_*_top_*_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        # Fallback to any CSV in folder
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not files:
        print(f"⚠️ No CSV files found in '{folder}'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(files)} files in '{folder}'.")

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(combined):,}")

    unique_df = (
        combined[["gene_name", "omics_type"]]
        .drop_duplicates()
        .rename(columns={"gene_name": "unique_genes"})
        .sort_values(["omics_type", "unique_genes"])
        .reset_index(drop=True)
    )

    output_path = os.path.join(output_dir, "unique_genes_omics.csv")
    unique_df.to_csv(output_path, index=False)
    print(f"Saved merged unique genes to: {output_path}")

    # Summary Statistics
    total_pairs = len(unique_df)
    total_genes = unique_df["unique_genes"].nunique()
    total_omics = unique_df["omics_type"].nunique()

    print(f"\n📊 Summary: {total_pairs:,} unique (gene, omics) pairs | {total_genes:,} unique genes | {total_omics} omics modalities")

    # PAM50 overlap analysis
    all_genes_in_data = set(combined["gene_name"].unique())
    pam50_found = PAM50 & all_genes_in_data
    pam50_missing = PAM50 - all_genes_in_data

    print(f"\n🎯 PAM50 Overlap: {len(pam50_found)}/{len(PAM50)} genes matched ({len(pam50_missing)} missing)")
    if pam50_found:
        print("  Matched PAM50 genes: " + ", ".join(sorted(pam50_found)))


if __name__ == "__main__":
    args = parse_args()
    merge_and_analyze(folder=args.folder, output_dir=args.output_dir)
