import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Filter raw omics matrices by extracted unique gene list.")
    parser.add_argument("--unique_genes_file", type=str, default="Top/unique_genes_omics.csv", help="Path to unique_genes_omics.csv.")
    parser.add_argument("--cnv_file", type=str, default="", help="Path to BRCA_CNV_aligned.csv.")
    parser.add_argument("--mrna_file", type=str, default="", help="Path to BRCA_mRNA_aligned.csv.")
    parser.add_argument("--methy_file", type=str, default="", help="Path to BRCA_Methy_aligned.csv.")
    parser.add_argument("--output_dir", type=str, default="Top", help="Output directory for filtered files.")
    return parser.parse_args()


def filter_omics(unique_genes_file, omics_files, output_dir):
    if not os.path.exists(unique_genes_file):
        print(f"⚠️ Unique genes file not found: '{unique_genes_file}'")
        return

    print(f"Loading unique genes from: {unique_genes_file}")
    unique_df = pd.read_csv(unique_genes_file)
    print(f"  Total pairs: {len(unique_df):,}")
    print(f"  Total unique genes: {unique_df['unique_genes'].nunique():,}")

    os.makedirs(output_dir, exist_ok=True)

    for omics_name, filepath in omics_files.items():
        if not filepath or not os.path.exists(filepath):
            continue

        print(f"\n🧬 Filtering {omics_name}: {filepath}")
        df = pd.read_csv(filepath, index_col=0)
        genes_for_omics = set(unique_df.loc[unique_df["omics_type"] == omics_name, "unique_genes"])

        mask = df.index.isin(genes_for_omics)
        df_filtered = df[mask]

        n_found = df_filtered.shape[0]
        n_missing = len(genes_for_omics) - n_found
        print(f"  Retained: {n_found:,} genes | Unmatched: {n_missing:,} genes")

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(output_dir, f"{base_name}_filtered.csv")
        df_filtered.to_csv(out_path)
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    omics_dict = {
        "CNV": args.cnv_file,
        "mRNA": args.mrna_file,
        "Methylation": args.methy_file,
    }
    filter_omics(args.unique_genes_file, omics_dict, args.output_dir)
