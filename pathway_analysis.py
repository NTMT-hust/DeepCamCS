import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Perform pathway enrichment analysis on unique selected genes.")
    parser.add_argument("--gene_file", type=str, default="Top/unique_genes_omics.csv", help="Path to CSV containing unique genes.")
    parser.add_argument("--gene_column", type=str, default="unique_genes", help="Column name containing gene symbols.")
    parser.add_argument("--gene_set", type=str, default="MSigDB_Hallmark_2020", help="Enrichr library name.")
    parser.add_argument("--output_file", type=str, default="hallmark_pathway_overlap.csv", help="Output CSV path for pathway results.")
    return parser.parse_args()


def normalize_gene_symbols(raw_genes):
    """Extract HGNC gene symbols (e.g. 'CDH3|1001' -> 'CDH3') and drop CpG/CNV IDs."""
    symbols = []
    seen = set()
    for item in raw_genes:
        if pd.isna(item):
            continue
        token = str(item).strip()
        if not token:
            continue
        symbol = token.split("|")[0].strip()
        if not symbol or symbol == "?":
            continue
        lower = symbol.lower()
        if lower.startswith("cg") or lower.startswith("chr"):
            continue
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def run_pathway_analysis(gene_file="Top/unique_genes_omics.csv", gene_column="unique_genes", gene_set="MSigDB_Hallmark_2020", output_file="hallmark_pathway_overlap.csv"):
    if not os.path.exists(gene_file):
        print(f"⚠️ Gene file not found: '{gene_file}'")
        return

    df = pd.read_csv(gene_file)
    if gene_column not in df.columns:
        # Fallback to first column
        gene_column = df.columns[0]

    raw_genes = df[gene_column].dropna().unique().tolist()
    genes = normalize_gene_symbols(raw_genes)
    print(
        f"Loaded {len(raw_genes)} unique features ({len(genes)} normalized gene symbols) "
        f"for pathway enrichment ({gene_set})..."
    )

    columns = [
        "Term",
        "overlap_gene_count",
        "total_pathway_genes",
        "Adjusted P-value",
        "gene_list",
    ]

    if not genes:
        print("⚠️ No valid gene symbols found after normalization.")
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
        return

    import gseapy as gp

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_set,
        organism="human",
        outdir=None,
    )
    results = enr.results

    if results is None or results.empty:
        final = pd.DataFrame(columns=columns)
    else:
        results["overlap_gene_count"] = results["Overlap"].apply(lambda x: int(x.split("/")[0]))
        results["total_pathway_genes"] = results["Overlap"].apply(lambda x: int(x.split("/")[1]))
        results["gene_list"] = results["Genes"]

        available_cols = [c for c in columns if c in results.columns]
        final = results[available_cols].sort_values(
            ["overlap_gene_count", "Adjusted P-value"], ascending=[False, True]
        )

    final.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")
    print("\nTop 10 enriched pathways:")
    print(final.head(10))


if __name__ == "__main__":
    args = parse_args()
    run_pathway_analysis(
        gene_file=args.gene_file,
        gene_column=args.gene_column,
        gene_set=args.gene_set,
        output_file=args.output_file,
    )