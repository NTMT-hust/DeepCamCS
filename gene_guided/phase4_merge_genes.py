"""Phase 4 - merge the selected genes of all labels and omics, and check the overlap with PAM50
(DeepCamCS ``merge_genes_omics.py``).

Outputs (``4_merged_genes/``):

    unique_genes_omics.csv   one row per selected (feature, omics) pair: in which label x sign lists it is
    pam50_overlap.csv        every PAM50 gene measured in a gene-level omics: selected or not, and where
    report.txt

PAM50 genes are matched by gene symbol, parsed from feature IDs such as ``ESR1|2099``. Methylation probes
(``cg...``) and CNV segments (``chr...``) are not gene symbols, so the check only applies to gene-level omics.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .phase3_choose_genes import out_dir as genes_dir
from .utils import gene_symbol

PHASE_DIR = "4_merged_genes"
PAM50 = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF", "CEP55",
    "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17",
    "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
}


def run(cfg: Config, run_dir: Path, logger: logging.Logger) -> Path:
    src = genes_dir(run_dir, cfg.genes.top_k)
    if not (src / "all_selected_features.csv").exists():
        raise FileNotFoundError(f"{src} not found - run phase 3 first")
    out = Path(run_dir) / PHASE_DIR
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(src / "all_selected_features.csv")
    background = pd.read_csv(src / "background_features.csv")
    selected["list"] = selected["predicted_label"] + " " + selected["sign"]

    unique = (selected.groupby(["omics_type", "gene_name"])
              .agg(n_lists=("list", "nunique"), lists=("list", lambda s: "; ".join(sorted(set(s)))))
              .reset_index().rename(columns={"gene_name": "unique_genes"}))
    unique = unique[["unique_genes", "omics_type", "n_lists", "lists"]].sort_values(["omics_type", "unique_genes"])
    unique.to_csv(out / "unique_genes_omics.csv", index=False)

    symbols = selected.assign(symbol=selected["gene_name"].map(gene_symbol))
    rows, lines = [], [f"# Merged gene selection (top {cfg.genes.top_k} per label x omics x sign)", "",
                       f"{len(unique)} unique (feature, omics) pairs | {unique['omics_type'].nunique()} omics"]
    lines += [f"  {o:<12}: {n}" for o, n in unique["omics_type"].value_counts().items()]
    lines += ["", "PAM50 overlap:"]
    for o, feats in background.groupby("omics_type")["gene_name"]:
        measured = PAM50 & ({gene_symbol(f) for f in feats} - {None})
        if not measured:
            lines.append(f"  {o:<12}: features are not gene symbols (e.g. {feats.iloc[0]}) -> not applicable")
            continue
        sel_o = symbols[symbols["omics_type"] == o]
        found = PAM50 & set(sel_o["symbol"].dropna())
        lines.append(f"  {o:<12}: {len(found)}/{len(measured)} PAM50 genes measured here were selected "
                     f"({len(PAM50) - len(measured)} of the 50 are not in the data) -> {', '.join(sorted(found))}")
        for cname, part in sel_o.groupby("predicted_label"):
            fc = PAM50 & set(part.loc[part["sign"] == "positive", "symbol"].dropna())
            lines.append(f"      {cname:<8} positive: {len(fc):>2}  {', '.join(sorted(fc))}")
        for g in sorted(measured):
            hit = sel_o[sel_o["symbol"] == g]
            rows.append({"omics_type": o, "pam50_gene": g, "selected": not hit.empty,
                         "lists": "; ".join(sorted(set(hit["list"])))})
    pd.DataFrame(rows, columns=["omics_type", "pam50_gene", "selected", "lists"]).to_csv(out / "pam50_overlap.csv",
                                                                                      index=False)
    report = "\n".join(lines)
    (out / "report.txt").write_text(report, encoding="utf-8")
    logger.info(report)
    return out
