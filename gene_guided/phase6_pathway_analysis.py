"""Phase 6 - MSigDB Hallmark enrichment of the selected genes (DeepCamCS ``pathway_analysis.py``).

Over-representation analysis (one-sided hypergeometric test, Benjamini-Hochberg adjusted) of
  * ``all_selected``: the pooled list of every selected gene (all labels, both signs; DeepCamCS's list), and
  * ``<label>_positive``: each label's top-K positive genes (the lists of phase 7),
against the measured genes of ``pathway.omics`` (default mRNA) as the universe. Gene sets are restricted to
the universe and tested if they keep >= ``min_size`` genes.

Differences from DeepCamCS: the test is computed here, not by the Enrichr web service, so no gene list leaves
the machine; and the universe is the measured genes, not the genome (a list drawn from ~3,200 measured genes
looks enriched for anything common among measured breast-tissue genes when tested against the genome). The
gene-set library is downloaded from Enrichr once (``gseapy.get_library``) and cached as a .gmt file in
``cache/``; ``pathway.gmt_file`` uses a local file instead.

Outputs (``6_pathway/``): ``<gene_set>_<list>.csv`` (every tested set) and ``summary.csv``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .config import Config
from .phase3_choose_genes import out_dir as genes_dir
from .utils import gene_symbol

PHASE_DIR = "6_pathway"
ROOT = Path(__file__).resolve().parents[1]


def read_gmt(path: Path) -> dict:
    sets = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def write_gmt(sets: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f"{k}\t\t" + "\t".join(v) + "\n" for k, v in sets.items()), encoding="utf-8")


def load_gene_sets(cfg: Config, logger: logging.Logger) -> dict:
    p = cfg.pathway
    if p.gmt_file:
        path = Path(p.gmt_file) if Path(p.gmt_file).is_absolute() else ROOT / p.gmt_file
        logger.info(f"[pathway] gene sets from {path}")
        return read_gmt(path)
    cached = ROOT / "cache" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', p.gene_set)}.gmt"
    if not cached.exists():
        import gseapy                                   # only needed for the one-time download
        logger.info(f"[pathway] downloading {p.gene_set} from Enrichr (the gene-set library only) -> {cached}")
        write_gmt(gseapy.get_library(name=p.gene_set, organism="Human"), cached)
    return read_gmt(cached)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def over_representation(genes: set, universe: set, gene_sets: dict, min_size: int = 5) -> pd.DataFrame:
    """One-sided hypergeometric test of every gene set (restricted to the universe) for the list ``genes``."""
    genes = genes & universe
    N, n = len(universe), len(genes)
    rows = []
    for term, members in gene_sets.items():
        S = {g.upper() for g in members} & universe
        if len(S) < min_size:
            continue
        hit = sorted(genes & S)
        rows.append({"Term": term, "overlap": len(hit), "set_size_in_universe": len(S), "list_size": n,
                     "universe_size": N, "expected": n * len(S) / N if N else np.nan,
                     "p_value": float(hypergeom.sf(len(hit) - 1, N, len(S), n)), "genes": ";".join(hit)})
    df = pd.DataFrame(rows, columns=["Term", "overlap", "set_size_in_universe", "list_size", "universe_size",
                                     "expected", "p_value", "genes"])
    df.insert(df.columns.get_loc("p_value") + 1, "p_adjusted", bh_adjust(df["p_value"].to_numpy()))
    return df.sort_values(["p_adjusted", "p_value"]).reset_index(drop=True)


def run(cfg: Config, run_dir: Path, logger: logging.Logger) -> Path | None:
    p = cfg.pathway
    src = genes_dir(run_dir, cfg.genes.top_k)
    if not (src / "all_selected_features.csv").exists():
        raise FileNotFoundError(f"{src} not found - run phase 3 first")
    try:
        gene_sets = load_gene_sets(cfg, logger)
    except Exception as e:                                      # noqa: BLE001  (offline, gseapy missing, ...)
        logger.info(f"[pathway] gene sets unavailable ({type(e).__name__}: {e}) - phase skipped. Set "
                    f"pathway.gmt_file to a local .gmt file, or install gseapy and connect once to download it.")
        return None
    out = Path(run_dir) / PHASE_DIR
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(src / "all_selected_features.csv")
    background = pd.read_csv(src / "background_features.csv")
    selected = selected[selected["omics_type"].isin(p.omics)]
    universe = {s for s in background.loc[background["omics_type"].isin(p.omics), "gene_name"].map(gene_symbol) if s}
    if not universe:
        logger.info(f"[pathway] no gene symbols among the features of {p.omics} - phase skipped")
        return None
    symbols = selected.assign(symbol=selected["gene_name"].map(gene_symbol)).dropna(subset=["symbol"])
    lists = {"all_selected": set(symbols["symbol"])}
    for cname, part in symbols[symbols["sign"] == "positive"].groupby("predicted_label"):
        lists[f"{cname}_positive"] = set(part["symbol"])

    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", p.gene_set if not p.gmt_file else Path(p.gmt_file).stem)
    summary = []
    for name, genes in lists.items():
        res = over_representation(genes, universe, gene_sets, p.min_size)
        res.to_csv(out / f"{tag}_{name}.csv", index=False)
        sig = res[res["p_adjusted"] <= p.padj]
        summary.append({"list": name, "n_genes": len(genes & universe), "n_sets_tested": len(res),
                        "n_significant": len(sig), "top_term": res["Term"].iloc[0] if len(res) else "",
                        "top_p_adjusted": float(res["p_adjusted"].iloc[0]) if len(res) else np.nan,
                        "significant_terms": "; ".join(sig["Term"])})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    logger.info(f"[pathway] {tag}, universe {len(universe)} measured genes ({', '.join(p.omics)}), "
                f"BH-adjusted p <= {p.padj:g}:\n" + summary.drop(columns=["significant_terms"]).to_string(index=False))
    return out
