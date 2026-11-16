"""Stage 6 - gene selection from the Integrated-Gradients attributions of a run.

Every sample is attributed once per seed, on the fold where it was in the test set. Per (label, omics),
the median attribution of every feature over the samples with that label gives a ranking:

1. top-K positive and top-K negative features per label  -> label_<cls>_top_<K>_{positive,negative}.csv
2. union of the selected (feature, omics) pairs           -> unique_genes_omics.csv
3. overlap with the PAM50 genes (gene symbols parsed from IDs such as ``ESR1|2099``)
4. stability: the same selection per seed, the mean pairwise Jaccard overlap of the top-K sets across
   seeds, and for every selected feature the number of seeds that also select it
5. every measured feature per omics (the gene universe of the GO / KEGG step) -> background_features.csv
A text summary goes to report.txt. GO / KEGG enrichment is ``scripts/go_kegg_enrichment.R``.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PAM50 = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF", "CEP55",
    "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17",
    "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
}
LABEL_KEYS = {"predicted": "pred_raw", "predicted_calibrated": "pred_calibrated", "true": "y_true"}


def gene_symbol(feature: str) -> str | None:
    s = str(feature).split("|")[0].strip()
    return None if s in ("", "?") else s.upper()


def load_attributions(run_dir: Path, label_by: str = "predicted") -> tuple:
    """(omics, class_names, features {omics: names}, attributions {omics: [N, n_features]}, meta DataFrame)."""
    files = sorted(Path(run_dir).glob("seed_*/fold_*/gene_attributions.npz"))
    if not files:
        raise FileNotFoundError(f"no gene_attributions.npz under {run_dir}")
    first = np.load(files[0], allow_pickle=False)
    omics = [str(o) for o in first["omics_names"]]
    class_names = [str(c) for c in first["class_names"]]
    features = {o: first[f"features_{o}"].astype(str) for o in omics}
    attrs, meta = {o: [] for o in omics}, []
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta.append(pd.DataFrame({"seed": int(z["seed"]), "fold": int(z["fold"]), "sample_id": z["sample_ids"].astype(str),
                                  "label": z[LABEL_KEYS[label_by]]}))
        for o in omics:
            if not np.array_equal(z[f"features_{o}"].astype(str), features[o]):
                raise ValueError(f"feature order differs in {f}")
            attrs[o].append(z[f"attr_{o}"])
    return omics, class_names, features, {o: np.concatenate(a) for o, a in attrs.items()}, pd.concat(meta, ignore_index=True)


def median_by_label(A: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((n_classes, A.shape[1]), np.nan)
    for c in range(n_classes):
        if (labels == c).any():
            out[c] = np.median(A[labels == c], axis=0)
    return out


def select_topk(medians: dict, features: dict, class_names: list, k: int) -> pd.DataFrame:
    """Top-K positive and top-K negative features per (label, omics)."""
    rows = []
    for o, M in medians.items():
        for c, cname in enumerate(class_names):
            s = M[c]
            pos = np.flatnonzero(s > 0)
            neg = np.flatnonzero(s < 0)
            for sign, idx in (("positive", pos[np.argsort(-s[pos], kind="stable")][:k]),
                              ("negative", neg[np.argsort(s[neg], kind="stable")][:k])):
                rows += [{"predicted_label": cname, "omics_type": o, "gene_name": features[o][j],
                          "attribute_score": float(s[j]), "sign": sign, "rank": rank} for rank, j in enumerate(idx, 1)]
    return pd.DataFrame(rows)


def stability_across_seeds(A: dict, meta: pd.DataFrame, features: dict, class_names: list, k: int) -> tuple:
    """Per-seed selections -> (per-seed sets, table of mean pairwise Jaccard per label x omics x sign)."""
    labels = meta["label"].to_numpy()
    seeds = sorted(meta["seed"].unique())
    sets = {}
    for s in seeds:
        sel = (meta["seed"] == s).to_numpy()
        med = {o: median_by_label(A[o][sel], labels[sel], len(class_names)) for o in A}
        top = select_topk(med, features, class_names, k)
        sets[s] = {key: set(g["gene_name"]) for key, g in top.groupby(["predicted_label", "omics_type", "sign"])}
    rows = []
    for key in sorted({key for d in sets.values() for key in d}):
        per_seed = [sets[s].get(key, set()) for s in seeds]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(per_seed, 2) if a | b]
        rows.append({"predicted_label": key[0], "omics_type": key[1], "sign": key[2],
                     "mean_pairwise_jaccard": float(np.mean(jac)) if jac else np.nan,
                     "n_selected_all_seeds": len(set.intersection(*per_seed))})
    return sets, pd.DataFrame(rows)


def select_genes(run_dir: Path, k: int = 400, label_by: str = "predicted", out_dir: Path | None = None,
                 log=print) -> Path:
    """Write the gene-selection folder of a run (default ``<run>/gene_selection_top<K>``) and return it."""
    out = Path(out_dir) if out_dir else Path(run_dir) / f"gene_selection_top{k}"
    out.mkdir(parents=True, exist_ok=True)
    omics, class_names, features, A, meta = load_attributions(run_dir, label_by)
    labels = meta["label"].to_numpy()
    log(f"[genes] {len(meta)} attributed samples ({meta['seed'].nunique()} seeds), omics {omics}, "
        f"grouped by {label_by} label")

    selected = select_topk({o: median_by_label(A[o], labels, len(class_names)) for o in omics}, features, class_names, k)
    per_seed, stability = stability_across_seeds(A, meta, features, class_names, k)
    selected["n_seeds_selected"] = [sum(r.gene_name in d.get((r.predicted_label, r.omics_type, r.sign), set())
                                        for d in per_seed.values()) for r in selected.itertuples()]

    cols = ["predicted_label", "omics_type", "gene_name", "attribute_score", "rank", "n_seeds_selected"]
    for cname in class_names:
        for sign in ("positive", "negative"):
            part = selected[(selected["predicted_label"] == cname) & (selected["sign"] == sign)]
            part[cols].to_csv(out / f"label_{cname}_top_{k}_{sign}.csv", index=False)
    selected.to_csv(out / "all_selected_features.csv", index=False)
    stability.to_csv(out / "selection_stability_across_seeds.csv", index=False)
    pd.DataFrame([(o, f) for o in omics for f in features[o]], columns=["omics_type", "gene_name"]) \
        .to_csv(out / "background_features.csv", index=False)
    unique = (selected[["gene_name", "omics_type"]].drop_duplicates().rename(columns={"gene_name": "unique_genes"})
              .sort_values(["omics_type", "unique_genes"]).reset_index(drop=True))
    unique.to_csv(out / "unique_genes_omics.csv", index=False)

    report = _report(selected, unique, stability, features, omics, class_names, k, label_by)
    (out / "report.txt").write_text(report, encoding="utf-8")
    log(report)
    return out


def _report(selected, unique, stability, features, omics, class_names, k, label_by) -> str:
    lines = [f"# Gene selection (top {k}, grouped by {label_by} label, median IG attribution)", "",
             f"Unique (feature, omics) pairs: {len(unique)}"]
    lines += [f"  {o:<12}: {n}" for o, n in unique["omics_type"].value_counts().items()]
    lines += ["", "Unique features per label x omics:",
              selected.groupby(["predicted_label", "omics_type"])["gene_name"].nunique().unstack().to_string(),
              "", "PAM50 overlap (gene symbols parsed from the feature IDs):"]
    symbols = selected.assign(symbol=selected["gene_name"].map(gene_symbol))
    for o in omics:
        available = PAM50 & ({gene_symbol(f) for f in features[o]} - {None})
        if not available:
            lines.append(f"  {o:<12}: features are not gene symbols (e.g. {features[o][0]}) -> not applicable")
            continue
        found = PAM50 & set(symbols.loc[symbols["omics_type"] == o, "symbol"].dropna())
        lines.append(f"  {o:<12}: {len(found)}/{len(available)} PAM50 genes in the data were selected -> "
                     f"{', '.join(sorted(found))}")
        for cname in class_names:
            in_class = symbols[(symbols["omics_type"] == o) & (symbols["predicted_label"] == cname)]
            fc = PAM50 & set(in_class["symbol"].dropna())
            lines.append(f"      {cname:<8}: {len(fc):>2}  {', '.join(sorted(fc))}")
    lines += ["", "Stability across seeds (mean pairwise Jaccard of the top-K sets):",
              stability.pivot_table(index=["predicted_label", "sign"], columns="omics_type",
                                    values="mean_pairwise_jaccard").round(3).to_string()]
    return "\n".join(lines)
