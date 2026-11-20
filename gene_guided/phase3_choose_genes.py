"""Phase 3 - select the top attributed genes (DeepCamCS ``ChooseGenes.py``).

Every sample is attributed once per seed, on the fold where it was in the test set (phase 2). Per (label,
omics), the median IG attribution of every feature over the samples with that label (the predicted label by
default) ranks the features. Outputs (``3_selected_genes_top<K>/``):

    label_<cls>_top_<K>_{positive,negative}.csv   top-K positive and top-K negative features per label
                                                  (n_seeds_selected: in how many seeds' own selection it is)
    all_selected_features.csv                     all of them in one table
    selection_stability_across_seeds.csv          mean pairwise Jaccard overlap of the per-seed top-K sets
    background_features.csv                       every measured feature (the gene universe of phases 6-7)
    report.txt
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

PHASE_DIR = "3_selected_genes_top{k}"
LABEL_KEYS = {"predicted": "pred", "true": "y_true"}


def out_dir(run_dir: Path, k: int) -> Path:
    return Path(run_dir) / PHASE_DIR.format(k=k)


def load_attributions(run_dir: Path, label_by: str = "predicted") -> tuple:
    """(omics, class_names, features {omics: names}, attributions {omics: [N, n_features]}, meta DataFrame)."""
    files = sorted((Path(run_dir) / "2_training").glob("seed_*/fold_*/gene_attributions.npz"))
    if not files:
        raise FileNotFoundError(f"no gene_attributions.npz under {Path(run_dir) / '2_training'} - run phase 2 first")
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


def run(cfg: Config, run_dir: Path, logger: logging.Logger) -> Path:
    k, label_by = cfg.genes.top_k, cfg.genes.label_by
    out = out_dir(run_dir, k)
    out.mkdir(parents=True, exist_ok=True)
    omics, class_names, features, A, meta = load_attributions(run_dir, label_by)
    labels = meta["label"].to_numpy()
    logger.info(f"[genes] {len(meta)} attributed samples ({meta['seed'].nunique()} seeds), omics {omics}, "
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

    lines = [f"# Gene selection (top {k}, grouped by {label_by} label, median IG attribution)", "",
             "Selected features per label x omics (positive + negative):",
             selected.groupby(["predicted_label", "omics_type"])["gene_name"].nunique().unstack().to_string(), "",
             f"Stability across {meta['seed'].nunique()} seeds (mean pairwise Jaccard of the top-K sets):",
             stability.pivot_table(index=["predicted_label", "sign"], columns="omics_type",
                                   values="mean_pairwise_jaccard").round(3).to_string()]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report, encoding="utf-8")
    logger.info(report)
    return out
