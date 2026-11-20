"""Phase 2 - train and evaluate the guided CNN, then the Integrated-Gradients attributions of the test samples
(DeepCamCS ``train.py``).

Per fold, from the phase-1 images: train (``training.train_fold``) -> SWA fold model -> predict the validation
and test splits (raw softmax) -> IG of every test sample at every gene's pixel. Then the results of all folds
and seeds are aggregated. Outputs (``2_training/``):

    summary.md / summary.json        mean +- std over fold models and seeds, multi-seed ensemble, mask check
    fold_metrics.csv / seed_metrics.csv / oof_predictions_all_seeds.csv / mask_check.csv
    seed_<s>/oof_predictions.csv     every sample's test prediction for that seed
    seed_<s>/fold_<f>/               predictions.csv, history.csv, fold_result.json, model.pt,
                                     gene_attributions.npz, guidance_mask.npy

Finished folds are loaded instead of retrained, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from .attribution import gene_level_attributions, integrated_gradients
from .config import Config
from .data import FoldSplit
from .guidance import RANK_BINS
from .metrics import EVAL_METRICS, KEY_METRICS, METRIC_LABELS, PERCENT_METRICS, compute_metrics, fmt, format_confusion
from .phase1_images import Dataset, load_fold_images, load_splits
from .training import eval_loader, predict_logits, to_float_image, train_fold
from .utils import fold_seed, load_json, save_json, seed_everything

PHASE_DIR = "2_training"
EXTRA_FIRST = KEY_METRICS[len(EVAL_METRICS)]      # first of the "additional" metrics in the report


def run(cfg: Config, run_dir: Path, device: torch.device, logger: logging.Logger) -> dict:
    ds = Dataset.load(run_dir)
    logger.info(f"setup   : {describe_setup(cfg)}")
    t0 = time.time()
    results = {}
    for seed in cfg.splits.seeds:
        results[int(seed)] = _seed(cfg, ds, int(seed), run_dir, device, logger)
        logger.info(f"[elapsed {(time.time() - t0) / 60:.1f} min]")
    return aggregate(cfg, ds.class_names, Path(run_dir) / PHASE_DIR, results, logger)


# ==============================================================================================
# One fold
# ==============================================================================================
def run_fold(cfg: Config, ds: Dataset, split: FoldSplit, run_dir: Path, device: torch.device,
             logger: logging.Logger) -> dict:
    seed, fold = split.seed, split.fold
    out = Path(run_dir) / PHASE_DIR / f"seed_{seed}" / f"fold_{fold + 1}"
    if (out / "fold_result.json").exists():
        logger.info(f"  seed {seed} fold {fold + 1}: already done -> loaded")
        return load_json(out / "fold_result.json")
    out.mkdir(parents=True, exist_ok=True)
    y, names = ds.labels, ds.class_names
    timings, t_start = {}, time.time()
    logger.info(f"\n{'=' * 22} SEED {seed} | FOLD {fold + 1}/{cfg.splits.n_folds} {'=' * 22}")
    for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        counts = np.bincount(y[idx], minlength=len(names))
        logger.info(f"  {name:<5} n={len(idx):>4}  " + " ".join(f"{c}={n}" for c, n in zip(names, counts)))
    seed_everything(fold_seed(seed, fold), cfg.train.deterministic)
    images, pixels, relevance = load_fold_images(run_dir, seed, fold, ds.omics_names)

    t0 = time.time()
    tr = train_fold(cfg, images, y, split.train, split.val, names, pixels, relevance if cfg.guidance.enabled else None,
                    device, fold_seed(seed, fold), logger)
    model = tr["model"]
    timings["train_s"] = time.time() - t0

    bs = cfg.train.eval_batch_size
    logits = {s: predict_logits(model, eval_loader(images[idx], y[idx], bs), device)[0]
              for s, idx in (("val", split.val), ("test", split.test))}
    if not all(np.isfinite(v).all() for v in logits.values()):
        raise RuntimeError(f"seed {seed} fold {fold + 1}: training diverged (non-finite outputs); "
                           "lower train.lr or guidance.lambda_reg")
    probs = {s: softmax(v, 1) for s, v in logits.items()}
    truth = {"val": y[split.val], "test": y[split.test]}
    metrics = {s: compute_metrics(truth[s], probs[s].argmax(1), probs[s], names) for s in probs}
    for s in ("val", "test"):
        logger.info(f"  {s.upper():<4} : {fmt(metrics[s])}")
    logger.info("  TEST confusion:\n" + format_confusion(metrics["test"]["confusion_matrix"], names))
    _predictions_frame(ds, split, probs).to_csv(out / "predictions.csv", index=False)
    hist = pd.DataFrame(tr["history"])
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(out / "history.csv", index=False)

    t0 = time.time()
    pred_test = probs["test"].argmax(1)
    attr = _attribute(model, images[split.test], pred_test, device, cfg)
    arrays = {f"attr_{o}": a for o, a in zip(ds.omics_names, gene_level_attributions(attr, pixels))}
    arrays.update({f"features_{o}": ds.features[o] for o in ds.omics_names})
    np.savez_compressed(out / "gene_attributions.npz", sample_ids=ds.sample_ids[split.test], y_true=truth["test"],
                        pred=pred_test, class_names=np.array(names), omics_names=np.array(ds.omics_names),
                        seed=seed, fold=fold + 1, **arrays)
    timings["attribution_s"] = time.time() - t0
    logger.info(f"  [attribution] IG ({cfg.attribution.ig_steps} steps) on {len(split.test)} test samples "
                f"in {timings['attribution_s']:.1f}s")
    if cfg.attribution.save_checkpoints:
        torch.save(model.state_dict(), out / "model.pt")
    if tr["final_mask"] is not None:
        np.save(out / "guidance_mask.npy", tr["final_mask"].cpu().numpy())

    timings["total_s"] = time.time() - t_start
    record = {"seed": seed, "fold": fold + 1, "n_train": len(split.train), "n_val": len(split.val),
              "n_test": len(split.test), "status": "diverged" if tr.get("divergence") else "completed",
              "divergence": tr.get("divergence"), "metrics": metrics, "mask_log": tr["mask_log"], "timings": timings}
    if tr.get("divergence"):
        logger.warning(f"  fold marked diverged at epoch {tr['divergence']['epoch']}; test metrics use last usable weights")
    save_json(record, out / "fold_result.json")
    logger.info(f"  fold done in {timings['total_s'] / 60:.1f} min")
    del model, tr, images
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _predictions_frame(ds: Dataset, split: FoldSplit, probs: dict) -> pd.DataFrame:
    frames = []
    for s, idx in (("val", split.val), ("test", split.test)):
        y = ds.labels[idx]
        df = pd.DataFrame({"sample_id": ds.sample_ids[idx], "seed": split.seed, "fold": split.fold + 1, "split": s,
                           "y_true": y, "true_label": np.array(ds.class_names)[y], "pred": probs[s].argmax(1)})
        for c, name in enumerate(ds.class_names):
            df[f"prob_{name}"] = probs[s][:, c]
        frames.append(df)
    return pd.concat(frames)


def _attribute(model, images_u8: torch.Tensor, targets: np.ndarray, device, cfg: Config) -> np.ndarray:
    """IG of the predicted class for every image, 16 images at a time."""
    max_batch = max(1, cfg.attribution.max_batch // model.memory_factor)
    targets = torch.as_tensor(targets, dtype=torch.long)
    out = []
    for s0 in range(0, images_u8.size(0), 16):
        x = to_float_image(images_u8[s0:s0 + 16]).to(device)
        out.append(integrated_gradients(model, x, targets[s0:s0 + 16].to(device), cfg.attribution.ig_steps,
                                        max_batch).cpu().numpy())
    return np.concatenate(out)


# ==============================================================================================
# Seeds and aggregation
# ==============================================================================================
def _seed(cfg: Config, ds: Dataset, seed: int, run_dir: Path, device, logger) -> dict:
    records = [run_fold(cfg, ds, sp, run_dir, device, logger) for sp in load_splits(run_dir, seed)]
    seed_dir = Path(run_dir) / PHASE_DIR / f"seed_{seed}"
    preds = pd.concat([pd.read_csv(seed_dir / f"fold_{r['fold']}" / "predictions.csv") for r in records])
    oof = preds[preds["split"] == "test"].reset_index(drop=True)
    oof.to_csv(seed_dir / "oof_predictions.csv", index=False)
    pooled = _oof_metrics(oof, ds.class_names)
    logger.info(f"\n### SEED {seed} pooled out-of-fold TEST (n={len(oof)}): {fmt(pooled)}")
    return {"records": records, "pooled": pooled, "oof": oof}


def _oof_metrics(oof: pd.DataFrame, class_names: list) -> dict:
    P = oof[[f"prob_{c}" for c in class_names]].to_numpy()
    return compute_metrics(oof["y_true"].to_numpy(), oof["pred"].to_numpy(), P, class_names)


def _mean_std(df: pd.DataFrame, prefix: str) -> dict:
    return {k: {"mean": float(df[f"{prefix}{k}"].mean()),
                "std": float(df[f"{prefix}{k}"].std(ddof=1)) if len(df) > 1 else 0.0} for k in KEY_METRICS}


def aggregate(cfg: Config, class_names: list, out: Path, seed_results: dict, logger) -> dict:
    records = [r for res in seed_results.values() for r in res["records"]]
    folds = pd.DataFrame([{"seed": r["seed"], "fold": r["fold"], "status": r.get("status", "completed"),
                           "divergence_epoch": (r.get("divergence") or {}).get("epoch"),
                           **{f"{s}_{k}": r["metrics"][s].get(k) for s in ("test", "val") for k in KEY_METRICS}}
                          for r in records]).sort_values(["seed", "fold"]).reset_index(drop=True)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    seeds = pd.DataFrame([{"seed": s, **{f"oof_{k}": res["pooled"].get(k) for k in KEY_METRICS}}
                          for s, res in seed_results.items()])
    seeds.to_csv(out / "seed_metrics.csv", index=False)

    # multi-seed ensemble: every sample is out-of-fold in every seed -> average its probabilities
    all_oof = pd.concat([res["oof"] for res in seed_results.values()])
    all_oof.to_csv(out / "oof_predictions_all_seeds.csv", index=False)
    ensemble = {}
    if len(seed_results) > 1:
        grouped = all_oof.groupby("sample_id")
        y = grouped["y_true"].first()
        P = grouped[[f"prob_{c}" for c in class_names]].mean().loc[y.index].to_numpy()
        ensemble = compute_metrics(y.to_numpy(), P.argmax(1), P, class_names)

    summary = {"seeds": list(seed_results), "n_folds": cfg.splits.n_folds, "n_fold_models": len(records),
               "n_diverged_folds": sum(r.get("status") == "diverged" for r in records),
               "diverged_folds": [{"seed": r["seed"], "fold": r["fold"], **(r.get("divergence") or {})}
                                  for r in records if r.get("status") == "diverged"],
               "fold_level": _mean_std(folds, "test_"), "seed_level_pooled_oof": _mean_std(seeds, "oof_"),
               "multi_seed_ensemble": ensemble}
    checks = pd.DataFrame([{"seed": r["seed"], "fold": r["fold"], "omics": om, **d}
                           for r in records for om, d in _last_mask_check(r).items()])
    if len(checks):
        checks.to_csv(out / "mask_check.csv", index=False)
        summary["mask_check"] = checks.drop(columns=["seed", "fold"]).groupby("omics", sort=False).mean().to_dict("index")
    save_json(summary, out / "summary.json")
    md = summary_markdown(summary, seeds, cfg)
    (out / "summary.md").write_text(md, encoding="utf-8")
    logger.info("\n" + md)
    return summary


def _last_mask_check(record: dict) -> dict:
    checks = [m["check"] for m in record.get("mask_log", []) if m.get("check")]
    return checks[-1] if checks else {}


# ==============================================================================================
# Report
# ==============================================================================================
def describe_setup(cfg: Config) -> str:
    m, g, t = cfg.model, cfg.guidance, cfg.train
    warm = t.warmup_epochs if t.epochs > t.warmup_epochs else 0
    text = (f"Model: {m.arch} (ImageNet-pretrained), one trunk per omics, gene readout of feature levels "
            f"{sorted(m.readout_levels)}; focal loss, class-aware augmentation; SGD (lr {t.lr:g}"
            + (f", {warm}-epoch warm-up" if warm else "") + f", cosine to epoch {t.epochs}); fold model = average of "
            f"the weights after epochs {t.swa_start_epoch + 1}-{t.epochs} of that same run (SWA). Images: snowfall "
            f"layout fitted on the training split, 0 reserved for empty pixels. Raw softmax probabilities.")
    if not g.enabled:
        return text + " Guidance: off."
    a, b = g.penalty_pair
    return (text + f" Guidance: Grad-CAM mask on the validation split (top {g.topk_ratio * 100:g}% wrong-reason gene "
            f"pixels, frequency > {g.frequency_threshold:g}; gene pixels only; one mask per class; top "
            f"{g.protect_top_fraction * 100:g}% genes by ANOVA never masked); penalty on ∇ₓ [log p({a}|x) − log "
            f"p({b}|x)] for {a}/{b} samples, ∇ₓ log p(y|x) otherwise, λ = {g.lambda_reg:g}, from epoch "
            f"{g.loss_start_epoch + 1}, on {g.penalty_samples} random samples per batch.")


def summary_markdown(summary: dict, seeds: pd.DataFrame, cfg: Config) -> str:
    def cell(stats: dict, k: str) -> str:
        m, s = stats[k]["mean"], stats[k]["std"]
        if m is None or not np.isfinite(m):
            return "n/a"
        s = s if s is not None and np.isfinite(s) else 0.0
        return f"{m * 100:.2f} ± {s * 100:.2f}" if k in PERCENT_METRICS else f"{m:.4f} ± {s:.4f}"

    def label(k: str) -> str:
        return METRIC_LABELS[k] + (" (%)" if k in PERCENT_METRICS else "")

    def value(v, k: str) -> str:
        if v is None or not np.isfinite(v):
            return "n/a"
        return f"{v * 100:.2f}" if k in PERCENT_METRICS else f"{v:.4f}"

    lines = [f"# Results - {len(summary['seeds'])} seeds × {summary['n_folds']} folds ({summary['n_fold_models']} models)",
             "", "Outer stratified K-fold (test), inner stratified train / validation split. Seeds: "
             + ", ".join(map(str, summary["seeds"])) + ".", describe_setup(cfg), "",
             f"Diverged folds: {summary.get('n_diverged_folds', 0)}. Test metrics for those folds use the last usable weights after training stopped.", "",
             "## Test metrics", "", "| metric | mean ± std over fold models | pooled out-of-fold, mean ± std over seeds |",
             "|---|---|---|"]
    for k in KEY_METRICS:
        if k == EXTRA_FIRST:
            lines.append("| **additional** | | |")
        lines.append(f"| {label(k)} | {cell(summary['fold_level'], k)} | {cell(summary['seed_level_pooled_oof'], k)} |")
    lines.append("")
    if summary.get("diverged_folds"):
        lines += ["## Diverged folds", "", "| seed | fold | epoch | learning rate | penalty | error |", "|---:|---:|---:|---:|:---:|---|"]
        for d in summary["diverged_folds"]:
            lines.append(f"| {d['seed']} | {d['fold']} | {d.get('epoch', 'n/a')} | {d.get('lr', float('nan')):.3g} | "
                         f"{'on' if d.get('penalty') else 'off'} | {d.get('error', '')} |")
        lines.append("")
    if summary["multi_seed_ensemble"]:
        e = summary["multi_seed_ensemble"]
        lines += ["## Multi-seed ensemble (mean of each sample's out-of-fold probabilities over seeds)", "",
                  "| metric | value |", "|---|---|"]
        lines += [f"| {label(k)} | {value(e[k], k)} |" for k in EVAL_METRICS + ["nll", "ece"]]
        lines.append("")
    lines += ["## Per-seed pooled out-of-fold test", "",
              "| seed | " + " | ".join(METRIC_LABELS[k] for k in EVAL_METRICS) + " |", "|---" * (len(EVAL_METRICS) + 1) + "|"]
    for _, r in seeds.iterrows():
        lines.append(f"| {int(r['seed'])} | " + " | ".join(value(r[f'oof_{k}'], k) for k in EVAL_METRICS) + " |")
    lines.append("")
    if summary.get("mask_check"):
        bins = [b for _, _, b in RANK_BINS]
        lines += ["## Guidance mask check (final mask of each fold model, mean)", "",
                  "Share of each omics' features inside the mask, by ANOVA-F rank on the training split. A mask of "
                  "wrong reasons should cover the top-ranked features much less than the rest.", "",
                  "| omics | " + " | ".join(bins) + " | all |", "|---" * (len(bins) + 2) + "|"]
        for om, d in summary["mask_check"].items():
            lines.append(f"| {om} | " + " | ".join(f"{d[b] * 100:.0f}%" for b in bins) + f" | {d['all'] * 100:.0f}% |")
        lines.append("")
    return "\n".join(lines)
