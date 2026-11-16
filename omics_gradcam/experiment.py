"""Run the cross-validation experiment: seeds x folds, each fold running stages 2-6, then aggregation.

Per fold (``run_fold``):
  2. fit the image layout on the training split, render every sample
  3-4. train the CNN (with guidance) -> SWA fold model
  5. calibrate on the validation split, predict the test split
  6. Integrated-Gradients gene attributions of the test samples
Finished folds are loaded instead of retrained, so an interrupted run resumes where it stopped.

Outputs (``runs/<name>/``):
  summary.md / summary.json         mean +- std over fold models and seeds, multi-seed ensemble
  fold_metrics.csv / seed_metrics.csv / oof_predictions_all_seeds.csv / mask_check.csv
  seed_<s>/fold_<f>/                predictions.csv, history.csv, fold_result.json, model.pt,
                                    gene_attributions.npz, layout.npz, gene_coordinates_<omics>.csv,
                                    guidance_mask.npy
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
from .calibration import build_calibrator
from .config import Config, load_config
from .data import MultiOmicsData
from .guidance import RANK_BINS, FeatureRelevance
from .imaging import OmicsImageTransformer
from .metrics import EVAL_METRICS, KEY_METRICS, METRIC_LABELS, PERCENT_METRICS, compute_metrics, fmt, format_confusion
from .splits import FoldSplit, make_cv_splits
from .training import eval_loader, predict_logits, to_float_image, train_fold
from .utils import load_json, save_json, seed_everything

STAGES = ("raw", "calibrated")
EXTRA_FIRST = KEY_METRICS[len(EVAL_METRICS)]      # first of the "additional" metrics in the report


# ==============================================================================================
# One fold
# ==============================================================================================
def run_fold(cfg: Config, data: MultiOmicsData, split: FoldSplit, run_dir: Path, device: torch.device,
             logger: logging.Logger) -> dict:
    seed, fold = split.seed, split.fold
    fold_dir = run_dir / f"seed_{seed}" / f"fold_{fold + 1}"
    if (fold_dir / "fold_result.json").exists():
        logger.info(f"  seed {seed} fold {fold + 1}: already done -> loaded")
        return load_json(fold_dir / "fold_result.json")
    fold_dir.mkdir(parents=True, exist_ok=True)
    y, names = data.labels, data.class_names
    timings, t_start = {}, time.time()
    logger.info(f"\n{'=' * 22} SEED {seed} | FOLD {fold + 1}/{cfg.splits.n_folds} {'=' * 22}")
    for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        counts = np.bincount(y[idx], minlength=data.num_classes)
        logger.info(f"  {name:<5} n={len(idx):>4}  " + " ".join(f"{c}={n}" for c, n in zip(names, counts)))
    fold_seed = seed * 100 + fold
    seed_everything(fold_seed, cfg.train.deterministic)

    # ---- 2. omics -> images (layout fitted on the training split only) ------------------------
    t0 = time.time()
    tfm = OmicsImageTransformer(data.omics_names, cfg.image, random_state=seed)
    tfm.fit(data.matrices(split.train), log=logger.info)
    images = torch.from_numpy(tfm.transform(data.matrices()))
    tfm.save(fold_dir / "layout.npz")
    for name, df in tfm.coordinates([o.feature_names for o in data.omics]).items():
        df.to_csv(fold_dir / f"gene_coordinates_{name}.csv", index=False)
    timings["images_s"] = time.time() - t0
    logger.info(f"  [images] {tuple(images.shape)} uint8 in {timings['images_s']:.1f}s")

    # ---- 3-4. CNN training with guidance ---------------------------------------------------------
    t0 = time.time()
    relevance = None
    if cfg.guidance.enabled:     # ANOVA ranks on the training split: protected genes + mask check
        relevance = FeatureRelevance.from_training_split(data.matrices(split.train), y[split.train],
                                                         tfm.pixel_index(), data.omics_names, tuple(images.shape[1:]))
    tr = train_fold(cfg, images, y, split.train, split.val, names, tfm.pixel_index(), relevance, device, fold_seed,
                    logger)
    model = tr["model"]
    timings["train_s"] = time.time() - t0

    # ---- 5. calibration on validation, prediction of the test split --------------------------------
    bs = cfg.train.eval_batch_size
    val_logits, y_val = predict_logits(model, eval_loader(images[split.val], y[split.val], bs), device)
    test_logits, y_test = predict_logits(model, eval_loader(images[split.test], y[split.test], bs), device)
    if not (np.isfinite(val_logits).all() and np.isfinite(test_logits).all()):
        raise RuntimeError(f"seed {seed} fold {fold + 1}: training diverged (non-finite outputs); "
                           "lower train.lr or guidance.lambda_reg")
    cal = build_calibrator(cfg.calibration.method).fit(val_logits, y_val)
    if cfg.calibration.tune_thresholds:
        cal.tune_thresholds(cal.predict_proba(val_logits), y_val)
    probs = {"val": (softmax(val_logits, 1), cal.predict_proba(val_logits)),
             "test": (softmax(test_logits, 1), cal.predict_proba(test_logits))}
    preds = {s: (raw.argmax(1), cal.predict(c)) for s, (raw, c) in probs.items()}
    truth = {"val": y_val, "test": y_test}
    metrics = {s: {"raw": compute_metrics(truth[s], preds[s][0], probs[s][0], names),
                   "calibrated": compute_metrics(truth[s], preds[s][1], probs[s][1], names)} for s in probs}
    for s in ("val", "test"):
        for st in STAGES:
            logger.info(f"  {s.upper():<4} {st[:3]} : {fmt(metrics[s][st])}")
    logger.info("  TEST confusion (calibrated):\n" + format_confusion(metrics["test"]["calibrated"]["confusion_matrix"],
                                                                      names))
    _predictions_frame(data, split, probs, preds).to_csv(fold_dir / "predictions.csv", index=False)
    hist = pd.DataFrame(tr["history"])
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(fold_dir / "history.csv", index=False)

    # ---- 6. gene-level Integrated Gradients on the test split ---------------------------------------
    if cfg.attribution.enabled:
        t0 = time.time()
        attr = _attribute(model, images[split.test], preds["test"][0], device, cfg)
        arrays = {f"attr_{o.name}": a for o, a in zip(data.omics, gene_level_attributions(attr, tfm.pixel_index()))}
        arrays.update({f"features_{o.name}": o.feature_names for o in data.omics})
        np.savez_compressed(fold_dir / "gene_attributions.npz", sample_ids=data.sample_ids[split.test], y_true=y_test,
                            pred_raw=preds["test"][0], pred_calibrated=preds["test"][1],
                            class_names=np.array(names), omics_names=np.array(data.omics_names),
                            seed=seed, fold=fold + 1, **arrays)
        timings["attribution_s"] = time.time() - t0
        logger.info(f"  [attribution] IG ({cfg.attribution.ig_steps} steps) on {len(split.test)} test samples "
                    f"in {timings['attribution_s']:.1f}s")
    if cfg.attribution.save_checkpoints:
        torch.save(model.state_dict(), fold_dir / "model.pt")
    if tr["final_mask"] is not None:
        np.save(fold_dir / "guidance_mask.npy", tr["final_mask"].cpu().numpy())

    timings["total_s"] = time.time() - t_start
    record = {"seed": seed, "fold": fold + 1, "n_train": len(split.train), "n_val": len(split.val),
              "n_test": len(split.test), "metrics": metrics,
              "thresholds": None if cal.thresholds is None else [float(v) for v in cal.thresholds],
              "temperature": getattr(cal, "T", None), "mask_log": tr["mask_log"], "timings": timings}
    save_json(record, fold_dir / "fold_result.json")
    logger.info(f"  fold done in {timings['total_s'] / 60:.1f} min")
    del model, tr, images
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _predictions_frame(data: MultiOmicsData, split: FoldSplit, probs: dict, preds: dict) -> pd.DataFrame:
    frames = []
    for s, idx in (("val", split.val), ("test", split.test)):
        y = data.labels[idx]
        df = pd.DataFrame({"sample_id": data.sample_ids[idx], "seed": split.seed, "fold": split.fold + 1, "split": s,
                           "y_true": y, "true_label": np.array(data.class_names)[y],
                           "pred_raw": preds[s][0], "pred_calibrated": preds[s][1]})
        for stage, p in zip(("raw", "cal"), probs[s]):
            for c, name in enumerate(data.class_names):
                df[f"prob_{stage}_{name}"] = p[:, c]
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
def run_seed(cfg: Config, data: MultiOmicsData, seed: int, run_dir: Path, device, logger) -> dict:
    splits = make_cv_splits(data.labels, seed, cfg.splits.n_folds, cfg.splits.val_fraction)
    save_json([s.as_dict(data.sample_ids) for s in splits], run_dir / f"seed_{seed}" / "splits.json")
    records = [run_fold(cfg, data, sp, run_dir, device, logger) for sp in splits]
    seed_dir = run_dir / f"seed_{seed}"
    preds = pd.concat([pd.read_csv(seed_dir / f"fold_{r['fold']}" / "predictions.csv") for r in records])
    oof = preds[preds["split"] == "test"].reset_index(drop=True)
    oof.to_csv(seed_dir / "oof_predictions.csv", index=False)
    pooled = _oof_metrics(oof, data.class_names)
    logger.info(f"\n### SEED {seed} pooled out-of-fold TEST (n={len(oof)})")
    for st in STAGES:
        logger.info(f"    {st:<10}: {fmt(pooled[st])}")
    return {"records": records, "pooled": pooled, "oof": oof}


def _oof_metrics(oof: pd.DataFrame, class_names: list) -> dict:
    y = oof["y_true"].to_numpy()
    return {st: compute_metrics(y, oof[pred].to_numpy(), oof[[f"{prefix}{c}" for c in class_names]].to_numpy(),
                                class_names)
            for st, pred, prefix in (("raw", "pred_raw", "prob_raw_"), ("calibrated", "pred_calibrated", "prob_cal_"))}


def _fold_table(records: list) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"seed": r["seed"], "fold": r["fold"]}
        for split in ("test", "val"):
            for st in STAGES:
                row.update({f"{split}_{st}_{k}": r["metrics"][split][st].get(k) for k in KEY_METRICS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "fold"]).reset_index(drop=True)


def _mean_std(df: pd.DataFrame, prefix: str) -> dict:
    cols = [f"{prefix}{k}" for k in KEY_METRICS]
    return {k: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)) if len(df) > 1 else 0.0}
            for k, c in zip(KEY_METRICS, cols)}


def aggregate(cfg: Config, class_names: list, run_dir: Path, seed_results: dict, logger) -> dict:
    records = [r for res in seed_results.values() for r in res["records"]]
    folds = _fold_table(records)
    folds.to_csv(run_dir / "fold_metrics.csv", index=False)
    seeds = pd.DataFrame([{"seed": s, **{f"oof_{st}_{k}": res["pooled"][st].get(k) for st in STAGES for k in KEY_METRICS}}
                          for s, res in seed_results.items()])
    seeds.to_csv(run_dir / "seed_metrics.csv", index=False)

    # multi-seed ensemble: every sample is out-of-fold in every seed -> average its probabilities
    all_oof = pd.concat([res["oof"] for res in seed_results.values()])
    all_oof.to_csv(run_dir / "oof_predictions_all_seeds.csv", index=False)
    ensemble = {}
    if len(seed_results) > 1:
        grouped = all_oof.groupby("sample_id")
        y = grouped["y_true"].first()
        for st, prefix in (("raw", "prob_raw_"), ("calibrated", "prob_cal_")):
            P = grouped[[f"{prefix}{c}" for c in class_names]].mean().loc[y.index].to_numpy()
            ensemble[st] = compute_metrics(y.to_numpy(), P.argmax(1), P, class_names)

    summary = {"seeds": list(seed_results), "n_folds": cfg.splits.n_folds, "n_fold_models": len(records),
               "fold_level": {st: _mean_std(folds, f"test_{st}_") for st in STAGES},
               "seed_level_pooled_oof": {st: _mean_std(seeds, f"oof_{st}_") for st in STAGES},
               "multi_seed_ensemble": ensemble}
    checks = pd.DataFrame([{"seed": r["seed"], "fold": r["fold"], "omics": om, **d}
                           for r in records for om, d in _last_mask_check(r).items()])
    if len(checks):
        checks.to_csv(run_dir / "mask_check.csv", index=False)
        summary["mask_check"] = checks.drop(columns=["seed", "fold"]).groupby("omics", sort=False).mean().to_dict("index")
    save_json(summary, run_dir / "summary.json")
    md = summary_markdown(summary, seeds, cfg)
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    logger.info("\n" + md)
    return summary


def _last_mask_check(record: dict) -> dict:
    checks = [m["check"] for m in record.get("mask_log", []) if m.get("check")]
    return checks[-1] if checks else {}


# ==============================================================================================
# Report
# ==============================================================================================
def describe_setup(cfg: Config) -> str:
    m, g = cfg.model, cfg.guidance
    head = ("global average pooling -> linear" if m.head == "gap" else
            f"gene readout of feature levels {sorted(m.readout_levels)}, one trunk per omics")
    text = (f"Model: {m.arch} (ImageNet-pretrained), head = {head}; focal loss, class-aware augmentation; "
            f"AdamW + cosine learning rate for epochs 1-{cfg.train.swa_start_epoch}, then SWA (SGD at constant lr "
            f"{cfg.train.swa_lr:g}) for epochs {cfg.train.swa_start_epoch + 1}-{cfg.train.epochs}; fold model = the "
            f"averaged weights. "
            f"Images: snowfall layout fitted on the training split"
            + (", 0 reserved for empty pixels" if cfg.image.reserve_zero else "") + ".")
    if not g.enabled:
        return text + " Guidance: off."
    a, b = g.penalty_pair
    target = {"true_class": "∇ₓ log p(y|x)", "all_classes": "∇ₓ Σ_c log p(c|x)",
              "pair": f"∇ₓ [log p({a}|x) − log p({b}|x)] for {a}/{b} samples, ∇ₓ log p(y|x) otherwise"}[g.penalty_target]
    opts = [f"top {g.topk_ratio * 100:g}% wrong-reason gene pixels, frequency > {g.frequency_threshold:g}",
            "background included" if g.mask_background else "gene pixels only"]
    if g.class_conditional:
        opts.append("one mask per class")
    if g.protect_top_fraction > 0:
        opts.append(f"top {g.protect_top_fraction * 100:g}% genes by ANOVA never masked")
    return (text + f" Guidance: {g.mask_method} mask on the {g.mask_source} split ({'; '.join(opts)}); penalty on "
            f"{target}, λ = {g.lambda_reg:g}, from epoch {g.loss_start_epoch + 1}.")


def summary_markdown(summary: dict, seeds: pd.DataFrame, cfg: Config) -> str:
    def cell(stats: dict, k: str) -> str:
        m, s = stats[k]["mean"], stats[k]["std"]
        if m is None or not np.isfinite(m):
            return "n/a"
        s = s if s is not None and np.isfinite(s) else 0.0
        return f"{m * 100:.2f} ± {s * 100:.2f}" if k in PERCENT_METRICS else f"{m:.4f} ± {s:.4f}"

    def label(k: str) -> str:
        return METRIC_LABELS[k] + (" (%)" if k in PERCENT_METRICS else "")

    cal = cfg.calibration
    cal_name = ("calibrated" if cal.method != "none" else "uncalibrated") + (" + thresholds" if cal.tune_thresholds else "")
    lines = [f"# Results - {len(summary['seeds'])} seeds × {summary['n_folds']} folds ({summary['n_fold_models']} models)",
             "", "Outer stratified K-fold (test), inner stratified train / validation split. Seeds: "
             + ", ".join(map(str, summary["seeds"])) + ".", describe_setup(cfg), ""]
    for title, block in (("Test metrics, mean ± std over fold models", summary["fold_level"]),
                         ("Pooled out-of-fold test metrics, mean ± std over seeds", summary["seed_level_pooled_oof"])):
        lines += [f"## {title}", "", f"| metric | raw (argmax) | {cal_name} |", "|---|---|---|"]
        for k in KEY_METRICS:
            if k == EXTRA_FIRST:
                lines.append("| **additional** | | |")
            lines.append(f"| {label(k)} | {cell(block['raw'], k)} | {cell(block['calibrated'], k)} |")
        lines.append("")
    if summary["multi_seed_ensemble"]:
        e = summary["multi_seed_ensemble"]
        lines += ["## Multi-seed ensemble (mean of each sample's out-of-fold probabilities over seeds)", "",
                  f"| metric | raw | {cal_name} |", "|---|---|---|"]
        for k in EVAL_METRICS + ["nll", "ece"]:
            f = (lambda v: f"{v * 100:.2f}") if k in PERCENT_METRICS else (lambda v: f"{v:.4f}")
            lines.append(f"| {label(k)} | {f(e['raw'][k])} | {f(e['calibrated'][k])} |")
        lines.append("")
    lines += ["## Per-seed pooled out-of-fold test (raw)", "",
              "| seed | " + " | ".join(METRIC_LABELS[k] for k in EVAL_METRICS) + " |", "|---" * (len(EVAL_METRICS) + 1) + "|"]
    for _, r in seeds.iterrows():
        vals = [r[f"oof_raw_{k}"] for k in EVAL_METRICS]
        lines.append(f"| {int(r['seed'])} | " + " | ".join(
            "n/a" if v is None or not np.isfinite(v) else (f"{v * 100:.2f}" if k in PERCENT_METRICS else f"{v:.4f}")
            for k, v in zip(EVAL_METRICS, vals)) + " |")
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



# ==============================================================================================
# Whole experiment
# ==============================================================================================
_RESUME_IGNORED = {("splits", "seeds"), ("device",), ("output_dir",), ("genes",)}


def _differences(old: dict, new: dict, prefix: tuple = ()) -> list:
    diffs = []
    for k in sorted(set(old) | set(new)):
        path = prefix + (k,)
        if path in _RESUME_IGNORED:
            continue
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += _differences(a, b, path)
        elif a != b:
            diffs.append(f"{'.'.join(path)}: {a!r} -> {b!r}")
    return diffs


def check_resume_compatible(cfg: Config, run_dir: Path) -> None:
    """Refuse to mix folds trained with different settings in one run directory."""
    prev = run_dir / "config_resolved.yaml"
    if not prev.exists() or not any(run_dir.glob("seed_*/fold_*/fold_result.json")):
        return
    diffs = _differences(load_config(prev).to_dict(), cfg.to_dict())
    if diffs:
        raise ValueError(f"{run_dir} holds folds trained with different settings:\n  " + "\n  ".join(diffs)
                         + "\nUse a new output directory (--out) for the new settings.")


def run_experiment(cfg: Config, data: MultiOmicsData, run_dir: Path, device, logger) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    check_resume_compatible(cfg, run_dir)
    cfg.save(run_dir / "config_resolved.yaml")
    save_json({"class_names": data.class_names, "n_samples": data.n_samples,
               "class_counts": np.bincount(data.labels).tolist(),
               "omics": {o.name: int(o.values.shape[0]) for o in data.omics}}, run_dir / "dataset_info.json")
    t0 = time.time()
    results = {}
    for seed in cfg.splits.seeds:
        results[int(seed)] = run_seed(cfg, data, int(seed), run_dir, device, logger)
        logger.info(f"[elapsed {(time.time() - t0) / 60:.1f} min]")
    return aggregate(cfg, data.class_names, run_dir, results, logger)
