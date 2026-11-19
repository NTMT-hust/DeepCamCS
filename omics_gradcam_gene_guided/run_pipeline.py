"""Run the pipeline, all seven phases or some of them (the DeepCamCS phases):

  1  images     omics matrices -> images and gene coordinates, per seed x fold (layout fitted on the training split)
  2  train      train and evaluate the guided CNN (SWA fold models), IG attributions of the test samples, summary
  3  genes      top-K attributed genes per subtype x omics x sign, stability across seeds
  4  merge      merge the selected genes, overlap with PAM50
  5  filter     original omics matrices restricted to the selected genes
  6  pathway    MSigDB Hallmark enrichment (local hypergeometric test against the measured genes)
  7  go_kegg    GO / KEGG enrichment per subtype in R (skipped with a note if R is not installed)

Each phase reads the files of the previous ones from the run directory, so phases can be rerun on their own.
Phases 1 and 2 skip finished folds, so an interrupted run resumes where it stopped.

Examples
  python run_pipeline.py                                     # config.yaml, phases 1-7
  python run_pipeline.py --phases 3 4 5 6 7                  # downstream of finished training
  python run_pipeline.py --seeds 12 3 --out runs/dev         # two seeds into another run directory
  python run_pipeline.py --set guidance.enabled=false --out runs/no_guidance
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gene_guided.config import Config, load_config  # noqa: E402
from gene_guided.utils import get_device, setup_logger  # noqa: E402

PHASES = {1: "images", 2: "train", 3: "genes", 4: "merge", 5: "filter", 6: "pathway", 7: "go_kegg"}
# settings that may differ between a finished part of a run and its continuation
_RESUME_IGNORED = {("splits", "seeds"), ("device",), ("output_dir",), ("genes",), ("pathway",), ("go_kegg",)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--phases", type=int, nargs="+", choices=sorted(PHASES), default=sorted(PHASES))
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="section.key=value (repeatable)")
    ap.add_argument("--seeds", type=int, nargs="+", help="override splits.seeds")
    ap.add_argument("--out", help="run directory (overrides output_dir)")
    ap.add_argument("--rscript", help="path to Rscript (default: PATH, $RSCRIPT, C:/Program Files/R)")
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.9,
                    help="cap on this process's share of GPU memory (1 = no cap). On Windows an over-full GPU "
                         "silently spills into system RAM (50-100x slower) instead of failing.")
    args = ap.parse_args()

    overrides = list(args.overrides)
    if args.seeds:
        overrides.append(f"splits.seeds=[{','.join(map(str, args.seeds))}]")
    if args.out:
        overrides.append(f"output_dir={args.out}")
    cfg = load_config(args.config, overrides)
    data_dir = Path(cfg.data.data_dir) if Path(cfg.data.data_dir).is_absolute() else ROOT / cfg.data.data_dir
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dir=str(data_dir)))
    run_dir = Path(cfg.output_dir) if Path(cfg.output_dir).is_absolute() else ROOT / cfg.output_dir
    logger = setup_logger(run_dir / "run.log")
    phases = sorted(set(args.phases))
    logger.info(f"run dir : {run_dir}\nphases  : {' -> '.join(f'{p} {PHASES[p]}' for p in phases)}")
    if {1, 2} & set(phases):
        check_resume_compatible(cfg, run_dir)
        cfg.save(run_dir / "config_resolved.yaml")

    t0 = time.time()
    for p in phases:
        logger.info(f"\n{'#' * 30} PHASE {p}: {PHASES[p]} {'#' * 30}")
        if p == 1:
            from gene_guided import phase1_images
            phase1_images.run(cfg, run_dir, logger)
        elif p == 2:
            train(cfg, run_dir, logger, args.gpu_memory_fraction)
        elif p == 3:
            from gene_guided import phase3_choose_genes
            phase3_choose_genes.run(cfg, run_dir, logger)
        elif p == 4:
            from gene_guided import phase4_merge_genes
            phase4_merge_genes.run(cfg, run_dir, logger)
        elif p == 5:
            from gene_guided import phase5_filter_omics
            phase5_filter_omics.run(cfg, run_dir, logger)
        elif p == 6:
            from gene_guided import phase6_pathway_analysis
            phase6_pathway_analysis.run(cfg, run_dir, logger)
        elif p == 7:
            from gene_guided import phase7_go_kegg
            phase7_go_kegg.run(cfg, run_dir, logger, args.rscript)
    logger.info(f"\nfinished in {(time.time() - t0) / 60:.1f} min - results in {run_dir}")


def train(cfg: Config, run_dir: Path, logger, gpu_memory_fraction: float) -> None:
    import torch
    from gene_guided import phase2_train

    device = get_device(cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda" and 0 < gpu_memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device.index or 0)
    logger.info(f"device  : {device}" + (f" ({torch.cuda.get_device_name(device.index or 0)})"
                                         if device.type == "cuda" else ""))
    logger.info(f"seeds   : {cfg.splits.seeds} | folds {cfg.splits.n_folds} | validation {cfg.splits.val_fraction:g}")
    phase2_train.run(cfg, run_dir, device, logger)


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
    """Refuse to mix folds made with different settings in one run directory."""
    prev = run_dir / "config_resolved.yaml"
    finished = any(run_dir.glob("1_images/seed_*/fold_*/done.json")) or any(run_dir.glob("2_training/seed_*/fold_*/fold_result.json"))
    if not prev.exists() or not finished:
        return
    diffs = _differences(load_config(prev).to_dict(), cfg.to_dict())
    if diffs:
        raise SystemExit(f"{run_dir} holds folds made with different settings:\n  " + "\n  ".join(diffs)
                         + "\nUse a new output directory (--out) for the new settings.")


if __name__ == "__main__":
    main()
