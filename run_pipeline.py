"""One command for the whole pipeline: train -> genes -> go_kegg.

  train    stages 1-6a: splits, omics -> images, CNN + guidance (SWA fold models), calibration,
           IG gene attributions. Finished folds are loaded, not retrained (an interrupted run resumes).
  genes    stage 6b: top-K genes per subtype / omics / sign, PAM50 overlap, stability across seeds
  go_kegg  stage 6c: GO / KEGG enrichment per subtype in R (skipped with a note if R is not installed)

Examples
  python run_pipeline.py                                         # configs/brca.yaml, all steps
  python run_pipeline.py --out runs/brca_gene --set model.head=gene
  python run_pipeline.py --seeds 12 3 --set guidance.enabled=false --out runs/no_guidance
  python run_pipeline.py --steps genes go_kegg --out runs/brca  # downstream of an existing run
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from omics_gradcam.config import Config, load_config  # noqa: E402
from omics_gradcam.utils import get_device, setup_logger  # noqa: E402

STEPS = ("train", "genes", "go_kegg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "brca.yaml"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="section.key=value (repeatable)")
    ap.add_argument("--seeds", type=int, nargs="+", help="override splits.seeds")
    ap.add_argument("--out", help="run directory (overrides output_dir)")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
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
    run_dir = Path(cfg.output_dir) if Path(cfg.output_dir).is_absolute() else ROOT / cfg.output_dir
    logger = setup_logger(run_dir / "run.log")
    logger.info(f"run dir : {run_dir}\nsteps   : {' -> '.join(s for s in STEPS if s in args.steps)}")

    t0 = time.time()
    if "train" in args.steps:
        train(cfg, run_dir, logger, args.gpu_memory_fraction)
    if "genes" in args.steps:
        from omics_gradcam.genes import select_genes
        select_genes(run_dir, cfg.genes.top_k, cfg.genes.label_by, log=logger.info)
    if "go_kegg" in args.steps:
        go_kegg(cfg, run_dir, logger, args.rscript)
    logger.info(f"\nfinished in {(time.time() - t0) / 60:.1f} min - results in {run_dir}")


def train(cfg: Config, run_dir: Path, logger, gpu_memory_fraction: float) -> None:
    import torch
    from omics_gradcam.data import load_multiomics
    from omics_gradcam.experiment import describe_setup, run_experiment

    device = get_device(cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda" and 0 < gpu_memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device.index or 0)
    logger.info(f"device  : {device}" + (f" ({torch.cuda.get_device_name(device.index or 0)})"
                                         if device.type == "cuda" else ""))
    logger.info(f"seeds   : {cfg.splits.seeds} | folds {cfg.splits.n_folds} | validation {cfg.splits.val_fraction:g}")
    logger.info(f"setup   : {describe_setup(cfg)}")
    data_dir = Path(cfg.data.data_dir) if Path(cfg.data.data_dir).is_absolute() else ROOT / cfg.data.data_dir
    data = load_multiomics(dataclasses.replace(cfg.data, data_dir=str(data_dir)), log=logger.info)
    run_experiment(cfg, data, run_dir, device, logger)


def find_rscript(explicit: str | None) -> str | None:
    for cand in (explicit, os.environ.get("RSCRIPT")):
        if cand:
            return cand if Path(cand).exists() or shutil.which(cand) else None
    found = shutil.which("Rscript")
    if found:
        return found
    for pattern in ("C:/Program Files/R/R-*/bin/Rscript.exe", "C:/Program Files/R/R-*/bin/x64/Rscript.exe",
                    os.path.expanduser("~/AppData/Local/Programs/R/R-*/bin/Rscript.exe")):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def go_kegg(cfg: Config, run_dir: Path, logger, rscript: str | None) -> None:
    g = cfg.genes
    gene_dir = run_dir / f"gene_selection_top{g.top_k}"
    if not gene_dir.exists():
        raise SystemExit(f"{gene_dir} not found - run the genes step first")
    rscript = find_rscript(rscript)
    if rscript is None:
        logger.info("[go_kegg] Rscript not found - step skipped. Install R, then once in R: "
                    "install.packages('BiocManager'); BiocManager::install(c('clusterProfiler', 'org.Hs.eg.db', "
                    "'enrichplot', 'ggplot2')). Or pass --rscript <path>.")
        return
    cmd = [rscript, str(ROOT / "scripts" / "go_kegg_enrichment.R"), "--dir", str(gene_dir), "--k", str(g.top_k),
           "--signs", ",".join(g.signs), "--omics", ",".join(g.enrichment_omics), "--ont", g.go_ontology,
           "--padj", str(g.padj), "--out", str(gene_dir / "go_kegg")] + ([] if g.kegg else ["--no-kegg"])
    logger.info("[go_kegg] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (proc.stdout + proc.stderr).splitlines():
        logger.info(f"  {line}")
    if proc.returncode != 0:
        raise SystemExit(f"GO / KEGG step failed (exit code {proc.returncode})")


if __name__ == "__main__":
    main()
