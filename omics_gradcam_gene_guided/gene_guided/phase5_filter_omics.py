"""Phase 5 - filter the original omics matrices to the selected genes (DeepCamCS
``filter_omics_by_unique_genes.py``).

Each omics CSV of ``data`` is read as it is on disk (all samples, original values, NaN kept) and only the rows
of the features selected in phase 4 are written, together with the labels file:

    5_filtered_omics/<omics file name>_selected.csv    selected features x all samples
    5_filtered_omics/<labels file name>                copy of the labels
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from .config import Config
from .data import read_omics_table
from .phase4_merge_genes import PHASE_DIR as MERGED_DIR

PHASE_DIR = "5_filtered_omics"


def run(cfg: Config, run_dir: Path, logger: logging.Logger) -> Path:
    src = Path(run_dir) / MERGED_DIR / "unique_genes_omics.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - run phase 4 first")
    unique = pd.read_csv(src)
    out = Path(run_dir) / PHASE_DIR
    out.mkdir(parents=True, exist_ok=True)
    base = Path(cfg.data.data_dir)
    for name, fname in cfg.data.omics.items():
        df = read_omics_table(base / fname)
        wanted = set(unique.loc[unique["omics_type"] == name, "unique_genes"].astype(str))
        kept = df[df.index.isin(wanted)]
        kept.index.name = "gene_id"
        path = out / f"{Path(fname).stem}_selected.csv"
        kept.to_csv(path)
        logger.info(f"[filter] {name:<12} {len(kept):>5} of {len(df)} features kept "
                    f"({len(wanted) - len(kept)} selected features not found) -> {path.name}")
    shutil.copyfile(base / cfg.data.labels_file, out / Path(cfg.data.labels_file).name)
    return out
