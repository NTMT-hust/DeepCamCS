"""Phase 7 - GO and KEGG enrichment per subtype (DeepCamCS ``go_kegg_enrichment.R``).

Runs ``scripts/go_kegg_enrichment.R`` (clusterProfiler) on phase 3's per-label lists, against the measured
genes as the universe. GO uses the local org.Hs.eg.db annotation; KEGG downloads the pathway annotation from
the KEGG REST API (no gene list is sent). Outputs in ``7_go_kegg/``; skipped with a note if R is not installed.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import Config
from .phase3_choose_genes import out_dir as genes_dir

PHASE_DIR = "7_go_kegg"
ROOT = Path(__file__).resolve().parents[1]


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


def run(cfg: Config, run_dir: Path, logger: logging.Logger, rscript: str | None = None) -> Path | None:
    g = cfg.go_kegg
    src = genes_dir(run_dir, cfg.genes.top_k)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - run phase 3 first")
    rscript = find_rscript(rscript)
    if rscript is None:
        logger.info("[go_kegg] Rscript not found - phase skipped. Install R, then once in R: "
                    "install.packages('BiocManager'); BiocManager::install(c('clusterProfiler', 'org.Hs.eg.db', "
                    "'enrichplot', 'ggplot2')). Or pass --rscript <path>.")
        return None
    out = Path(run_dir) / PHASE_DIR
    cmd = [rscript, str(ROOT / "scripts" / "go_kegg_enrichment.R"), "--dir", str(src), "--k", str(cfg.genes.top_k),
           "--signs", ",".join(g.signs), "--omics", ",".join(g.omics), "--ont", g.ontology, "--padj", str(g.padj),
           "--out", str(out)] + ([] if g.kegg else ["--no-kegg"])
    logger.info("[go_kegg] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (proc.stdout + proc.stderr).splitlines():
        logger.info(f"  {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"GO / KEGG phase failed (exit code {proc.returncode})")
    return out
