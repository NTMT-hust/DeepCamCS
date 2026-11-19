"""Phase 1 - transform the omics matrices into images and gene coordinates (DeepCamCS ``produceIMG.py``).

The image layout is fitted on the training split of each fold (no test sample influences where a gene is
drawn), so there is one image set per seed x fold. Per fold, ``1_images/seed_<s>/fold_<f>/``:

    images.npz                     uint8 [n_samples, n_omics, size, size], every sample rendered with this layout
    layout.npz                     pixel (row, col) of every feature, per omics
    gene_coordinates_<omics>.csv   the same as a table: gene_name / row / col
    anova_rank.npz                 ANOVA-F rank of every feature on the training split (guidance: protected genes)
    done.json                      written last: the fold is complete

Plus ``1_images/dataset.npz`` (sample IDs, labels, class / omics / feature names) and
``1_images/seed_<s>/splits.json`` (train / val / test indices of every fold). Finished folds are skipped.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import FoldSplit, MultiOmicsData, load_multiomics, make_cv_splits
from .guidance import FeatureRelevance
from .imaging import OmicsImageTransformer, anova_ranks
from .utils import fold_seed, load_json, save_json, seed_everything

PHASE_DIR = "1_images"


@dataclass
class Dataset:
    """What later phases need to know about the samples (from ``dataset.npz``)."""
    sample_ids: np.ndarray
    labels: np.ndarray
    class_names: list
    omics_names: list
    features: dict           # omics -> feature names

    @classmethod
    def load(cls, run_dir: Path) -> "Dataset":
        path = Path(run_dir) / PHASE_DIR / "dataset.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run phase 1 first")
        z = np.load(path, allow_pickle=False)
        omics = [str(o) for o in z["omics_names"]]
        return cls(z["sample_ids"].astype(str), z["labels"].astype(np.int64), [str(c) for c in z["class_names"]],
                   omics, {o: z[f"features_{o}"].astype(str) for o in omics})


def save_dataset(data: MultiOmicsData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dataset.npz", sample_ids=data.sample_ids, labels=data.labels,
                        class_names=np.array(data.class_names), omics_names=np.array(data.omics_names),
                        **{f"features_{o.name}": o.feature_names for o in data.omics})


def fold_dir(run_dir: Path, seed: int, fold: int) -> Path:
    """``fold`` 0-based."""
    return Path(run_dir) / PHASE_DIR / f"seed_{seed}" / f"fold_{fold + 1}"


def load_splits(run_dir: Path, seed: int) -> list:
    path = Path(run_dir) / PHASE_DIR / f"seed_{seed}" / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run phase 1 for seed {seed} first")
    return [FoldSplit.from_dict(d) for d in load_json(path)]


def load_fold_images(run_dir: Path, seed: int, fold: int, omics: list) -> tuple:
    """(images uint8 tensor, pixel_index [(rows, cols)] per omics, FeatureRelevance) of one fold."""
    d = fold_dir(run_dir, seed, fold)
    if not (d / "done.json").exists():
        raise FileNotFoundError(f"{d} is incomplete - run phase 1 first")
    images = torch.from_numpy(np.load(d / "images.npz")["images"])
    lay = np.load(d / "layout.npz", allow_pickle=False)
    pixels = [(lay[f"rows_{i}"], lay[f"cols_{i}"]) for i in range(len(omics))]
    rk = np.load(d / "anova_rank.npz", allow_pickle=False)
    relevance = FeatureRelevance(list(omics), pixels, [rk[f"rank_{o}"] for o in omics], tuple(images.shape[1:]))
    return images, pixels, relevance


def run(cfg: Config, run_dir: Path, logger: logging.Logger) -> None:
    data = load_multiomics(cfg.data, log=logger.info)
    out = Path(run_dir) / PHASE_DIR
    save_dataset(data, out)
    save_json({"class_names": data.class_names, "n_samples": len(data.sample_ids),
               "class_counts": np.bincount(data.labels).tolist(),
               "omics": {o.name: int(o.values.shape[0]) for o in data.omics}}, Path(run_dir) / "dataset_info.json")
    for seed in cfg.splits.seeds:
        splits = make_cv_splits(data.labels, seed, cfg.splits.n_folds, cfg.splits.val_fraction)
        save_json([sp.as_dict() for sp in splits], out / f"seed_{seed}" / "splits.json")
        for sp in splits:
            _fold(cfg, data, sp, run_dir, logger)


def _fold(cfg: Config, data: MultiOmicsData, sp: FoldSplit, run_dir: Path, logger: logging.Logger) -> None:
    d = fold_dir(run_dir, sp.seed, sp.fold)
    if (d / "done.json").exists():
        logger.info(f"  [images] seed {sp.seed} fold {sp.fold + 1}: already done")
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info(f"  [images] seed {sp.seed} fold {sp.fold + 1}: layout fitted on {len(sp.train)} training samples")
    seed_everything(fold_seed(sp.seed, sp.fold), cfg.train.deterministic)
    tfm = OmicsImageTransformer(data.omics_names, cfg.image, random_state=sp.seed)
    tfm.fit(data.matrices(sp.train), log=logger.info)
    images = tfm.transform(data.matrices())
    np.savez_compressed(d / "images.npz", images=images)
    tfm.save(d / "layout.npz")
    for name, df in tfm.coordinates([o.feature_names for o in data.omics]).items():
        df.to_csv(d / f"gene_coordinates_{name}.csv", index=False)
    ranks = anova_ranks(data.matrices(sp.train), data.labels[sp.train])
    np.savez_compressed(d / "anova_rank.npz", **{f"rank_{o}": r for o, r in zip(data.omics_names, ranks)})
    save_json({"seed": sp.seed, "fold": sp.fold + 1, "shape": list(images.shape), "seconds": round(time.time() - t0, 1)},
              d / "done.json")
    logger.info(f"  [images] {tuple(images.shape)} uint8 in {time.time() - t0:.1f}s")
