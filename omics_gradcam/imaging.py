"""Stage 2 - omics -> image (DeepInsight-style; refactor of ``PrepareData4.py``).

Layout: which pixel each feature gets. Fitted per fold, per omics, on the training samples only.
  1. UMAP embeds every feature in 2-D from its values across the training samples (cosine distance).
  2. The minimum-area rotated bounding rectangle of the embedding is rotated to be axis-aligned.
  3. The embedding is scaled onto the ``size x size`` grid and rounded to pixels.
  4. Snowfall: a feature whose pixel is already taken moves to the nearest free pixel (breadth-first
     search), so every feature has its own pixel. ~3,200 features fill ~5.5% of a 240 x 240 channel.

Pixel values: per sample and channel, from that sample's own values only (no information is shared
between samples). Write each feature at its pixel, clip to the sample's [p5, p95], min-max to [0, 1],
log1p, min-max again; with ``reserve_zero`` map the genes to [1.5/255, 1] so that 0 only ever means
"empty pixel"; quantise to uint8. The omics are stacked as channels (mRNA, methylation, CNV = R, G, B).
"""
from __future__ import annotations

import os
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ImageConfig

# Lowest gene value with reserve_zero: 1.5/255 survives the uint8 conversion as 1 (1/255 can round to 0)
ZERO_FLOOR = 1.5 / 255


@dataclass
class PixelLayout:
    rows: np.ndarray       # [n_features] int, 0-based: feature j is drawn at image[rows[j], cols[j]]
    cols: np.ndarray


# ----------------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------------
def umap_embedding(Q: np.ndarray, n_neighbors: int, min_dist: float, metric: str, init: str,
                   random_state: int) -> np.ndarray:
    """2-D UMAP embedding of the rows of ``Q`` (features x samples)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # umap imports TensorFlow when it is installed
    import umap                                       # heavy import, only needed here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, Q.shape[0] - 1), min_dist=min_dist,
                            metric=metric, init=init, random_state=random_state)
        return reducer.fit_transform(Q)


def axis_aligned(embedding: np.ndarray) -> tuple:
    """Rotate the embedding so its minimum-area bounding rectangle is axis-aligned."""
    from shapely.geometry import MultiPoint
    x, y = embedding[:, 0], embedding[:, 1]
    rect = np.asarray(MultiPoint(np.column_stack([x, y])).convex_hull.minimum_rotated_rectangle.exterior.coords)
    theta = np.arctan2(rect[1, 1] - rect[0, 1], rect[1, 0] - rect[0, 0])
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    z = rotation @ np.vstack([x, y])
    return z[0], z[1]


def snowfall(xp: np.ndarray, yp: np.ndarray, size: int) -> tuple:
    """Give every feature its own pixel (1-based coordinates in 1..size), in feature order."""
    xp, yp = np.asarray(xp).copy(), np.asarray(yp).copy()
    occupied: set = set()
    for i in range(len(xp)):
        x, y = int(min(max(1, xp[i]), size)), int(min(max(1, yp[i]), size))
        if (x, y) in occupied:
            x, y = _nearest_free(x, y, occupied, size)
        xp[i], yp[i] = x, y
        occupied.add((x, y))
    return xp, yp


def _nearest_free(x: int, y: int, occupied: set, size: int) -> tuple:
    """Breadth-first search (4-neighbourhood) for the nearest free pixel."""
    visited = {(x, y)}
    queue = deque([(x, y)])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) not in occupied:
            return cx, cy
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 1 <= nx <= size and 1 <= ny <= size and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    raise ValueError(f"more features than pixels ({size} x {size}); increase image.size")


def fit_layout(Q: np.ndarray, size: int, *, n_neighbors: int = 30, min_dist: float = 0.1,
               metric: str = "cosine", init: str = "spectral", random_state: int = 42) -> PixelLayout:
    """Pixel of every feature (rows of ``Q`` = [n_features, n_fit_samples])."""
    zx, zy = axis_aligned(umap_embedding(Q, n_neighbors, min_dist, metric, init, random_state))
    xp = np.round(1 + size * (zx - zx.min()) / (zx.max() - zx.min())).astype(int)
    yp = np.round(1 - size * (zy - zy.max()) / (zy.max() - zy.min())).astype(int)
    xp, yp = snowfall(xp, yp, size)
    return PixelLayout(rows=xp.astype(np.int64) - 1, cols=yp.astype(np.int64) - 1)


# ----------------------------------------------------------------------------------------------
# Pixel values
# ----------------------------------------------------------------------------------------------
def normalise_per_sample(v: np.ndarray, clip_percentile) -> np.ndarray:
    """``v``: [n_features, n_samples] float64. Each column (sample) separately: clip to its
    percentiles, min-max, log1p, min-max."""
    if clip_percentile is not None:
        lo, hi = np.percentile(v, clip_percentile, axis=0)
        v = np.clip(v, lo[None, :], hi[None, :])
    for step in ("minmax", "log", "minmax"):
        if step == "log":
            v = np.log1p(v)
            continue
        v = v - v.min(axis=0, keepdims=True)
        mx = v.max(axis=0, keepdims=True)
        v = np.divide(v, mx, out=v, where=mx > 0)
    return v


def render(F: np.ndarray, layout: PixelLayout, size: int, clip_percentile=(5, 95),
           value_floor: float = 0.0) -> np.ndarray:
    """Images of one omics: ``F`` [n_features, n_samples] -> [n_samples, size, size] float64 in [0, 1]."""
    values = np.nan_to_num(np.asarray(F, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = normalise_per_sample(values, clip_percentile)
    if value_floor > 0:
        values = value_floor + (1.0 - value_floor) * values
    out = np.zeros((values.shape[1], size * size), dtype=np.float64)
    out[:, layout.rows * size + layout.cols] = values.T
    return out.reshape(values.shape[1], size, size)


# ----------------------------------------------------------------------------------------------
# Fit / transform
# ----------------------------------------------------------------------------------------------
class OmicsImageTransformer:
    """One pixel layout per omics; renders [n_samples, n_omics, size, size] uint8 images."""

    def __init__(self, omics_names: list, cfg: ImageConfig, random_state: int):
        self.omics_names = list(omics_names)
        self.cfg = cfg
        self.random_state = int(random_state)
        self.layouts: list = []

    @property
    def value_floor(self) -> float:
        return ZERO_FLOOR if self.cfg.reserve_zero else 0.0

    def fit(self, matrices: list, log=None) -> "OmicsImageTransformer":
        """``matrices``: per omics [n_features, n_fit_samples] (the training split)."""
        c = self.cfg
        self.layouts = [fit_layout(Q, c.size, n_neighbors=c.umap_n_neighbors, min_dist=c.umap_min_dist,
                                   metric=c.umap_metric, init=c.umap_init, random_state=self.random_state)
                        for Q in matrices]
        if log:
            for name, Q in zip(self.omics_names, matrices):
                log(f"  [layout] {name:<12} {Q.shape[0]} features on {Q.shape[1]} samples")
        return self

    def transform(self, matrices: list, chunk: int = 128) -> np.ndarray:
        """uint8 images [n_samples, n_omics, size, size] (same values as the original PNG round-trip)."""
        if not self.layouts:
            raise RuntimeError("call fit() or load() first")
        n, size = matrices[0].shape[1], self.cfg.size
        out = np.empty((n, len(self.layouts), size, size), dtype=np.uint8)
        for s0 in range(0, n, chunk):
            for c, (layout, F) in enumerate(zip(self.layouts, matrices)):
                img = render(F[:, s0:s0 + chunk], layout, size, self.cfg.clip_percentile, self.value_floor)
                out[s0:s0 + chunk, c] = (img * 255).astype(np.uint8)
        return out

    def pixel_index(self) -> list:
        """Per omics: (rows, cols), usable as ``image[c, rows, cols]``."""
        return [(lay.rows, lay.cols) for lay in self.layouts]

    def coordinates(self, feature_names: list) -> dict:
        """Per omics: DataFrame gene_name / row / col."""
        return {name: pd.DataFrame({"gene_name": np.asarray(feats), "row": lay.rows, "col": lay.cols})
                for name, lay, feats in zip(self.omics_names, self.layouts, feature_names)}

    def save(self, path: Path) -> None:
        arrays = {f"rows_{i}": lay.rows for i, lay in enumerate(self.layouts)}
        arrays.update({f"cols_{i}": lay.cols for i, lay in enumerate(self.layouts)})
        np.savez_compressed(path, omics_names=np.array(self.omics_names), size=self.cfg.size, **arrays)

    def load(self, path: Path) -> "OmicsImageTransformer":
        z = np.load(path, allow_pickle=False)
        self.layouts = [PixelLayout(z[f"rows_{i}"], z[f"cols_{i}"]) for i in range(len(self.omics_names))]
        return self
