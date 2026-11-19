"""EfficientNet-B1 (ImageNet-pretrained, timm) with one trunk per omics and a gene-indexed readout.

* Every omics image goes through its own copy of the trunk (one input channel; all copies start from the
  same pretrained weights, the RGB stem weights summed), so convolutions never mix the independent layouts
  of different omics. The trunk stops after the deepest readout level (stage 5 for levels [2, 3]).
* Each readout level (timm feature level: EfficientNet 0-4 = stride 2/4/8/16/32) is projected by a 1x1 conv
  to one class-evidence map per class; the maps are read at every gene's pixel (bilinear, exactly
  ``F.interpolate(..., align_corners=False)`` at that pixel) and summed over levels:
      z[b, k, g]  = sum_l M_l[b, k](pos_g)
      logit[b, k] = sum_g w[k, g] * dropout(z)[b, k, g] / sqrt(G) + bias[k]
  with w = 1/sqrt(G) at the start, i.e. the average evidence over the gene pixels.
* The classifier only reads learned convolutional features, never the input pixels.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import timm
import torch
import torch.nn as nn

from .config import ModelConfig


class PerOmics(nn.Module):
    """``n`` copies of ``module`` (same initial weights); copy o runs on omics o of a stream batch
    [B * n, ...] laid out sample-major (b0o0, b0o1, ..., b1o0, ...). The output keeps that layout."""

    def __init__(self, module: nn.Module, n: int):
        super().__init__()
        self.n = n
        self.mods = nn.ModuleList(copy.deepcopy(module) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.reshape(-1, self.n, *x.shape[1:])
        return torch.stack([m(xs[:, o]) for o, m in enumerate(self.mods)], 1).flatten(0, 1)


def bilinear_taps(rows: np.ndarray, cols: np.ndarray, image_size: int, map_size: int) -> tuple:
    """Flat indices [G, 4] and weights [G, 4] that read a ``map_size``² map at the image pixels
    (rows, cols) exactly as ``F.interpolate(map, image_size, mode="bilinear", align_corners=False)``."""
    scale = map_size / image_size

    def axis(p):
        src = np.maximum((np.asarray(p, dtype=np.float64) + 0.5) * scale - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), map_size - 1)
        return i0, np.minimum(i0 + 1, map_size - 1), src - i0

    y0, y1, ly = axis(rows)
    x0, x1, lx = axis(cols)
    idx = np.stack([y0 * map_size + x0, y0 * map_size + x1, y1 * map_size + x0, y1 * map_size + x1], 1)
    wts = np.stack([(1 - ly) * (1 - lx), (1 - ly) * lx, ly * (1 - lx), ly * lx], 1)
    return torch.as_tensor(idx), torch.as_tensor(wts, dtype=torch.float32)


class GeneReadoutClassifier(nn.Module):
    """One EfficientNet trunk per omics + gene-indexed readout (see the module docstring)."""

    def __init__(self, arch: str, num_classes: int, pixel_index: list, image_size: int, levels=(2, 3),
                 pretrained: bool = True, dropout: float = 0.6):
        super().__init__()
        self.n_omics = len(pixel_index)
        self.memory_factor = self.n_omics                    # one trunk pass per omics image
        net = timm.create_model(arch, pretrained=pretrained, in_chans=1)
        if not all(hasattr(net, a) for a in ("conv_stem", "bn1", "blocks")):
            raise ValueError(f"model.arch must be an EfficientNet-style timm model, got {arch!r}")
        info = net.feature_info
        self.levels = sorted(int(v) for v in levels)
        self.taps = [int(info[lv]["module"].split(".")[1]) for lv in self.levels]   # block index per level

        # one trunk per omics: stem, BatchNorm and every block up to the deepest readout level
        self.conv_stem = PerOmics(net.conv_stem, self.n_omics)
        self.bn1 = PerOmics(net.bn1, self.n_omics)
        self.blocks = nn.Sequential(*(PerOmics(b, self.n_omics) for b in list(net.blocks)[:max(self.taps) + 1]))
        self.proj = nn.ModuleList(nn.ModuleList(nn.Conv2d(info[lv]["num_chs"], num_classes, 1)
                                                for _ in range(self.n_omics)) for lv in self.levels)
        self.dropout = nn.Dropout(p=dropout)

        map_sizes = self._map_sizes(image_size)
        for o, (rows, cols) in enumerate(pixel_index):       # genes of omics o are read from trunk o
            for i, size in enumerate(map_sizes):
                idx, wts = bilinear_taps(rows, cols, image_size, size)
                self.register_buffer(f"taps_idx{i}_{o}", idx)
                self.register_buffer(f"taps_w{i}_{o}", wts)
        self.n_genes = sum(len(r) for r, _ in pixel_index)
        self.scale = 1.0 / math.sqrt(self.n_genes)
        self.weight = nn.Parameter(torch.full((num_classes, self.n_genes), self.scale))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    @torch.no_grad()
    def _map_sizes(self, image_size: int) -> list:
        self.eval()
        h = self.bn1(self.conv_stem(torch.zeros(self.n_omics, 1, image_size, image_size)))
        sizes = []
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i in self.taps:
                sizes.append(h.shape[-1])
        self.train()
        return sizes

    def gene_evidence(self, x: torch.Tensor) -> torch.Tensor:
        """z [B, K, G]: class evidence of the network at every gene's pixel (genes ordered by omics)."""
        B, O = x.shape[0], self.n_omics
        if x.shape[1] != O:
            raise ValueError(f"expected {O} omics channels, got {x.shape[1]}")
        h = self.bn1(self.conv_stem(x.reshape(B * O, 1, *x.shape[2:])))
        feats = []
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i in self.taps:
                feats.append(h)
        parts = []
        for o in range(O):
            z = 0
            for i, f in enumerate(feats):
                m = self.proj[i][o](f.view(B, O, *f.shape[1:])[:, o]).flatten(2)          # [B, K, h*w]
                z = z + (m[:, :, getattr(self, f"taps_idx{i}_{o}")] * getattr(self, f"taps_w{i}_{o}")).sum(-1)
            parts.append(z)
        return torch.cat(parts, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gene_evidence(x)
        return (self.dropout(z) * self.weight).sum(-1) * self.scale + self.bias

    def head_parameters(self) -> list:
        return [*self.proj.parameters(), self.weight, self.bias]

    def freeze_backbone(self) -> None:
        head = {id(p) for p in self.head_parameters()}
        for p in self.parameters():
            p.requires_grad = id(p) in head

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def gradcam_layer(self) -> nn.Module:
        """Last kept block; its batch is B x O, so Grad-CAM gives one map per omics."""
        return self.blocks[-1]


def build_model(cfg: ModelConfig, num_classes: int, pixel_index: list, image_size: int) -> GeneReadoutClassifier:
    return GeneReadoutClassifier(cfg.arch, num_classes, pixel_index, image_size, cfg.readout_levels,
                                 cfg.pretrained, cfg.dropout)
