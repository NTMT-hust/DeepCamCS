"""Stage 3a - class-aware augmentation of the training images (refactor of ``AwareAugmentation.py``).

One uniform draw ``u`` per sample picks at most one operation (defaults in brackets):

    u <= p_noise (0.2)          Gaussian noise (std 0.03) on the gene pixels only
    next p_cutmix (0.1)         CutMix with another image of the SAME class (20-50% of the side)
    next p_mixup  (0.1)         MixUp  with another image of the SAME class (Beta(0.2, 0.2))
    next p_smote  (0.1)         SMOTE-style interpolation towards another image of the same class
    otherwise                   unchanged

There are no flips, rotations or shifts: they would move genes off their pixels. The partner image is
never the sample itself, and every random generator is seeded, so runs are reproducible.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .config import AugmentConfig


class ClassAwareAugmenter:
    def __init__(self, images_uint8: torch.Tensor, labels: np.ndarray, seed: int, cfg: AugmentConfig):
        self.images = images_uint8                     # [N_train, C, H, W] uint8, CPU
        self.pools = {int(c): np.flatnonzero(np.asarray(labels) == c) for c in np.unique(labels)}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.cfg = cfg
        self.edges = np.cumsum([cfg.p_noise, cfg.p_cutmix, cfg.p_mixup, cfg.p_smote])

    def __call__(self, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``images`` float [B, C, H, W]; ``indices`` = positions of the samples in ``images_uint8``."""
        return torch.stack([self.augment_one(images[i], int(labels[i]), int(indices[i]))
                            for i in range(images.size(0))])

    def augment_one(self, image: torch.Tensor, label: int, idx: int) -> torch.Tensor:
        image = image.clone()
        _, H, W = image.shape
        u = self.rng.random()
        e_noise, e_cutmix, e_mixup, e_smote = self.edges
        if u <= e_noise:
            noise = torch.randn(image.shape, generator=self.gen) * self.cfg.noise_std
            return image + noise * (image != 0).float()
        if u > e_smote:
            return image
        other = self._partner(label, idx)
        if other is None:
            return image
        if u <= e_cutmix:
            r = self.rng.uniform(*self.cfg.cutmix_ratio)
            cw, ch = int(W * r), int(H * r)
            cx, cy = self.rng.randint(0, W - cw), self.rng.randint(0, H - ch)
            image[:, cy:cy + ch, cx:cx + cw] = other[:, cy:cy + ch, cx:cx + cw]
        elif u <= e_mixup:
            lam = float(self.np_rng.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha))
            image = lam * image + (1 - lam) * other
        else:
            lam = self.rng.uniform(*self.cfg.smote_alpha_range)
            image = image + lam * (other - image)
        return image

    def _partner(self, label: int, idx: int) -> torch.Tensor | None:
        """Random other training image of the same class (None if the class has one image)."""
        pool = self.pools.get(label)
        if pool is None or len(pool) <= 1:
            return None
        j = idx
        while j == idx:
            j = int(pool[self.rng.randrange(len(pool))])
        return self.images[j].float().div_(255.0)
