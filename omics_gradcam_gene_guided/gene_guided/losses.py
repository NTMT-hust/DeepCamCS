"""Focal loss with inverse-frequency class weights."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: mean over samples of alpha_y * (1 - p_y)^gamma * CE."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha, persistent=False)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """N / (K * n_c), normalised to sum to 1."""
    counts = np.maximum(np.bincount(labels, minlength=num_classes).astype(np.float64), 1)
    w = len(labels) / (num_classes * counts)
    return torch.tensor(w / w.sum(), dtype=torch.float32)
