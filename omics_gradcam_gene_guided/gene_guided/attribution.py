"""Attributions: Grad-CAM (the guidance mask) and Integrated Gradients (the gene export).

* ``gradcam``                  Grad-CAM of the model's last kept block, one map per omics
* ``integrated_gradients``     IG of the target logit, zero baseline, batched over the interpolation path
                               (eval mode: the same result as one image at a time)
* ``gene_level_attributions``  read an attribution map at every gene's pixel
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gradcam(model: nn.Module, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple:
    """Raw Grad-CAM (no ReLU, no normalisation), bilinearly upsampled to the input size.

    Returns ``(cam, logits)``; ``cam`` is [B, C, H, W], one map per omics (each omics has its own trunk).
    ``targets`` default to the predicted classes.
    """
    model.eval()
    store = {}
    handle = model.gradcam_layer().register_forward_hook(lambda m, i, out: store.__setitem__("act", out))
    try:
        with torch.enable_grad():
            logits = model(x.detach().requires_grad_(True))     # input grad keeps the graph when frozen
            tg = logits.argmax(1) if targets is None else targets
            (grad,) = torch.autograd.grad(logits.gather(1, tg.unsqueeze(1)).sum(), store["act"])
    finally:
        handle.remove()
    act = store["act"].detach()
    cam = (grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1, keepdim=True)      # [B*C, 1, h, w]
    B, C = x.shape[:2]
    cam = cam.view(B, C, *cam.shape[2:])
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    return cam, logits.detach()


def integrated_gradients(model: nn.Module, x: torch.Tensor, targets: torch.Tensor, steps: int = 64,
                         max_batch: int = 72) -> torch.Tensor:
    """IG of the target logit: ``x * sum_s grad logit(alpha_s * x) / (steps + 1)`` with alpha_s = s / steps,
    s = 0..steps (zero baseline)."""
    model.eval()
    alphas = np.arange(steps + 1) / steps
    B, S = x.size(0), len(alphas)
    a = torch.as_tensor(np.asarray(alphas), dtype=x.dtype, device=x.device)
    total = torch.zeros_like(x)

    def grads(xs: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
        xs = xs.detach().requires_grad_(True)
        out = model(xs)
        return torch.autograd.grad(out.gather(1, tg.unsqueeze(1)).sum(), xs)[0]

    if S <= max_batch:                      # several samples per pass, all steps of a sample together
        per_pass = max(1, max_batch // S)
        for s0 in range(0, B, per_pass):
            xb = x[s0:s0 + per_pass]
            k = xb.size(0)
            xs = (a.view(1, S, 1, 1, 1) * xb.unsqueeze(1)).reshape(k * S, *x.shape[1:])
            total[s0:s0 + k] = grads(xs, targets[s0:s0 + k].repeat_interleave(S)).view(k, S, *x.shape[1:]).sum(1)
    else:                                   # one sample at a time, steps in chunks
        for b in range(B):
            for t0 in range(0, S, max_batch):
                at = a[t0:t0 + max_batch]
                total[b] += grads(at.view(-1, 1, 1, 1) * x[b].unsqueeze(0), targets[b].repeat(len(at))).sum(0)
    return x.detach() * total / (steps + 1)


def gene_level_attributions(attr: np.ndarray, pixel_index: list) -> list:
    """``attr`` [N, C, H, W] -> per omics [N, n_features] float32 (the value at each feature's pixel)."""
    return [attr[:, c, rows, cols].astype(np.float32) for c, (rows, cols) in enumerate(pixel_index)]
