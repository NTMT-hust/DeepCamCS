"""Attributions shared by the guidance masks (stage 4) and the gene export (stage 6).

* ``gradcam``                   Grad-CAM of the model's ``gradcam_layer`` (batched; DeepCamCS GradCAM)
* ``integrated_gradients``      IG of the target logit, zero baseline (gene export)
* ``sparse_integrated_gradients``  IG of the target probability, few steps (IG guidance mask)
* ``gene_level_attributions``   read an attribution map at every gene's pixel

All IG variants are evaluated batched over the interpolation path (eval mode: same result as one
image at a time).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gradcam(model: nn.Module, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple:
    """Raw Grad-CAM (no ReLU, no normalisation), bilinearly upsampled to the input size.

    Returns ``(cam, logits)``. ``cam`` is [B, C, H, W]: for the one-trunk-per-omics model there is one
    map per omics; for the original model the single spatial map is repeated on every channel.
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
    cam = (grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1, keepdim=True)      # [B or B*C, 1, h, w]
    B, C = x.shape[:2]
    if cam.shape[0] == B * C and C > 1:                         # one trunk per omics
        cam = cam.view(B, C, *cam.shape[2:])
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    if cam.shape[1] == 1:
        cam = cam.expand(-1, C, -1, -1)
    return cam, logits.detach()


def _path_integrated_gradients(model: nn.Module, x: torch.Tensor, targets: torch.Tensor, alphas: np.ndarray,
                               normaliser: float, use_softmax: bool, max_batch: int) -> torch.Tensor:
    """``x * sum_s grad score(alpha_s * x) / normaliser`` (zero baseline)."""
    model.eval()
    B, S = x.size(0), len(alphas)
    a = torch.as_tensor(np.asarray(alphas), dtype=x.dtype, device=x.device)
    total = torch.zeros_like(x)

    def grads(xs: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
        xs = xs.detach().requires_grad_(True)
        out = model(xs)
        if use_softmax:
            out = torch.softmax(out, dim=1)
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
    return x.detach() * total / normaliser


def integrated_gradients(model, x, targets, steps: int = 64, max_batch: int = 72) -> torch.Tensor:
    """IG of the target logit: alphas i/steps, i = 0..steps, averaged over the steps + 1 points."""
    return _path_integrated_gradients(model, x, targets, np.arange(steps + 1) / steps, steps + 1,
                                      use_softmax=False, max_batch=max_batch)


def sparse_integrated_gradients(model, x, targets, steps: int = 8, max_batch: int = 72) -> torch.Tensor:
    """IG of the target probability: alphas i/steps, i = 1..steps, sum divided by steps + 1."""
    return _path_integrated_gradients(model, x, targets, np.arange(1, steps + 1) / steps, steps + 1,
                                      use_softmax=True, max_batch=max_batch)


def gene_level_attributions(attr: np.ndarray, pixel_index: list) -> list:
    """``attr`` [N, C, H, W] -> per omics [N, n_features] float32 (the value at each feature's pixel)."""
    return [attr[:, c, rows, cols].astype(np.float32) for c, (rows, cols) in enumerate(pixel_index)]
