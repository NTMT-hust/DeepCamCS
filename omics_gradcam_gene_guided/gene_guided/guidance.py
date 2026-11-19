"""Attribution-guided input-gradient penalty ("right for the right reasons").

Training loss = focal loss (whole batch) + lambda * mean_b ||grad_x S(x_b) * m_b||^2 / |m_b|  (b in a random
subset of ``penalty_samples`` samples of the batch)

* ``S``: log p(a | x) - log p(b | x) for samples labelled a or b (``penalty_pair``, LumA / LumB: the a-vs-b
  decision the model gets most wrong), log p(y | x) for the others.
* ``m_b``: the guidance mask of sample b's class - gene pixels the model uses for the wrong reasons
  (``MaskBuilder``), rebuilt on a schedule from the current model with Grad-CAM on the validation split.
  Per sample, the top ``topk_ratio`` of its gene pixels by "wrong-reason" score is selected: -Grad-CAM if
  the prediction is correct (evidence against the true class), +Grad-CAM if wrong (evidence for the wrong
  class); each class's mask keeps the pixels selected in more than ``frequency_threshold`` of its samples.
  The top ``protect_top_fraction`` of each omics' genes by ANOVA F on the training split are never masked,
  and empty pixels never are.

The penalty gradient is computed exactly with one extra forward-mode (JVP) pass instead of PyTorch's
double backward, which is ~40x slower for EfficientNet's depthwise convolutions. The penalty passes run in
training mode on the drawn samples (BatchNorm with their batch statistics) without touching the BatchNorm
running statistics, which only the focal-loss pass updates.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F

from .attribution import gradcam
from .config import GuidanceConfig


# ==============================================================================================
# Penalty
# ==============================================================================================
def penalty_objective(class_names: list, cfg: GuidanceConfig) -> tuple:
    """``("pair", a, b)`` with the class indices of ``penalty_pair``."""
    missing = [c for c in cfg.penalty_pair if c not in class_names]
    if missing:
        raise ValueError(f"guidance.penalty_pair {cfg.penalty_pair}: unknown classes {missing} (have {class_names})")
    return ("pair", class_names.index(cfg.penalty_pair[0]), class_names.index(cfg.penalty_pair[1]))


def _objective(log_probs: torch.Tensor, labels: torch.Tensor, target: tuple) -> torch.Tensor:
    """Scalar S summed over the batch, from per-class log-probabilities [B, K]."""
    _, a, b = target
    own = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    in_pair = (labels == a) | (labels == b)
    return torch.where(in_pair, log_probs[:, a] - log_probs[:, b], own).sum()


def _log_softmax_primitive(out: torch.Tensor) -> torch.Tensor:
    """log-softmax from primitive ops: the fused op's forward-AD rule is not reverse-differentiable."""
    z = out - out.detach().amax(dim=1, keepdim=True)
    return z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))


def _batch_norm_primitive(input, running_mean, running_var, weight=None, bias=None, training=False,
                          momentum=0.1, eps=1e-5):
    """BatchNorm from primitive ops, never updating the running statistics. The fused op's forward-AD
    rule drops the batch-statistics term from its reverse-mode derivative (wrong in train mode)."""
    shape = (1, -1) + (1,) * (input.dim() - 2)
    if training or running_mean is None:
        dims = [0] + list(range(2, input.dim()))
        mean = input.mean(dim=dims, keepdim=True)
        var = (input - mean).pow(2).mean(dim=dims, keepdim=True)
    else:
        mean, var = running_mean.view(shape), running_var.view(shape)
    out = (input - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.view(shape)
    if bias is not None:
        out = out + bias.view(shape)
    return out


def _bn_counters(model: nn.Module) -> list:
    return [(m, m.num_batches_tracked.clone()) for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.num_batches_tracked is not None]


@contextmanager
def _batch_statistics_only(model: nn.Module):
    """Fused BatchNorm normalising with the batch statistics (training mode) without reading or updating the
    running statistics, so nothing is changed in place; BatchNorm counters left untouched."""
    saved = _bn_counters(model)
    original = F.batch_norm

    def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
        if training:
            return original(input, None, None, weight, bias, True, momentum, eps)
        return original(input, running_mean, running_var, weight, bias, training, momentum, eps)
    F.batch_norm = batch_norm
    try:
        yield
    finally:
        F.batch_norm = original
        for m, n in saved:
            m.num_batches_tracked.copy_(n)


@contextmanager
def _dual_pass(model: nn.Module):
    """Primitive BatchNorm for the dual pass; BatchNorm counters left untouched."""
    saved = _bn_counters(model)
    original = F.batch_norm
    F.batch_norm = _batch_norm_primitive
    try:
        yield
    finally:
        F.batch_norm = original
        for m, n in saved:
            m.num_batches_tracked.copy_(n)


def penalty(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, target: tuple) -> tuple:
    """``(term, value)``: ``value`` = mean_b ||g_b m_b||^2 / |m_b| for the batch ``x`` with one mask per
    sample ([B, C, H, W]); ``term`` has that value and its exact gradient in the model parameters.

    With g = d/dx S and v_b = stopgrad(g_b m_b) / (B |m_b|): value = sum_b <g_b m_b, v_b> and
    d value / d theta = 2 d/d theta <g, v>; <g, v> is a Jacobian-vector product of S in direction v,
    evaluated in a second forward pass with forward-mode AD. That pass replays the same dropout masks
    (RNG state restored) and leaves the BatchNorm running statistics alone, so it computes exactly the
    same function as the first pass.
    """
    dev = x.device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None

    xr = x.detach().requires_grad_(True)
    logits = model(xr)
    (g,) = torch.autograd.grad(_objective(torch.log_softmax(logits, 1), labels, target), xr)
    m = mask.to(dtype=g.dtype, device=dev)
    size = m.flatten(1).sum(1)
    if float(size.sum()) == 0:
        return g.new_zeros(()), g.new_zeros(())
    w = torch.where(size > 0, 1.0 / (size.clamp_min(1) * x.size(0)), torch.zeros_like(size))
    gm = g * m
    value = (gm.detach().pow(2).flatten(1).sum(1) * w).sum()
    v = (gm * w.view(-1, *([1] * (g.dim() - 1)))).detach()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, dev)
    with _dual_pass(model), fwAD.dual_level():
        out = model(fwAD.make_dual(x.detach(), v))
        jvp = fwAD.unpack_dual(_objective(_log_softmax_primitive(out), labels, target)).tangent
    surrogate = 2.0 * jvp
    return surrogate - surrogate.detach() + value, value       # value = penalty, gradient = d penalty / d theta


def guided_loss(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, criterion: nn.Module, mask: torch.Tensor,
                lambda_reg: float, target: tuple, n_samples: int, generator: torch.Generator | None = None) -> tuple:
    """``criterion(model(x), y) + lambda * penalty``; the penalty uses ``min(n_samples, B)`` samples drawn
    without replacement (``generator``, CPU) and their masks (``mask`` [B, C, H, W]).

    The first penalty pass uses the fused BatchNorm with batch statistics only, the forward-mode pass the
    primitive one, so neither updates the BatchNorm running statistics: they are updated once per step, by the
    focal-loss pass (an in-place update before the backward pass would also invalidate the focal-loss graph).
    The subset estimates the batch mean without bias; memory: the penalty passes cost ~n_samples images.
    Returns ``(logits, total_loss, penalty_value)``.
    """
    dev = x.device
    logits = model(x)
    loss = criterion(logits, labels)
    k = min(int(n_samples), x.size(0))
    idx = torch.randperm(x.size(0), generator=generator)[:k].to(dev)
    with _batch_statistics_only(model):
        term, value = penalty(model, x[idx].detach(), labels[idx], mask[idx], target)
    return logits, loss + lambda_reg * term, value


# ==============================================================================================
# Feature relevance: protected genes and the mask check
# ==============================================================================================
RANK_BINS = ((0, 50, "top 50"), (50, 200, "51-200"), (200, 1000, "201-1000"), (1000, None, ">1000"))


@dataclass
class FeatureRelevance:
    """ANOVA-F rank of every feature on a fold's training split, mapped to its pixel."""
    omics: list
    pixels: list            # per omics (rows, cols)
    rank: list              # per omics [n_features], 0 = most label-associated
    shape: tuple            # (C, H, W)

    def protected_map(self, fraction: float) -> np.ndarray | None:
        """bool [C, H, W]: pixels of the top ``fraction`` of each omics' features (None if 0)."""
        if fraction <= 0:
            return None
        out = np.zeros(self.shape, dtype=bool)
        for c, ((rows, cols), r) in enumerate(zip(self.pixels, self.rank)):
            keep = r < int(round(fraction * len(r)))
            out[c, rows[keep], cols[keep]] = True
        return out

    def mask_check(self, mask: torch.Tensor) -> dict:
        """Share of each omics' features inside the mask, overall and by ANOVA-rank bin (the per-class
        masks are averaged). A mask of wrong reasons should cover the top-ranked genes much less."""
        m = mask.detach().cpu().numpy().astype(bool)
        m = m[None] if m.ndim == 3 else m
        out = {}
        for c, (name, (rows, cols), r) in enumerate(zip(self.omics, self.pixels, self.rank)):
            inside = m[:, c, rows, cols].mean(axis=0)
            d = {"all": float(inside.mean())}
            for lo, hi, label in RANK_BINS:
                sel = (r >= lo) & (r < (len(r) if hi is None else hi))
                d[label] = float(inside[sel].mean()) if sel.any() else float("nan")
            out[name] = d
        return out


def format_mask_check(check: dict) -> str:
    parts = [f"{name} {d['top 50']*100:.0f}/{d['>1000']*100:.0f}/{d['all']*100:.0f}%" for name, d in check.items()]
    return "features masked, top-50 / rank>1000 / all by ANOVA (train): " + " | ".join(parts)


# ==============================================================================================
# Mask
# ==============================================================================================
def topk_gene_mask(score: torch.Tensor, x: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    """The top ``topk_ratio`` of the gene pixels (x != 0) of one sample by ``score``."""
    background = x == 0
    scores = score.flatten().masked_fill(background.flatten(), float("-inf"))
    n_genes = int((~background).sum())
    mask = torch.zeros_like(background.flatten())
    if n_genes > 0:
        mask[torch.topk(scores, max(1, int(topk_ratio * n_genes))).indices] = True
    return mask.view_as(x)


class MaskBuilder:
    """Builds the per-class guidance masks from the current model on the validation samples."""

    def __init__(self, cfg: GuidanceConfig, num_classes: int, protected: np.ndarray | None, max_batch: int = 72):
        self.cfg = cfg
        self.num_classes = num_classes
        self.protected = protected
        self.max_batch = max_batch
        self.last_stats: dict = {}

    def _gradcam(self, model: nn.Module, x: torch.Tensor) -> tuple:
        """(Grad-CAM [B, C, H, W] for the predicted class, predictions [B]), in chunks of at most
        ``max_batch`` images (eval mode: the same result as one pass) so the three trunks fit in memory."""
        cams, logits = [], []
        for s0 in range(0, x.size(0), self.max_batch):
            cam, lg = gradcam(model, x[s0:s0 + self.max_batch])
            cams.append(cam)
            logits.append(lg)
        return torch.cat(cams), torch.cat(logits).argmax(1)

    def build(self, model: nn.Module, loader, device: torch.device) -> torch.Tensor | None:
        """bool masks [K, C, H, W] (a class without samples gets the mask of all samples)."""
        model.eval()
        counts: dict = {}                       # group (None = all samples, or a class) -> [sum, n]
        n = n_correct = 0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            score, preds = self._gradcam(model, x)
            for i in range(x.size(0)):
                correct = bool(preds[i] == y[i])
                sel = topk_gene_mask(-score[i] if correct else score[i], x[i], self.cfg.topk_ratio)
                for key in (None, int(y[i])):
                    c = counts.setdefault(key, [0, 0])
                    c[0] = c[0] + sel.float()
                    c[1] += 1
                n += 1
                n_correct += int(correct)
        if None not in counts:
            mask = None
        else:
            thr = self.cfg.frequency_threshold
            overall = counts[None][0] / counts[None][1] > thr
            mask = torch.stack([counts[k][0] / counts[k][1] > thr if k in counts else overall
                                for k in range(self.num_classes)])
            if self.protected is not None:
                mask = mask & ~torch.as_tensor(self.protected, dtype=torch.bool, device=mask.device)
        self.last_stats = {"n_samples": n, "accuracy": n_correct / max(n, 1),
                           "mask_fraction": float("nan") if mask is None else float(mask.float().mean())}
        return mask


def is_mask_epoch(epoch: int, cfg: GuidanceConfig) -> bool:
    return epoch == cfg.mask_build_epoch or (epoch >= cfg.mask_refresh_start and epoch % cfg.mask_refresh_every == 0)
