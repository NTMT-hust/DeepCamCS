"""Stage 4 - attribution-guided input-gradient penalty ("right for the right reasons").

Training loss = focal loss + lambda * mean_b ||grad_x S(x_b) * m_b||^2 / |m_b|

* ``S`` (``guidance.penalty_target``):
    true_class   log p(y | x)                                     (original new method)
    all_classes  sum_c log p(c | x)                               (Ross et al. 2017)
    pair         log p(a | x) - log p(b | x) for samples labelled a or b (``penalty_pair``,
                 default LumA / LumB: the a-vs-b decision), log p(y | x) for the others
* ``m`` is the guidance mask: pixels the model uses for the wrong reasons (``MaskBuilder``), rebuilt
  on a schedule from the current model. Per sample, the top ``topk_ratio`` of its gene pixels by
  "wrong-reason" score is selected: -attribution if the prediction is correct (evidence against the
  true class), +attribution if wrong (evidence for the wrong class); the mask keeps the pixels selected
  in more than ``frequency_threshold`` of the samples. Attribution = Grad-CAM (default) or sparse IG.
  Options: ``class_conditional`` (one mask per class, each sample penalised with its class's mask),
  ``protect_top_fraction`` (the most label-associated genes by ANOVA F on the training split are never
  masked) and ``mask_background`` (also mask the empty pixels, as in the original rule).

The penalty gradient is computed exactly with one extra forward-mode (JVP) pass instead of PyTorch's
double backward, which is ~40x slower for EfficientNet's depthwise convolutions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import f_classif

from .attribution import gradcam, sparse_integrated_gradients
from .config import GuidanceConfig


# ==============================================================================================
# Penalty
# ==============================================================================================
def penalty_objective(class_names: list, cfg: GuidanceConfig):
    """``"true_class"`` / ``"all_classes"`` or ``("pair", a, b)`` with class indices."""
    if cfg.penalty_target != "pair":
        return cfg.penalty_target
    missing = [c for c in cfg.penalty_pair if c not in class_names]
    if missing:
        raise ValueError(f"guidance.penalty_pair {cfg.penalty_pair}: unknown classes {missing} (have {class_names})")
    return ("pair", class_names.index(cfg.penalty_pair[0]), class_names.index(cfg.penalty_pair[1]))


def _objective(log_probs: torch.Tensor, labels: torch.Tensor, target) -> torch.Tensor:
    """Scalar S summed over the batch, from per-class log-probabilities [B, K]."""
    if target == "true_class":
        return log_probs.gather(1, labels.unsqueeze(1)).sum()
    if target == "all_classes":
        return log_probs.sum()
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


@contextmanager
def _dual_pass(model: nn.Module):
    """Primitive BatchNorm for the dual pass; BatchNorm counters left untouched."""
    saved = [(m, m.num_batches_tracked.clone()) for m in model.modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.num_batches_tracked is not None]
    original = F.batch_norm
    F.batch_norm = _batch_norm_primitive
    try:
        yield
    finally:
        F.batch_norm = original
        for m, n in saved:
            m.num_batches_tracked.copy_(n)


def guided_loss(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, criterion: nn.Module,
                mask: torch.Tensor, lambda_reg: float, target="true_class") -> tuple:
    """``criterion(model(x), y) + lambda * penalty``, differentiable in the model parameters.

    With g = d/dx S and v = stopgrad(g * m) / (B |m|): penalty = <g * m, g * m> / (B |m|) and
    d penalty / d theta = 2 d/d theta <g, v>; <g, v> is a Jacobian-vector product of S in direction v,
    evaluated in a second forward pass with forward-mode AD. That pass replays the same dropout masks
    (RNG state restored) and leaves the BatchNorm running statistics alone, so it computes exactly the
    same function as the first pass.

    ``mask``: [C, H, W] (one mask for the batch) or [B, C, H, W] (one per sample).
    Returns ``(logits, total_loss, penalty_value)``.
    """
    dev = x.device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None

    xr = x.detach().requires_grad_(True)
    logits = model(xr)
    loss = criterion(logits, labels)
    (g,) = torch.autograd.grad(_objective(torch.log_softmax(logits, 1), labels, target), xr, retain_graph=True)

    m = mask.to(dtype=g.dtype, device=dev)
    per_sample = m.dim() == g.dim()
    if per_sample:                                                      # v_b = g_b m_b / (B |m_b|)
        size = m.flatten(1).sum(1)
        if float(size.sum()) == 0:
            return logits, loss, g.new_zeros(())
        w = torch.where(size > 0, 1.0 / (size.clamp_min(1) * x.size(0)), torch.zeros_like(size))
        gm = g * m
        reg_value = (gm.detach().pow(2).flatten(1).sum(1) * w).sum()
        v = (gm * w.view(-1, *([1] * (g.dim() - 1)))).detach()
    else:                                                               # v = g m, divided by B |m| below
        denom = m.sum() * x.size(0)
        if float(denom) == 0:
            return logits, loss, g.new_zeros(())
        v = (g * m).detach()
        reg_value = v.pow(2).sum() / denom

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, dev)
    with _dual_pass(model), fwAD.dual_level():
        out = model(fwAD.make_dual(x.detach(), v))
        jvp = fwAD.unpack_dual(_objective(_log_softmax_primitive(out), labels, target)).tangent
    surrogate = 2.0 * jvp if per_sample else 2.0 * jvp / denom
    penalty = surrogate - surrogate.detach() + reg_value                # value = reg, gradient = d reg/d theta
    return logits, loss + lambda_reg * penalty, reg_value


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

    @classmethod
    def from_training_split(cls, matrices: list, labels: np.ndarray, pixel_index: list, omics: list,
                            shape: tuple) -> "FeatureRelevance":
        ranks = []
        for Q in matrices:
            with np.errstate(divide="ignore", invalid="ignore"):
                f, _ = f_classif(np.asarray(Q, dtype=np.float64).T, np.asarray(labels))
            order = np.argsort(-np.nan_to_num(f, nan=0.0, posinf=np.finfo(np.float64).max), kind="stable")
            r = np.empty(len(order), dtype=np.int64)
            r[order] = np.arange(len(order))
            ranks.append(r)
        return cls(list(omics), [(np.asarray(a), np.asarray(b)) for a, b in pixel_index], ranks, tuple(shape))

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
        """Share of each omics' features inside the mask, overall and by ANOVA-rank bin (per-class
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
def topk_sample_mask(score: torch.Tensor, x: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    """Background (x == 0) plus the top ``topk_ratio`` of the gene pixels (x != 0) by ``score``."""
    background = x == 0
    scores = score.flatten().masked_fill(background.flatten(), float("-inf"))
    n_genes = int((~background).sum())
    mask = background.flatten().clone()
    if n_genes > 0:
        mask[torch.topk(scores, max(1, int(topk_ratio * n_genes))).indices] = True
    return mask.view_as(x)


class MaskBuilder:
    """Builds the guidance mask from the current model on the mask-source samples."""

    def __init__(self, cfg: GuidanceConfig, num_classes: int, protected: np.ndarray | None, max_batch: int = 72):
        self.cfg = cfg
        self.num_classes = num_classes if cfg.class_conditional else None
        self.protected = protected
        self.max_batch = max_batch
        self.last_stats: dict = {}

    def _attributions(self, model: nn.Module, x: torch.Tensor) -> tuple:
        """(score [B, C, H, W] for the predicted class, predictions [B])."""
        if self.cfg.mask_method == "gradcam":
            cam, logits = gradcam(model, x)
            return cam, logits.argmax(1)
        with torch.no_grad():
            preds = model(x).argmax(1)
        return sparse_integrated_gradients(model, x, preds, self.cfg.ig_steps, self.max_batch), preds

    def build(self, model: nn.Module, loader, device: torch.device) -> torch.Tensor | None:
        """bool mask [C, H, W], or [K, C, H, W] with ``class_conditional`` (a class without samples
        gets the mask of all samples)."""
        model.eval()
        counts: dict = {}                       # group (None = all samples, or a class) -> [sum, n]
        n = n_correct = 0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            score, preds = self._attributions(model, x)
            for i in range(x.size(0)):
                correct = bool(preds[i] == y[i])
                sel = topk_sample_mask(-score[i] if correct else score[i], x[i], self.cfg.topk_ratio)
                if not self.cfg.mask_background:
                    sel = sel & (x[i] != 0)
                groups = (None,) if self.num_classes is None else (None, int(y[i]))
                for key in groups:
                    c = counts.setdefault(key, [0, 0])
                    c[0] = c[0] + sel.float()
                    c[1] += 1
                n += 1
                n_correct += int(correct)
        mask = self._frequency_mask(counts)
        if mask is not None and self.protected is not None:
            mask = mask & ~torch.as_tensor(self.protected, dtype=torch.bool, device=mask.device)
        self.last_stats = {"n_samples": n, "accuracy": n_correct / max(n, 1),
                           "mask_fraction": float("nan") if mask is None else float(mask.float().mean())}
        return mask

    def _frequency_mask(self, counts: dict) -> torch.Tensor | None:
        if None not in counts:
            return None
        thr = self.cfg.frequency_threshold
        overall = counts[None][0] / counts[None][1] > thr
        if self.num_classes is None:
            return overall
        return torch.stack([counts[k][0] / counts[k][1] > thr if k in counts else overall
                            for k in range(self.num_classes)])


def is_mask_epoch(epoch: int, cfg: GuidanceConfig) -> bool:
    return epoch == cfg.mask_build_epoch or (epoch >= cfg.mask_refresh_start and epoch % cfg.mask_refresh_every == 0)
