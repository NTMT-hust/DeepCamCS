"""Train one fold; the fold model is a stochastic weight average of the same SGD run.

SGD (momentum, Nesterov): the learning rate rises linearly from ``lr * warmup_start_factor`` to ``lr`` over
``warmup_epochs``, then follows a cosine down to ``min_lr`` at the last epoch. From epoch swa_start_epoch+1 on
(31-50 by default) the weights after every epoch are added to a running average (AveragedModel); at the end
the BatchNorm statistics of the averaged weights are re-estimated on the training images and the averaged
model is the fold model. Focal loss, class-aware augmentation, backbone frozen for ``freeze_epochs``. With
guidance on, the mask is built on its schedule and the penalty is active from ``loss_start_epoch``. Nothing
is selected on the validation split: it is monitored and gives the guidance mask.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset

from .augmentation import ClassAwareAugmenter
from .config import Config
from .guidance import FeatureRelevance, MaskBuilder, format_mask_check, guided_loss, is_mask_epoch, penalty_objective
from .losses import FocalLoss, inverse_frequency_weights
from .metrics import compute_metrics
from .model import build_model


def to_float_image(img: torch.Tensor) -> torch.Tensor:
    """uint8 -> float in [0, 1]."""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()


class ImageDataset(Dataset):
    """In-memory uint8 images; items are (float image, label, position in this dataset)."""

    def __init__(self, images: torch.Tensor, labels: np.ndarray):
        self.images = images
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, i):
        return to_float_image(self.images[i]), self.labels[i], i


def eval_loader(images: torch.Tensor, labels: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(ImageDataset(images, labels), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple:
    model.eval()
    logits, labels = [], []
    for x, y, _ in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def sgd_schedule(model: torch.nn.Module, t) -> tuple:
    """SGD + linear warm-up (``warmup_epochs``) + cosine to ``min_lr`` at the last epoch. A warm-up at least
    as long as the run is skipped."""
    optimizer = torch.optim.SGD(model.parameters(), lr=t.lr, momentum=t.momentum, nesterov=t.nesterov,
                                weight_decay=t.weight_decay)
    warm = t.warmup_epochs if t.epochs > t.warmup_epochs else 0
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t.epochs - warm), eta_min=t.min_lr)
    if warm == 0:
        return optimizer, cosine
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=t.warmup_start_factor, total_iters=warm)
    return optimizer, torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warm])


def _train_epoch(model, loader, criterion, optimizer, device, augmenter, mask, lambda_reg, target,
                 penalty_samples: int, penalty_rng: torch.Generator) -> dict:
    """One pass over the training data. ``mask`` None = no penalty this epoch."""
    model.train()
    tot_loss = tot_reg = 0.0
    ys, probs = [], []
    for x, y, idx in loader:
        if augmenter is not None:
            x = augmenter(x, y, idx)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if mask is None:
            logits = model(x)
            loss = criterion(logits, y)
        else:
            logits, loss, reg = guided_loss(model, x, y, criterion, mask[y], lambda_reg, target, penalty_samples,
                                            penalty_rng)
            tot_reg += float(reg)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite training loss ({value})")
        loss.backward()
        optimizer.step()
        tot_loss += value
        ys.append(y.cpu())
        probs.append(torch.softmax(logits.detach().float(), 1).cpu())
    n = max(len(ys), 1)
    return {"loss": tot_loss / n, "reg": tot_reg / n, "y": torch.cat(ys).numpy(), "prob": torch.cat(probs).numpy()}


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    tot, n, ys, probs = 0.0, 0, [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        lg = model(x).float()
        tot += float(criterion(lg, y))
        n += 1
        ys.append(y.cpu())
        probs.append(torch.softmax(lg, 1).cpu())
    return {"loss": tot / max(n, 1), "y": torch.cat(ys).numpy(), "prob": torch.cat(probs).numpy()}


def train_fold(cfg: Config, images: torch.Tensor, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
               class_names: list, pixel_index: list, relevance: FeatureRelevance | None, device: torch.device,
               seed: int, logger: logging.Logger) -> dict:
    """Returns ``{"model": SWA fold model (eval mode), "history", "mask_log", "final_mask"}``."""
    t, g = cfg.train, cfg.guidance
    K = len(class_names)
    y_tr, y_va = labels[train_idx], labels[val_idx]

    ds_tr = ImageDataset(images[train_idx], y_tr)
    loader_tr = DataLoader(ds_tr, batch_size=t.batch_size, shuffle=True, generator=torch.Generator().manual_seed(int(seed)))
    loader_va = eval_loader(images[val_idx], y_va, t.eval_batch_size)
    augmenter = ClassAwareAugmenter(ds_tr.images, y_tr, seed, cfg.augment) if cfg.augment.enabled else None

    model = build_model(cfg.model, K, pixel_index, cfg.image.size).to(device)
    if t.freeze_epochs > 0:
        model.freeze_backbone()
    criterion = FocalLoss(alpha=inverse_frequency_weights(y_tr, K).to(device), gamma=t.focal_gamma).to(device)
    optimizer, scheduler = sgd_schedule(model, t)

    protected = relevance.protected_map(g.protect_top_fraction) if (g.enabled and relevance is not None) else None
    if g.enabled and g.protect_top_fraction > 0 and protected is None:
        raise ValueError("guidance.protect_top_fraction needs the ANOVA ranks of the training split")
    mask_builder = MaskBuilder(g, K, protected, max(1, cfg.attribution.max_batch // model.memory_factor))
    target = penalty_objective(class_names, g)
    penalty_rng = torch.Generator().manual_seed(int(seed) + 7919)       # draws of the penalised samples

    history = defaultdict(list)
    divergence = None
    mask, mask_log, swa = None, [], None
    for epoch in range(t.epochs):
        t0 = time.time()
        if t.freeze_epochs > 0 and epoch == t.freeze_epochs:
            model.unfreeze_backbone()
            logger.info("    [unfreezing backbone]")
        if epoch == t.swa_start_epoch:
            swa = AveragedModel(model)
            logger.info(f"    [swa] epochs {epoch + 1}-{t.epochs}: weights averaged on the same SGD run (no restart)")
        if g.enabled and is_mask_epoch(epoch, g):
            mask = _build_mask(mask_builder, model, loader_va, device, relevance, epoch, mask, mask_log, logger, g)
        penalise = g.enabled and g.lambda_reg > 0 and epoch >= g.loss_start_epoch and mask is not None

        try:
            tr = _train_epoch(model, loader_tr, criterion, optimizer, device, augmenter, mask if penalise else None,
                              g.lambda_reg, target, g.penalty_samples, penalty_rng)
        except FloatingPointError as e:
            divergence = {"epoch": epoch + 1, "lr": float(optimizer.param_groups[0]["lr"]),
                          "penalty": bool(penalise), "error": str(e)}
            logger.error(f"    [diverged] epoch {epoch + 1} (lr {optimizer.param_groups[0]['lr']:.3g}, "
                         f"penalty {'on' if penalise else 'off'}): {e}; stopping this fold and evaluating "
                         "the last usable weights")
            break
        va = _evaluate(model, loader_va, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        if swa is not None:
            swa.update_parameters(model)
        scheduler.step()

        trm = compute_metrics(tr["y"], tr["prob"].argmax(1), tr["prob"], class_names)
        vam = compute_metrics(va["y"], va["prob"].argmax(1), va["prob"], class_names)
        for split, res, mm in (("train", tr, trm), ("val", va, vam)):
            history[f"{split}_loss"].append(res["loss"])
            for k in ("accuracy", "f1_macro", "roc_auc", "balanced_accuracy"):
                history[f"{split}_{k}"].append(mm[k])
        history["train_reg"].append(tr["reg"])
        history["lr"].append(lr)
        history["guidance"].append(int(penalise))
        logger.info(f"    ep {epoch + 1:>2}/{t.epochs} lr={lr:.4g} | train loss={tr['loss']:.4f}"
                    + (f" reg={tr['reg']:.2e}" if penalise else "")
                    + f" acc={trm['accuracy']*100:.1f} F1={trm['f1_macro']*100:.1f} | val loss={va['loss']:.4f} "
                      f"acc={vam['accuracy']*100:.1f} F1={vam['f1_macro']*100:.1f} AUC={vam['roc_auc']:.3f} "
                      f"({time.time() - t0:.1f}s)")

    if swa is not None:
        update_bn(eval_loader(images[train_idx], y_tr, t.eval_batch_size), swa, device=device)
        model.load_state_dict(swa.module.state_dict())
    model.eval()
    if swa is not None:
        logger.info(f"    [swa] fold model = average of the weights after epochs {t.swa_start_epoch + 1}-{t.epochs}, "
                    f"BatchNorm re-estimated")
    else:
        logger.info("    [swa] no averaged weights available; evaluating the last usable training weights")
    return {"model": model, "history": dict(history), "mask_log": mask_log, "final_mask": mask,
            "divergence": divergence}


def _build_mask(builder, model, loader, device, relevance, epoch, current, mask_log, logger, g):
    """Rebuild the guidance mask; keeps ``current`` when no mask could be built."""
    t0 = time.time()
    if device.type == "cuda":
        # Release cached blocks around the build: on Windows an over-full GPU cache silently spills into
        # system RAM (50-100x slower) instead of raising an out-of-memory error.
        torch.cuda.empty_cache()
    new = builder.build(model, loader, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    st = dict(builder.last_stats, epoch=epoch + 1, seconds=round(time.time() - t0, 2), updated=new is not None)
    if new is None:
        logger.info(f"    [mask] epoch {epoch + 1}: no samples -> mask unchanged")
        mask_log.append(st)
        return current
    if relevance is not None:
        st["check"] = relevance.mask_check(new)
    mask_log.append(st)
    logger.info(f"    [mask] epoch {epoch + 1}: gradcam on val (n={st['n_samples']}, acc={st['accuracy']*100:.1f}%), "
                f"{st['mask_fraction']*100:.1f}% of pixels masked (one per class) ({st['seconds']:.1f}s)")
    if "check" in st:
        logger.info(f"    [mask-check] {format_mask_check(st['check'])}")
    return new
