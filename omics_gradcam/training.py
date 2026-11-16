"""Stage 3d - train one fold, in two phases (torch.optim.swa_utils for the second).

Phase 1, epochs 1 .. swa_start_epoch (1-75 by default):
    AdamW; the learning rate follows CosineAnnealingLR from ``lr`` down to ``min_lr``.
Phase 2, SWA, epochs swa_start_epoch+1 .. epochs (76-100 by default):
    a new SGD optimizer (momentum, Nesterov) started at the large learning rate ``swa_lr``, held constant
    by SWALR; after every epoch the weights are added to the running average (AveragedModel). At the end
    the BatchNorm statistics of the averaged weights are re-estimated on the training images (update_bn).
    The averaged model is the fold model.

Both phases use focal loss and class-aware augmentation; the backbone is frozen for the first
``freeze_epochs``; with guidance on, the mask is built on its schedule and the input-gradient penalty is
active from ``loss_start_epoch``. Nothing is selected on the validation split: it is only monitored (and
later used for calibration).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
import torch
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentation import ClassAwareAugmenter
from .config import Config
from .guidance import FeatureRelevance, MaskBuilder, format_mask_check, guided_loss, is_mask_epoch, penalty_objective
from .losses import FocalLoss, inverse_frequency_weights
from .metrics import compute_metrics
from .models import build_model


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


def train_loader(ds: ImageDataset, num_classes: int, batch_size: int, weighted: bool, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(int(seed))
    if not weighted:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    labels = ds.labels.numpy()
    weights = 1.0 / np.bincount(labels, minlength=num_classes).astype(np.float64)[labels]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True,
                                    generator=g)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple:
    model.eval()
    logits, labels = [], []
    for x, y, _ in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def adamw_phase(model: torch.nn.Module, t) -> tuple:
    """Phase 1: AdamW + CosineAnnealingLR (lr -> min_lr over the epochs before SWA)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.swa_start_epoch, eta_min=t.min_lr)
    return optimizer, scheduler


def swa_phase(model: torch.nn.Module, t) -> tuple:
    """Phase 2: SGD started at the large learning rate swa_lr, kept constant by SWALR, and the running
    average of the weights."""
    optimizer = torch.optim.SGD(model.parameters(), lr=t.swa_lr, momentum=t.swa_momentum, nesterov=t.swa_nesterov,
                                weight_decay=t.swa_weight_decay)
    scheduler = SWALR(optimizer, swa_lr=t.swa_lr, anneal_epochs=0)
    return optimizer, scheduler, AveragedModel(model)


def _train_epoch(model, loader, criterion, optimizer, device, augmenter, mask, lambda_reg, target) -> dict:
    """One pass over the training data; ``mask`` None = no penalty this epoch."""
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
            m = mask[y] if mask.dim() == x.dim() else mask          # class-conditional: each sample its class's mask
            logits, loss, reg = guided_loss(model, x, y, criterion, m, lambda_reg, target)
            tot_reg += float(reg)
        loss.backward()
        optimizer.step()
        tot_loss += float(loss.detach())
        ys.append(y.cpu())
        probs.append(torch.softmax(logits.detach().float(), 1).cpu())
    n = max(len(ys), 1)
    prob = torch.cat(probs).numpy()
    return {"loss": tot_loss / n, "reg": tot_reg / n, "y": torch.cat(ys).numpy(), "prob": prob}


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
    loader_tr = train_loader(ds_tr, K, t.batch_size, t.weighted_sampling, seed)
    loader_va = eval_loader(images[val_idx], y_va, t.eval_batch_size)
    loader_mask = (eval_loader(images[train_idx], y_tr, t.eval_batch_size) if g.mask_source == "train"
                   else loader_va)
    augmenter = ClassAwareAugmenter(ds_tr.images, y_tr, seed, cfg.augment) if cfg.augment.enabled else None

    model = build_model(cfg.model, K, pixel_index, cfg.image.size).to(device)
    if t.freeze_epochs > 0:
        model.freeze_backbone()
    alpha = inverse_frequency_weights(y_tr, K).to(device) if t.focal_alpha == "inverse_freq" else None
    criterion = FocalLoss(alpha=alpha, gamma=t.focal_gamma).to(device)
    optimizer, scheduler = adamw_phase(model, t)

    protected = relevance.protected_map(g.protect_top_fraction) if (g.enabled and relevance is not None) else None
    if g.enabled and g.protect_top_fraction > 0 and protected is None:
        raise ValueError("guidance.protect_top_fraction needs the feature relevance of the training split")
    mask_builder = MaskBuilder(g, K, protected, max(1, cfg.attribution.max_batch // model.memory_factor))
    target = penalty_objective(class_names, g)

    history = defaultdict(list)
    mask, mask_log, swa = None, [], None
    for epoch in range(t.epochs):
        t0 = time.time()
        if t.freeze_epochs > 0 and epoch == t.freeze_epochs:
            model.unfreeze_backbone()
            logger.info("    [unfreezing backbone]")
        if epoch == t.swa_start_epoch:
            optimizer, scheduler, swa = swa_phase(model, t)
            logger.info(f"    [swa] epochs {epoch + 1}-{t.epochs}: SGD at constant lr {t.swa_lr:g}, weights averaged")
        if g.enabled and is_mask_epoch(epoch, g):
            mask = _build_mask(mask_builder, model, loader_mask, device, relevance, epoch, mask, mask_log, logger, g)
        penalise = g.enabled and g.lambda_reg > 0 and epoch >= g.loss_start_epoch and mask is not None

        tr = _train_epoch(model, loader_tr, criterion, optimizer, device, augmenter, mask if penalise else None,
                          g.lambda_reg, target)
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

    # BatchNorm statistics of the averaged weights, on the (un-augmented) training images
    update_bn(eval_loader(images[train_idx], y_tr, t.eval_batch_size), swa, device=device)
    model.load_state_dict(swa.module.state_dict())
    model.eval()
    logger.info(f"    [swa] fold model = average of the weights after epochs {t.swa_start_epoch + 1}-{t.epochs}, "
                f"BatchNorm re-estimated")
    return {"model": model, "history": dict(history), "mask_log": mask_log, "final_mask": mask}


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
    logger.info(f"    [mask] epoch {epoch + 1}: {g.mask_method} on {g.mask_source} (n={st['n_samples']}, "
                f"acc={st['accuracy']*100:.1f}%), {st['mask_fraction']*100:.1f}% of pixels masked"
                f"{' (one per class)' if new.dim() == 4 else ''} ({st['seconds']:.1f}s)")
    if "check" in st:
        logger.info(f"    [mask-check] {format_mask_check(st['check'])}")
    return new
