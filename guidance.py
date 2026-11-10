"""Guidance mask for the "Right for the Right Reasons" input-gradient penalty.

The mask marks input pixels the model uses for the wrong reasons:
  A. per-sample attribution for the predicted class (Grad-CAM or sparse IG),
  B. sign flip by correctness (-score if correct, +score if wrong),
  C. per-sample top-k among gene pixels (x != 0),
  D. frequency consensus across samples (optionally per true class),
  E. removal of ANOVA-protected (label-informative) genes.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_selection import f_classif

from models import unwrap_model
from sparse_integrated_gradients import SparseIntegratedGradients


def batched_gradcam(model, x, class_idx):
    """Raw Grad-CAM (no ReLU, no normalisation), upsampled to [B, C, H, W]."""
    net = unwrap_model(model)
    with torch.enable_grad():
        acts = net.forward_features(x)
        if acts.ndim != 4:
            raise ValueError("Grad-CAM needs a convolutional backbone with 4D feature maps.")
        logits = net.forward_head(acts)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, acts)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    return cam.detach().expand(-1, x.size(1), -1, -1)


class FeatureRelevance:
    """ANOVA F-statistic of every pixel (per omics channel) vs. training labels."""

    def __init__(self, protected):
        self.protected = protected  # bool tensor [C, H, W]

    @classmethod
    def from_training_split(cls, dataset, protect_top_fraction=0.5):
        images = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()
        labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])
        n, c, h, w = images.shape
        protected = np.zeros((c, h * w), dtype=bool)
        if protect_top_fraction > 0:
            for ch in range(c):
                flat = images[:, ch].reshape(n, -1)
                gene_idx = np.where((flat != 0).any(axis=0))[0]
                if gene_idx.size == 0:
                    continue
                with np.errstate(all="ignore"):
                    f_stat, _ = f_classif(flat[:, gene_idx], labels)
                f_stat = np.nan_to_num(f_stat, nan=0.0, posinf=0.0)
                k = max(1, math.ceil(protect_top_fraction * gene_idx.size))
                top = gene_idx[np.argsort(-f_stat)[:k]]
                protected[ch, top] = True
        return cls(torch.from_numpy(protected.reshape(c, h, w)))


def topk_sample_mask(score, x, topk_ratio, mask_background=False):
    """Per-sample top-k gene pixels with the highest wrong-reason score. Shapes [B, C, H, W]."""
    b = score.size(0)
    flat_score = score.reshape(b, -1)
    gene = (x.reshape(b, -1) != 0)
    masked = flat_score.masked_fill(~gene, float("-inf"))
    sel = torch.zeros_like(gene)
    for i in range(b):
        n_genes = int(gene[i].sum().item())
        if n_genes == 0:
            continue
        k = max(1, int(topk_ratio * n_genes))
        idx = masked[i].topk(k).indices
        sel[i, idx] = True
    if mask_background:
        sel |= ~gene
    return sel.view_as(x)


class MaskBuilder:
    def __init__(
        self,
        num_classes,
        relevance=None,
        method="gradcam",
        topk_ratio=0.22,
        frequency_threshold=0.2,
        class_conditional=True,
        mask_background=False,
        ig_steps=8,
    ):
        self.num_classes = num_classes
        self.relevance = relevance
        self.method = method
        self.topk_ratio = topk_ratio
        self.frequency_threshold = frequency_threshold
        self.class_conditional = class_conditional
        self.mask_background = mask_background
        self.ig_steps = ig_steps

    def _attributions(self, model, x, pred):
        if self.method == "gradcam":
            return batched_gradcam(model, x, pred)
        if self.method == "ig":
            ig = SparseIntegratedGradients(model, x.device, steps=self.ig_steps)
            return torch.cat(
                [ig.attribute(x[i:i + 1], int(pred[i].item())).detach() for i in range(x.size(0))]
            )
        raise ValueError(f"Unknown mask method: {self.method}")

    def build(self, model, dataloader, device):
        """Return a float mask [K, C, H, W] (class conditional) or [1, C, H, W]."""
        model.to(device)
        model.eval()
        sel_all, label_all = [], []

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            with torch.no_grad():
                pred = model(inputs).argmax(dim=1)
            score = self._attributions(model, inputs, pred)
            correct = (pred == labels).view(-1, 1, 1, 1)
            wrong_reason = torch.where(correct, -score, score)
            sel = topk_sample_mask(wrong_reason, inputs, self.topk_ratio, self.mask_background)
            sel_all.append(sel.cpu())
            label_all.append(labels.cpu())

        model.zero_grad()
        sel_all = torch.cat(sel_all).float()
        label_all = torch.cat(label_all)

        overall = sel_all.mean(dim=0) > self.frequency_threshold
        if self.class_conditional:
            masks = []
            for k in range(self.num_classes):
                members = sel_all[label_all == k]
                masks.append(
                    members.mean(dim=0) > self.frequency_threshold if len(members) > 0 else overall
                )
            mask = torch.stack(masks)
        else:
            mask = overall.unsqueeze(0)

        if self.relevance is not None:
            mask = mask & ~self.relevance.protected.unsqueeze(0)
        return mask.float().to(device)


def guided_penalty(grads, mask, labels):
    """Mean over the batch of ||g * m||^2 / |m|, using each sample's true-class mask."""
    batch_mask = mask[labels] if mask.size(0) > 1 else mask.expand(grads.size(0), -1, -1, -1)
    num = ((grads * batch_mask) ** 2).flatten(1).sum(dim=1)
    den = batch_mask.flatten(1).sum(dim=1).clamp(min=1.0)
    return (num / den).mean()
