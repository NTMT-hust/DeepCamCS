"""Checks for every stage, plus an end-to-end run on a small synthetic dataset (CPU, a few minutes;
the end-to-end run uses the ImageNet weights, downloaded by timm on first use).

    python tests/test_pipeline.py          (or: python -m pytest tests)
"""
from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omics_gradcam.calibration import build_calibrator  # noqa: E402
from omics_gradcam.config import ImageConfig, config_from_dict, load_config  # noqa: E402
from omics_gradcam.guidance import MaskBuilder, guided_loss, _objective  # noqa: E402
from omics_gradcam.imaging import OmicsImageTransformer  # noqa: E402
from omics_gradcam.losses import FocalLoss  # noqa: E402
from omics_gradcam.models import GapClassifier, GeneReadoutClassifier  # noqa: E402
from omics_gradcam.splits import make_cv_splits  # noqa: E402

SIZE = 64


def synthetic_matrices(n_samples=40, sizes=(150, 140, 130), seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n_samples) % 4
    mats = []
    for n in sizes:
        effect = rng.normal(size=(n, 4)) * (rng.random((n, 1)) < 0.3)
        mats.append((rng.normal(size=(n, 1)) + effect[:, y] + rng.normal(scale=0.5, size=(n, n_samples))).astype(np.float32))
    return mats, y


def small_images(seed=0):
    mats, y = synthetic_matrices(seed=seed)
    tfm = OmicsImageTransformer(["a", "b", "c"], ImageConfig(size=SIZE), random_state=seed).fit(mats)
    return torch.from_numpy(tfm.transform(mats)).float() / 255, torch.as_tensor(y), tfm.pixel_index()


# ---------------------------------------------------------------------------------------------- 1
def test_config():
    cfg = load_config(ROOT / "configs" / "brca.yaml", ["model.head=gene", "splits.seeds=[1, 2]"])
    assert cfg.model.head == "gene" and cfg.splits.seeds == [1, 2] and cfg.train.swa_start_epoch == 75
    for bad in ({"model": {"head": "attention"}}, {"guidance": {"topk_ratio": 0}}, {"train": {"lr_typo": 1}},
                {"guidance": {"enabled": "maybe"}}, {"train": {"swa_start_epoch": 100}}, {"train": {"swa_start_epoch": 0}}, {"other": {}}):
        try:
            config_from_dict(bad)
        except (ValueError, KeyError):
            continue
        raise AssertionError(f"invalid config accepted: {bad}")


def test_splits():
    y = np.repeat(np.arange(4), [20, 10, 25, 15])
    splits = make_cv_splits(y, seed=3, n_folds=5, val_fraction=0.2)
    tested = np.concatenate([s.test for s in splits])
    assert sorted(tested) == list(range(len(y)))
    for s in splits:
        assert not (set(s.train) & set(s.val)) and not (set(s.train) & set(s.test))


# ---------------------------------------------------------------------------------------------- 2
def test_imaging():
    mats, _ = synthetic_matrices()
    tfm = OmicsImageTransformer(["a", "b", "c"], ImageConfig(size=SIZE, reserve_zero=True), random_state=0).fit(mats)
    images = tfm.transform(mats)
    assert images.shape == (40, 3, SIZE, SIZE) and images.dtype == np.uint8
    for c, (rows, cols) in enumerate(tfm.pixel_index()):
        assert len(set(zip(rows, cols))) == len(rows), "every feature needs its own pixel"
        gene = np.zeros((SIZE, SIZE), bool)
        gene[rows, cols] = True
        assert (images[:, c][:, gene] >= 1).all(), "reserve_zero: gene pixels must be >= 1"
        assert (images[:, c][:, ~gene] == 0).all(), "empty pixels must be 0"
    alone = tfm.transform([m[:, :1] for m in mats])                   # per-sample normalisation
    assert np.array_equal(alone[0], images[0])


# ---------------------------------------------------------------------------------------------- 3
def test_gene_readout():
    x, _, pix = small_images()
    model = GeneReadoutClassifier("efficientnet_b1", 4, pix, SIZE, pretrained=False, dropout=0.0).eval()
    with torch.no_grad():
        z = model.gene_evidence(x[:4])
        assert torch.allclose(model(x[:4]), z.mean(-1) + model.bias, atol=1e-5), "init = average gene evidence"
        # taps == bilinear interpolation of the class-evidence maps at the gene pixels
        h = model.bn1(model.conv_stem(x[:4].reshape(12, 1, SIZE, SIZE)))
        feats = []
        for i, block in enumerate(model.blocks):
            h = block(h)
            if i in model.taps:
                feats.append(h.view(4, 3, *h.shape[1:]))
        rows, cols = (torch.as_tensor(a) for a in pix[0])
        ref = sum(F.interpolate(model.proj[i][0](f[:, 0]), size=(SIZE, SIZE), mode="bilinear",
                                align_corners=False)[:, :, rows, cols] for i, f in enumerate(feats))
        assert torch.allclose(z[..., :len(rows)], ref, atol=1e-4), "gene taps != bilinear interpolation"
        # the trunk of omics 1 only affects omics 1
        for p in model.blocks[2].mods[1].parameters():
            p.add_(torch.randn_like(p) * 0.1)
        z2 = model.gene_evidence(x[:4])
    n0, n1 = len(pix[0][0]), len(pix[1][0])
    assert torch.allclose(z2[..., :n0], z[..., :n0]) and not torch.allclose(z2[..., n0:n0 + n1], z[..., n0:n0 + n1])


def test_training_schedule():
    """AdamW with a cosine learning rate, then SGD at the constant SWA learning rate with weight averaging."""
    from omics_gradcam.config import TrainConfig
    from omics_gradcam.training import adamw_phase, swa_phase
    t = TrainConfig(epochs=10, swa_start_epoch=6, lr=1e-3, min_lr=1e-5, swa_lr=0.05)
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = adamw_phase(model, t)
    assert isinstance(optimizer, torch.optim.AdamW)
    lrs, swa = [], None
    for epoch in range(t.epochs):
        if epoch == t.swa_start_epoch:
            optimizer, scheduler, swa = swa_phase(model, t)
            assert isinstance(optimizer, torch.optim.SGD)
        lrs.append(optimizer.param_groups[0]["lr"])
        model(torch.randn(4, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        if swa is not None:
            swa.update_parameters(model)
        scheduler.step()
    assert lrs[0] == 1e-3 and all(a > b for a, b in zip(lrs[:6], lrs[1:6])), lrs   # cosine decay
    assert all(abs(lr - 0.05) < 1e-12 for lr in lrs[6:]), lrs                      # constant SWA lr
    assert int(swa.n_averaged) == 4


# ---------------------------------------------------------------------------------------------- 4
def _double_backward_loss(model, x, y, criterion, mask, lam, target):
    """Reference: the penalty by double backward (slow, obviously correct)."""
    xr = x.clone().requires_grad_(True)
    logits = model(xr)
    (g,) = torch.autograd.grad(_objective(torch.log_softmax(logits, 1), y, target), xr, create_graph=True)
    if mask.dim() == g.dim():
        reg = ((g.pow(2) * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()
    else:
        reg = ((g.pow(2) * mask).sum(dim=(1, 2, 3)) / mask.sum()).mean()
    return criterion(logits, y) + lam * reg


def test_guided_loss_equals_double_backward():
    x, y, pix = small_images()
    x, y = x[:6], y[:6]
    criterion = FocalLoss(alpha=torch.tensor([0.4, 0.3, 0.2, 0.1]))
    shared = torch.rand(3, SIZE, SIZE) > 0.9
    per_sample = torch.rand(6, 3, SIZE, SIZE) > 0.9
    for name, model in (("gap", GapClassifier("efficientnet_b1", 4, pretrained=False, dropout=0.5)),
                        ("gene", GeneReadoutClassifier("efficientnet_b1", 4, pix, SIZE, pretrained=False, dropout=0.5))):
        model.train()
        for mask, target in ((shared, "true_class"), (per_sample, ("pair", 2, 3)), (shared, "all_classes")):
            m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
            torch.manual_seed(1)
            _double_backward_loss(m1, x, y, criterion, mask.float(), 100.0, target).backward()
            torch.manual_seed(1)
            guided_loss(m2, x, y, criterion, mask, 100.0, target)[1].backward()
            g1 = torch.cat([p.grad.flatten() for p in m1.parameters() if p.grad is not None])
            g2 = torch.cat([p.grad.flatten() for p in m2.parameters() if p.grad is not None])
            rel = float((g1 - g2).norm() / g1.norm())
            assert rel < 1e-3, f"{name} {target}: JVP penalty gradient differs from double backward (rel {rel:.1e})"
            for (_, b1), (_, b2) in zip(m1.named_buffers(), m2.named_buffers()):
                assert torch.equal(b1, b2), "the dual pass must not change BatchNorm statistics"


def test_mask_builder():
    from omics_gradcam.config import GuidanceConfig
    x, y, pix = small_images()
    model = GapClassifier("efficientnet_b1", 4, pretrained=False).eval()
    loader = [(x[:20], y[:20]), (x[20:], y[20:])]
    protected = np.zeros((3, SIZE, SIZE), bool)
    protected[0, pix[0][0][:30], pix[0][1][:30]] = True
    for method in ("gradcam", "ig"):
        cfg = GuidanceConfig(mask_method=method, topk_ratio=0.3, frequency_threshold=0.1, ig_steps=4,
                             mask_background=False, class_conditional=True)
        mask = MaskBuilder(cfg, 4, protected).build(model, loader, torch.device("cpu"))
        assert mask.shape == (4, 3, SIZE, SIZE) and mask.any()
        assert not (mask & (x.sum(0) == 0)).any(), "gene-only mask must not cover empty pixels"
        assert not (mask & torch.as_tensor(protected)).any(), "protected genes must never be masked"


# ---------------------------------------------------------------------------------------------- 5
def test_calibration():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    logits = rng.normal(size=(200, 4)) + 2 * np.eye(4)[y]
    for method in ("beta", "temperature", "none"):
        cal = build_calibrator(method).fit(logits, y)
        p = cal.predict_proba(logits)
        assert np.allclose(p.sum(1), 1) and (p >= 0).all()
        cal.tune_thresholds(p, y)
        assert cal.predict(p).shape == (200,)


# ---------------------------------------------------------------------------------------------- end to end
def write_synthetic_dataset(folder: Path) -> None:
    rng = np.random.default_rng(0)
    classes = np.array(["Basal", "Her2", "LumA", "LumB"])[np.arange(48) % 4]
    ids = [f"S{i:03d}" for i in range(48)]
    pd.DataFrame({"patient_id": ids, "PAM50": classes}).to_csv(folder / "labels.csv", index=False)
    mats, _ = synthetic_matrices(n_samples=48)
    for (name, fmt), X in zip((("mRNA_filtered.csv", "GENE{i}|{e}"), ("methylation_filtered.csv", "cg{i:08d}"),
                               ("CNA_filtered.csv", "chr1.{i}")), mats):
        index = pd.Index([fmt.format(i=i, e=1000 + i) for i in range(len(X))], name="gene_id")
        pd.DataFrame(X, index=index, columns=ids).to_csv(folder / name)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_synthetic_dataset(tmp)
        for head in ("gap", "gene"):
            out = tmp / f"run_{head}"
            sets = [f"data.data_dir={tmp.as_posix()}", f"output_dir={out.as_posix()}", f"model.head={head}",
                    "image.size=64", "splits.seeds=[7]", "splits.n_folds=2", "train.epochs=3",
                    "train.freeze_epochs=1", "train.batch_size=8", "train.lr=0.0003", "train.swa_lr=0.001",
                    "train.swa_start_epoch=1", "guidance.mask_build_epoch=0", "guidance.mask_refresh_start=1",
                    "guidance.mask_refresh_every=1", "guidance.loss_start_epoch=1", "guidance.lambda_reg=1",
                    "attribution.ig_steps=4", "genes.top_k=10", "device=cpu"]
            cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--steps", "train", "genes"]
            for s in sets:
                cmd += ["--set", s]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
            for f in ("summary.md", "fold_metrics.csv", "seed_7/fold_1/predictions.csv", "seed_7/fold_2/model.pt",
                      "seed_7/fold_1/gene_attributions.npz", "seed_7/fold_1/guidance_mask.npy",
                      "gene_selection_top10/report.txt", "gene_selection_top10/label_LumA_top_10_positive.csv"):
                assert (out / f).exists(), f"{head}: missing {f}"
            # resume: a second run loads the finished folds instead of training again
            proc = subprocess.run(cmd[:2] + ["--steps", "train"] + cmd[5:], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            assert proc.returncode == 0 and "already done -> loaded" in proc.stdout, proc.stdout[-2000:]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print(f"all {len(tests)} tests passed")
