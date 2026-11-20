"""Checks for every phase, plus an end-to-end run of phases 1-6 on a small synthetic dataset (CPU, a few
minutes; it uses the ImageNet weights, downloaded by timm on first use).

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
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gene_guided.config import GuidanceConfig, ImageConfig, TrainConfig, config_from_dict, load_config  # noqa: E402
from gene_guided.data import make_cv_splits  # noqa: E402
from gene_guided.guidance import MaskBuilder, _dual_pass, _objective, guided_loss  # noqa: E402
from gene_guided.imaging import OmicsImageTransformer  # noqa: E402
from gene_guided.losses import FocalLoss  # noqa: E402
from gene_guided.model import GeneReadoutClassifier  # noqa: E402
from gene_guided.phase6_pathway_analysis import bh_adjust, over_representation  # noqa: E402
from gene_guided.training import sgd_schedule  # noqa: E402

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


def gene_model(pix, dropout=0.5):
    return GeneReadoutClassifier("efficientnet_b1", 4, pix, SIZE, pretrained=False, dropout=dropout)


# ---------------------------------------------------------------------------------------------- config, phase 1
def test_config():
    cfg = load_config(ROOT / "config.yaml", ["splits.seeds=[1, 2]", "train.swa_start_epoch=40"])
    assert cfg.splits.seeds == [1, 2] and cfg.train.swa_start_epoch == 40 and cfg.guidance.penalty_samples == 4
    for bad in ({"model": {"head": "gap"}}, {"guidance": {"topk_ratio": 0}}, {"train": {"lr_typo": 1}},
                {"guidance": {"enabled": "maybe"}}, {"train": {"swa_start_epoch": 100}},
                {"guidance": {"penalty_samples": 0}}, {"train": {"warmup_start_factor": 0}},
                {"genes": {"label_by": "calibrated"}}, {"go_kegg": {"signs": ["both"]}}, {"other": {}}):
        try:
            config_from_dict(bad)
        except (ValueError, KeyError):
            continue
        raise AssertionError(f"invalid config accepted: {bad}")


def test_splits():
    y = np.repeat(np.arange(4), [20, 10, 25, 15])
    splits = make_cv_splits(y, seed=3, n_folds=5, val_fraction=0.2)
    assert sorted(np.concatenate([s.test for s in splits])) == list(range(len(y)))
    for s in splits:
        assert not (set(s.train) & set(s.val)) and not (set(s.train) & set(s.test))


def test_imaging():
    mats, _ = synthetic_matrices()
    tfm = OmicsImageTransformer(["a", "b", "c"], ImageConfig(size=SIZE), random_state=0).fit(mats)
    images = tfm.transform(mats)
    assert images.shape == (40, 3, SIZE, SIZE) and images.dtype == np.uint8
    for c, (rows, cols) in enumerate(tfm.pixel_index()):
        assert len(set(zip(rows, cols))) == len(rows), "every feature needs its own pixel"
        gene = np.zeros((SIZE, SIZE), bool)
        gene[rows, cols] = True
        assert (images[:, c][:, gene] >= 1).all(), "gene pixels must be >= 1 (0 = empty)"
        assert (images[:, c][:, ~gene] == 0).all(), "empty pixels must be 0"
    alone = tfm.transform([m[:, :1] for m in mats])                   # per-sample normalisation
    assert np.array_equal(alone[0], images[0])


# ---------------------------------------------------------------------------------------------- phase 2
def test_gene_readout():
    x, _, pix = small_images()
    model = gene_model(pix, dropout=0.0).eval()
    with torch.no_grad():
        z = model.gene_evidence(x[:4])
        assert torch.allclose(model(x[:4]), z.mean(-1) + model.bias, atol=1e-5), "init = average gene evidence"
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
        for p in model.blocks[2].mods[1].parameters():                 # trunk 1 only affects omics 1
            p.add_(torch.randn_like(p) * 0.1)
        z2 = model.gene_evidence(x[:4])
    n0, n1 = len(pix[0][0]), len(pix[1][0])
    assert torch.allclose(z2[..., :n0], z[..., :n0]) and not torch.allclose(z2[..., n0:n0 + n1], z[..., n0:n0 + n1])


def test_sgd_schedule():
    t = TrainConfig(epochs=20, warmup_epochs=5, warmup_start_factor=0.05, lr=0.01, min_lr=0.0)
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = sgd_schedule(model, t)
    assert isinstance(optimizer, torch.optim.SGD) and optimizer.param_groups[0]["nesterov"]
    lrs = []
    for _ in range(t.epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        model(torch.randn(4, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    assert abs(lrs[0] - 0.0005) < 1e-12 and all(a < b for a, b in zip(lrs[:5], lrs[1:6])), lrs     # warm-up
    assert abs(lrs[5] - 0.01) < 1e-12 and all(a > b for a, b in zip(lrs[5:], lrs[6:])), lrs       # cosine
    assert lrs[-1] < 2e-4, lrs


def test_penalty_equals_double_backward():
    """Focal loss on the whole batch + the penalty of a random subset (training mode, BatchNorm running
    statistics untouched): the JVP gradient must equal the double-backward gradient of that loss."""
    x, y, pix = small_images()
    x, y = x[:6], y[:6]
    criterion = FocalLoss(alpha=torch.tensor([0.4, 0.3, 0.2, 0.1]))
    mask = torch.rand(6, 3, SIZE, SIZE) > 0.9
    target = ("pair", 2, 3)
    model = gene_model(pix).train()
    m1, m2 = copy.deepcopy(model), copy.deepcopy(model)
    idx = torch.randperm(6, generator=torch.Generator().manual_seed(5))[:3]
    torch.manual_seed(1)
    ce = criterion(m1(x), y)                                   # training mode, whole batch
    xs = x[idx].clone().requires_grad_(True)                   # penalty: training mode, the subset only,
    with _dual_pass(m1):                                       #   BatchNorm running statistics untouched
        (g,) = torch.autograd.grad(_objective(torch.log_softmax(m1(xs), 1), y[idx], target), xs, create_graph=True)
    mk = mask.float()[idx]
    reg = ((g.pow(2) * mk).flatten(1).sum(1) / mk.flatten(1).sum(1).clamp_min(1)).mean()
    (ce + 100.0 * reg).backward()
    torch.manual_seed(1)
    _, loss, value = guided_loss(m2, x, y, criterion, mask, 100.0, target, 3, torch.Generator().manual_seed(5))
    loss.backward()
    assert m2.training, "the model must stay in training mode"
    r = float(reg.detach())
    assert abs(float(value) - r) <= 1e-4 * max(abs(r), 1e-12), (float(value), r)
    g1 = torch.cat([p.grad.flatten() for p in m1.parameters() if p.grad is not None])
    g2 = torch.cat([p.grad.flatten() for p in m2.parameters() if p.grad is not None])
    rel = float((g1 - g2).norm() / g1.norm())
    assert rel < 1e-3, f"penalty gradient differs from double backward (rel {rel:.1e})"
    for (_, b1), (_, b2) in zip(m1.named_buffers(), m2.named_buffers()):
        assert torch.equal(b1, b2), "the penalty passes must not change BatchNorm statistics"


def test_mask_builder():
    x, y, pix = small_images()
    model = gene_model(pix).eval()
    loader = [(x[:20], y[:20]), (x[20:], y[20:])]
    protected = np.zeros((3, SIZE, SIZE), bool)
    protected[0, pix[0][0][:30], pix[0][1][:30]] = True
    cfg = GuidanceConfig(topk_ratio=0.3, frequency_threshold=0.1)
    mask = MaskBuilder(cfg, 4, protected).build(model, loader, torch.device("cpu"))
    assert mask.shape == (4, 3, SIZE, SIZE) and mask.any()
    chunked = MaskBuilder(cfg, 4, protected, max_batch=7).build(model, loader, torch.device("cpu"))
    assert (chunked != mask).float().mean() < 1e-3          # (float ties flipped by batch-size-dependent kernels)
    assert not (mask & (x.sum(0) == 0)).any(), "the mask must not cover empty pixels"
    assert not (mask & torch.as_tensor(protected)).any(), "protected genes must never be masked"


# ---------------------------------------------------------------------------------------------- phase 6
def test_over_representation():
    universe = {f"G{i}" for i in range(100)}
    sets = {"A": [f"G{i}" for i in range(20)] + ["NOT_MEASURED"], "B": [f"G{i}" for i in range(50, 60)], "tiny": ["G1"]}
    genes = {f"G{i}" for i in range(10)} | {"G55", "G70"}
    res = over_representation(genes, universe, sets, min_size=5).set_index("Term")
    assert set(res.index) == {"A", "B"} and res.loc["A", "set_size_in_universe"] == 20
    for term, k, K in (("A", 10, 20), ("B", 1, 10)):        # one-sided Fisher = hypergeometric upper tail
        n, N = len(genes), len(universe)
        table = [[k, n - k], [K - k, N - n - K + k]]
        assert abs(res.loc[term, "p_value"] - fisher_exact(table, alternative="greater")[1]) < 1e-12
    p = np.array([0.01, 0.04, 0.03, 0.5])                   # ranks 1, 3, 2, 4: p * 4 / rank, then cumulative min
    assert np.allclose(bh_adjust(p), [0.04, 0.16 / 3, 0.16 / 3, 0.5])


# ---------------------------------------------------------------------------------------------- end to end
def write_synthetic_dataset(folder: Path) -> None:
    classes = np.array(["Basal", "Her2", "LumA", "LumB"])[np.arange(48) % 4]
    ids = [f"S{i:03d}" for i in range(48)]
    pd.DataFrame({"patient_id": ids, "PAM50": classes}).to_csv(folder / "labels.csv", index=False)
    mats, _ = synthetic_matrices(n_samples=48)
    for (name, fmt), X in zip((("mRNA_filtered.csv", "GENE{i}|{e}"), ("methylation_filtered.csv", "cg{i:08d}"),
                               ("CNA_filtered.csv", "chr1.{i}")), mats):
        index = pd.Index([fmt.format(i=i, e=1000 + i) for i in range(len(X))], name="gene_id")
        pd.DataFrame(X, index=index, columns=ids).to_csv(folder / name)
    rng = np.random.default_rng(1)
    (folder / "sets.gmt").write_text("".join(f"SET{j}\t\t" + "\t".join(f"GENE{i}" for i in rng.choice(150, 15, replace=False))
                                             + "\n" for j in range(12)), encoding="utf-8")


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_synthetic_dataset(tmp)
        out = tmp / "run"
        sets = [f"data.data_dir={tmp.as_posix()}", f"output_dir={out.as_posix()}", "image.size=64", "splits.seeds=[7, 8]",
                "splits.n_folds=2", "train.epochs=3", "train.freeze_epochs=1", "train.batch_size=8", "train.lr=0.0003",
                "train.warmup_epochs=1", "train.swa_start_epoch=1", "guidance.mask_build_epoch=0",
                "guidance.mask_refresh_start=1", "guidance.mask_refresh_every=1", "guidance.loss_start_epoch=1",
                "guidance.lambda_reg=1", "guidance.penalty_samples=2", "attribution.ig_steps=4", "genes.top_k=10",
                f"pathway.gmt_file={(tmp / 'sets.gmt').as_posix()}", "pathway.min_size=3", "device=cpu"]
        cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--phases", "1", "2", "3", "4", "5", "6"]
        for s in sets:
            cmd += ["--set", s]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        for f in ("1_images/dataset.npz", "1_images/seed_7/fold_1/images.npz", "1_images/seed_8/fold_2/done.json",
                  "1_images/seed_7/fold_1/gene_coordinates_mRNA.csv", "2_training/summary.md",
                  "2_training/fold_metrics.csv", "2_training/seed_7/fold_1/predictions.csv",
                  "2_training/seed_7/fold_2/model.pt", "2_training/seed_8/fold_1/gene_attributions.npz",
                  "2_training/seed_7/fold_1/guidance_mask.npy", "3_selected_genes_top10/label_LumA_top_10_positive.csv",
                  "3_selected_genes_top10/selection_stability_across_seeds.csv", "4_merged_genes/unique_genes_omics.csv",
                  "4_merged_genes/pam50_overlap.csv", "5_filtered_omics/mRNA_filtered_selected.csv",
                  "5_filtered_omics/labels.csv", "6_pathway/summary.csv", "6_pathway/sets_all_selected.csv"):
            assert (out / f).exists(), f"missing {f}"
        unique = pd.read_csv(out / "4_merged_genes/unique_genes_omics.csv")
        kept = pd.read_csv(out / "5_filtered_omics/mRNA_filtered_selected.csv", index_col=0)
        assert set(kept.index) == set(unique.loc[unique.omics_type == "mRNA", "unique_genes"]) and kept.shape[1] == 48
        # resume: a second run loads the finished folds and images instead of making them again
        proc = subprocess.run(cmd[:2] + ["--phases", "1", "2"] + cmd[9:], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        assert proc.returncode == 0 and "already done -> loaded" in proc.stdout and "[images] seed 7 fold 1: already done" \
            in proc.stdout, proc.stdout[-2000:]
        # a run directory refuses folds made with other settings
        proc = subprocess.run(cmd[:2] + ["--phases", "2"] + cmd[9:] + ["--set", "guidance.lambda_reg=5"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode != 0 and "different settings" in proc.stdout + proc.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print(f"all {len(tests)} tests passed")
