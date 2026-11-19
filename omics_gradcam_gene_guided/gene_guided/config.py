"""Configuration: ``config.yaml`` plus ``section.key=value`` overrides.

Only the settings of this method are options. The choices it fixes are part of the code, not the config:
per-omics gene head, one SGD run with SWA, 0 reserved for empty pixels, Grad-CAM mask built on the
validation split from gene pixels only with one mask per class, LumA/LumB pair penalty on a random subset of
each batch, raw softmax probabilities (no calibration). Unknown keys and invalid values are rejected when the
configuration is loaded, so a typo never silently falls back to a default.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---- Phase 1: data, splits, images ------------------------------------------------------------
@dataclass
class DataConfig:
    data_dir: str = "data"
    # omics name -> CSV (rows = features, columns = sample IDs, first column = feature ID); order = channel order
    omics: dict = field(default_factory=lambda: {"mRNA": "mRNA_filtered.csv", "Methylation": "methylation_filtered.csv",
                                                 "CNV": "CNA_filtered.csv"})
    labels_file: str = "labels.csv"
    label_id_col: str = "patient_id"
    label_col: str = "PAM50"


@dataclass
class SplitConfig:
    seeds: list = field(default_factory=lambda: [12, 3, 44, 63, 16, 1])
    n_folds: int = 5             # outer folds: 80% train+val / 20% test
    val_fraction: float = 0.2    # share of the outer training part: 64% train / 16% validation


@dataclass
class ImageConfig:
    size: int = 240
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_init: str = "spectral"
    clip_percentile: tuple = (5.0, 95.0)


# ---- Phase 2: CNN, training, guidance, attribution ----------------------------------------------
@dataclass
class ModelConfig:
    arch: str = "efficientnet_b1"
    pretrained: bool = True
    readout_levels: list = field(default_factory=lambda: [2, 3])   # timm feature levels: 2 = 30x30, 3 = 15x15
    dropout: float = 0.6


@dataclass
class TrainConfig:
    epochs: int = 50
    swa_start_epoch: int = 30               # 0-based: the weights after epochs 31-50 are averaged
    freeze_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 0.01                        # SGD peak learning rate
    weight_decay: float = 1e-3
    min_lr: float = 0.0                     # end of the cosine
    momentum: float = 0.9
    nesterov: bool = True
    warmup_epochs: int = 10                 # linear warm-up from lr * warmup_start_factor (0 = none)
    warmup_start_factor: float = 0.05
    focal_gamma: float = 2.0
    deterministic: bool = True


@dataclass
class AugmentConfig:
    enabled: bool = True
    noise_std: float = 0.03
    p_noise: float = 0.2
    p_cutmix: float = 0.1
    p_mixup: float = 0.1
    p_smote: float = 0.1
    cutmix_ratio: tuple = (0.2, 0.5)
    mixup_alpha: float = 0.2
    smote_alpha_range: tuple = (0.1, 0.4)


@dataclass
class GuidanceConfig:
    enabled: bool = True                    # false = the same model without guidance (control)
    lambda_reg: float = 1000.0
    mask_build_epoch: int = 17              # 0-based epochs: first mask at epoch 18,
    mask_refresh_start: int = 20            #   rebuilt every 3 epochs from epoch 21,
    mask_refresh_every: int = 3
    loss_start_epoch: int = 18              #   penalty from epoch 19
    topk_ratio: float = 0.22                # per sample, top share of gene pixels by wrong-reason score
    frequency_threshold: float = 0.2        # keep pixels selected in > 20% of a class's samples
    protect_top_fraction: float = 0.5       # top 50% of each omics' genes by ANOVA F (training split) never masked
    penalty_pair: list = field(default_factory=lambda: ["LumA", "LumB"])
    penalty_samples: int = 4                # samples per batch that get the penalty (4 fit in 16 GB at batch 16)


@dataclass
class AttributionConfig:
    ig_steps: int = 64
    max_batch: int = 72                     # images per IG / Grad-CAM pass, divided by the number of omics
    save_checkpoints: bool = True


# ---- Phases 3-7: genes and enrichment ------------------------------------------------------------
@dataclass
class GenesConfig:
    top_k: int = 400
    label_by: str = "predicted"             # predicted | true: which label groups the attributions


@dataclass
class PathwayConfig:
    gene_set: str = "MSigDB_Hallmark_2020"  # Enrichr library, downloaded once into cache/ (no gene list is sent)
    gmt_file: str | None = None             # or a local .gmt file (used instead of the download)
    omics: list = field(default_factory=lambda: ["mRNA"])   # gene-level omics only
    padj: float = 0.05
    min_size: int = 5                       # smallest gene set tested (genes in the universe)


@dataclass
class GoKeggConfig:
    omics: list = field(default_factory=lambda: ["mRNA"])
    signs: list = field(default_factory=lambda: ["positive"])
    ontology: str = "BP"                    # BP | MF | CC
    padj: float = 0.05
    kegg: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    genes: GenesConfig = field(default_factory=GenesConfig)
    pathway: PathwayConfig = field(default_factory=PathwayConfig)
    go_kegg: GoKeggConfig = field(default_factory=GoKeggConfig)
    output_dir: str = "runs/default"
    device: str = "auto"

    def to_dict(self) -> dict:
        def plain(v):
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            return v
        return plain(dataclasses.asdict(self))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


SECTIONS = {f.name: f.default_factory for f in dataclasses.fields(Config) if f.default_factory is not dataclasses.MISSING}
CHOICES = {("genes", "label_by"): {"predicted", "true"}, ("go_kegg", "ontology"): {"BP", "MF", "CC"}}


def _to_bool(where: str, v) -> bool:
    """YAML reads typos such as ``none`` / ``off`` in quotes as strings, which Python treats as true."""
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0", "none", "null"}:
        return False
    raise ValueError(f"{where} must be true/false, got {v!r}")


def _build_section(name: str, values: dict | None):
    cls = SECTIONS[name]
    values = values or {}
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - fields
    if unknown:
        raise KeyError(f"unknown keys in section '{name}': {sorted(unknown)}")
    kwargs = {}
    for key, v in values.items():
        default = cls().__dict__[key]
        if isinstance(default, bool):
            v = _to_bool(f"{name}.{key}", v)
        elif isinstance(default, tuple) and isinstance(v, list):
            v = tuple(v)
        kwargs[key] = v
    return cls(**kwargs)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate(cfg: Config) -> None:
    for (section, key), allowed in CHOICES.items():
        v = getattr(getattr(cfg, section), key)
        if v not in allowed:
            raise ValueError(f"{section}.{key} must be one of {' | '.join(sorted(allowed))}, got {v!r}")
    g, t, m = cfg.guidance, cfg.train, cfg.model
    if not 0 < g.topk_ratio <= 1:
        raise ValueError(f"guidance.topk_ratio must be in (0, 1], got {g.topk_ratio!r}")
    for key in ("frequency_threshold", "protect_top_fraction"):
        if not 0 <= getattr(g, key) < 1:
            raise ValueError(f"guidance.{key} must be in [0, 1), got {getattr(g, key)!r}")
    pair = g.penalty_pair
    if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] != pair[1]):
        raise ValueError(f"guidance.penalty_pair must be two different class names, got {pair!r}")
    if not (_is_int(g.penalty_samples) and g.penalty_samples >= 1):
        raise ValueError(f"guidance.penalty_samples must be an integer >= 1, got {g.penalty_samples!r}")
    if not g.lambda_reg >= 0:
        raise ValueError(f"guidance.lambda_reg must be >= 0, got {g.lambda_reg!r}")
    if not 1 <= t.swa_start_epoch < t.epochs:
        raise ValueError(f"train.swa_start_epoch must be in [1, {t.epochs}), got {t.swa_start_epoch!r}")
    if not t.lr > 0 or not 0 <= t.min_lr < t.lr:
        raise ValueError(f"train.lr must be > 0 and train.min_lr in [0, lr), got {t.lr!r} / {t.min_lr!r}")
    if not (_is_int(t.warmup_epochs) and t.warmup_epochs >= 0):
        raise ValueError(f"train.warmup_epochs must be an integer >= 0, got {t.warmup_epochs!r}")
    if not 0 < t.warmup_start_factor <= 1:
        raise ValueError(f"train.warmup_start_factor must be in (0, 1], got {t.warmup_start_factor!r}")
    lv = m.readout_levels
    if not (isinstance(lv, (list, tuple)) and lv and len(set(lv)) == len(lv) and all(_is_int(v) and 0 <= v <= 4 for v in lv)):
        raise ValueError(f"model.readout_levels must be distinct integers in 0..4, got {lv!r}")
    if cfg.splits.n_folds < 2 or not 0 < cfg.splits.val_fraction < 1:
        raise ValueError("splits.n_folds must be >= 2 and splits.val_fraction in (0, 1)")
    if not (_is_int(cfg.genes.top_k) and cfg.genes.top_k >= 1):
        raise ValueError(f"genes.top_k must be an integer >= 1, got {cfg.genes.top_k!r}")
    for key in ("padj",):
        for sec in (cfg.pathway, cfg.go_kegg):
            if not 0 < getattr(sec, key) <= 1:
                raise ValueError(f"padj must be in (0, 1], got {getattr(sec, key)!r}")
    bad = set(cfg.go_kegg.signs) - {"positive", "negative"}
    if bad or not cfg.go_kegg.signs:
        raise ValueError(f"go_kegg.signs must be positive and/or negative, got {cfg.go_kegg.signs!r}")


def config_from_dict(d: dict | None) -> Config:
    kwargs = {}
    for key, value in dict(d or {}).items():
        if key in SECTIONS:
            kwargs[key] = _build_section(key, value)
        elif key in ("output_dir", "device"):
            kwargs[key] = value
        else:
            raise KeyError(f"unknown top-level config key: {key}")
    cfg = Config(**kwargs)
    validate(cfg)
    return cfg


def apply_overrides(d: dict, overrides: list[str] | None) -> dict:
    """``section.key=value`` overrides; the value is parsed as YAML (``[1, 2]``, ``true``, ``0.5``)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value: Any = yaml.safe_load(raw)
        *path, last = key.strip().split(".")
        node = d
        for p in path:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[last] = value
    return d


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    d: dict = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    return config_from_dict(apply_overrides(d, overrides))
