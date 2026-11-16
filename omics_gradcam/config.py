"""Typed configuration: a YAML file plus ``section.key=value`` overrides.

Every option is documented in ``configs/brca.yaml``. Unknown keys and invalid values are rejected
when the configuration is loaded, so a typo never silently falls back to a default.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ----------------------------------------------------------------------------------------------
# 1. Data and splits
# ----------------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    data_dir: str = "data"
    # omics name -> CSV file (rows = features, columns = sample IDs, first column = feature ID).
    # The order is the image channel order.
    omics: dict = field(default_factory=lambda: {
        "mRNA": "mRNA_filtered.csv",
        "Methylation": "methylation_filtered.csv",
        "CNV": "CNA_filtered.csv",
    })
    labels_file: str = "labels.csv"
    label_id_col: str = "patient_id"
    label_col: str = "PAM50"


@dataclass
class SplitConfig:
    seeds: list = field(default_factory=lambda: [12, 3, 44, 63, 16, 1])
    n_folds: int = 5             # outer folds: 80% train+val / 20% test
    val_fraction: float = 0.2    # share of the outer training part: 64% train / 16% validation


# ----------------------------------------------------------------------------------------------
# 2. Omics -> image
# ----------------------------------------------------------------------------------------------
@dataclass
class ImageConfig:
    size: int = 240
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_init: str = "spectral"
    clip_percentile: tuple = (5.0, 95.0)
    # gene values in [1.5/255, 1] so 0 only ever means "empty pixel"
    reserve_zero: bool = True


# ----------------------------------------------------------------------------------------------
# 3. CNN
# ----------------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    arch: str = "efficientnet_b1"
    pretrained: bool = True
    head: str = "gap"                       # gap | gene
    readout_levels: list = field(default_factory=lambda: [2, 3])   # head "gene" only
    dropout: float = 0.6


@dataclass
class TrainConfig:
    epochs: int = 100                       # total: AdamW phase + SWA phase
    swa_start_epoch: int = 75               # 0-based: epochs 1-75 AdamW, 76-100 SWA
    freeze_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    # phase 1: AdamW, learning rate lr -> min_lr by CosineAnnealingLR over the first swa_start_epoch epochs
    lr: float = 1e-3
    weight_decay: float = 0.01
    min_lr: float = 1e-6
    # phase 2: SWA (torch.optim.swa_utils) - a new SGD optimizer started at the large learning rate swa_lr,
    # held constant by SWALR; the weights after every SWA epoch are averaged (AveragedModel)
    swa_lr: float = 0.01
    swa_momentum: float = 0.9
    swa_nesterov: bool = True
    swa_weight_decay: float = 1e-3
    focal_gamma: float = 2.0
    focal_alpha: str = "inverse_freq"       # inverse_freq | none
    weighted_sampling: bool = False
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


# ----------------------------------------------------------------------------------------------
# 4. Guidance
# ----------------------------------------------------------------------------------------------
@dataclass
class GuidanceConfig:
    enabled: bool = True
    lambda_reg: float = 1000.0
    # schedule (0-based epochs)
    mask_build_epoch: int = 17
    mask_refresh_start: int = 20
    mask_refresh_every: int = 3
    loss_start_epoch: int = 18
    # mask
    mask_source: str = "val"                # val | train
    mask_method: str = "gradcam"            # gradcam | ig
    topk_ratio: float = 0.22                # gradcam 0.22 / ig 0.175 cover ~50% of the gene pixels
    frequency_threshold: float = 0.2
    ig_steps: int = 8                       # mask_method ig only
    mask_background: bool = False
    class_conditional: bool = True
    protect_top_fraction: float = 0.5
    # penalised input gradient
    penalty_target: str = "pair"            # true_class | all_classes | pair
    penalty_pair: list = field(default_factory=lambda: ["LumA", "LumB"])


# ----------------------------------------------------------------------------------------------
# 5. Calibration
# ----------------------------------------------------------------------------------------------
@dataclass
class CalibrationConfig:
    method: str = "beta"                    # beta | temperature | none
    tune_thresholds: bool = False


# ----------------------------------------------------------------------------------------------
# 6. Downstream
# ----------------------------------------------------------------------------------------------
@dataclass
class AttributionConfig:
    enabled: bool = True
    ig_steps: int = 64
    max_batch: int = 72                     # images per IG forward pass (also used by the IG mask)
    save_checkpoints: bool = True


@dataclass
class GenesConfig:
    top_k: int = 400
    label_by: str = "predicted"             # predicted | predicted_calibrated | true
    enrichment_omics: list = field(default_factory=lambda: ["mRNA"])
    signs: list = field(default_factory=lambda: ["positive"])
    go_ontology: str = "BP"                 # BP | MF | CC
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
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    genes: GenesConfig = field(default_factory=GenesConfig)
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

CHOICES = {
    ("model", "head"): {"gap", "gene"},
    ("train", "focal_alpha"): {"inverse_freq", "none"},
    ("guidance", "mask_source"): {"train", "val"},
    ("guidance", "mask_method"): {"gradcam", "ig"},
    ("guidance", "penalty_target"): {"true_class", "all_classes", "pair"},
    ("calibration", "method"): {"beta", "temperature", "none"},
    ("genes", "label_by"): {"predicted", "predicted_calibrated", "true"},
    ("genes", "go_ontology"): {"BP", "MF", "CC"},
}


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
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
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
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
    if not 1 <= t.swa_start_epoch < t.epochs:
        raise ValueError(f"train.swa_start_epoch must be in [1, {t.epochs}) (at least one AdamW and one SWA epoch), "
                         f"got {t.swa_start_epoch!r}")
    for key in ("lr", "swa_lr"):
        if not getattr(t, key) > 0:
            raise ValueError(f"train.{key} must be > 0, got {getattr(t, key)!r}")
    if not 0 <= t.min_lr < t.lr:
        raise ValueError(f"train.min_lr must be in [0, train.lr), got {t.min_lr!r}")
    lv = m.readout_levels
    if not (isinstance(lv, (list, tuple)) and lv and len(set(lv)) == len(lv)
            and all(isinstance(v, int) and 0 <= v <= 4 for v in lv)):
        raise ValueError(f"model.readout_levels must be distinct integers in 0..4, got {lv!r}")
    if cfg.splits.n_folds < 2 or not 0 < cfg.splits.val_fraction < 1:
        raise ValueError("splits.n_folds must be >= 2 and splits.val_fraction in (0, 1)")


def config_from_dict(d: dict | None) -> Config:
    d = dict(d or {})
    kwargs = {}
    for key, value in d.items():
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
            if not isinstance(node.get(p), dict):      # missing, or an empty YAML section (None)
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
