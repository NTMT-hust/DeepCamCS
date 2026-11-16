# Attribution-guided multi-omics image classification

Each sample's omics profiles (mRNA, methylation, CNV) are drawn as a 3-channel image, and an ImageNet-pretrained
EfficientNet-B1 is trained on the images. During training, a penalty on the input gradient keeps the CNN from
relying on pixels it uses for the "wrong reasons". Integrated-Gradients attributions then rank the genes per subtype.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1. Data and splits | `data.py`, `splits.py` | Join the omics tables and labels by sample ID. Repeated stratified CV: per seed, 5 outer folds (20% test); the rest is split 64% train / 16% validation. |
| 2. Omics → image | `imaging.py` | Per fold and omics, fitted on the training split only: UMAP places the features in 2-D, the map is rotated to its minimum bounding rectangle and scaled to 240×240, and snowfall gives every feature its own pixel. Each sample is clipped to its own 5th–95th percentile, then min-max, log1p, min-max, then uint8. |
| 3. CNN | `models.py`, `losses.py`, `augmentation.py`, `training.py` | EfficientNet-B1 (ImageNet), focal loss, class-aware augmentation (noise, same-class CutMix / MixUp / SMOTE). Training: AdamW with a cosine learning rate for epochs 1–75, then SWA (`torch.optim.swa_utils`: SGD at a large constant learning rate, weights averaged after every epoch) for epochs 76–100; the averaged model is the fold model. Two heads (`model.head`): `gap` (global pooling, original) or `gene` (one trunk per omics, readout at every gene's pixel). |
| 4. Guidance | `guidance.py`, `attribution.py` | The mask holds the gene pixels the model uses for the wrong reasons (Grad-CAM or sparse IG, top-k per sample, frequency across samples). It is built from epoch 18 and refreshed every 3 epochs. The penalty is λ·‖∇ₓ S · mask‖², computed exactly with a forward-mode JVP. |
| 5. Calibration | `calibration.py` | Beta (or temperature) calibration on the validation split. Per-class decision thresholds are optional. |
| 6. Downstream | `attribution.py`, `genes.py`, `scripts/go_kegg_enrichment.R` | IG of every test sample, read at each gene's pixel. Then top-K genes per subtype × omics × sign, overlap with PAM50, stability across seeds, and GO / KEGG enrichment in R. |

`experiment.py` runs stages 2–6 for every seed × fold, resumes interrupted runs, and writes the summary.
`run_pipeline.py` is the only entry point.

## Setup

```bash
pip install -r requirements.txt          # PyTorch: install the CUDA build for your GPU first
```

Put the data in `data/`; [data/README.md](data/README.md) describes the files. The GO / KEGG step needs R with
`clusterProfiler`, `org.Hs.eg.db`, `enrichplot` and `ggplot2`; without R it is skipped with a note.

## Usage

```bash
python run_pipeline.py                                              # all steps, configs/brca.yaml
python run_pipeline.py --out runs/brca_gene --set model.head=gene    # any option: --set section.key=value
python run_pipeline.py --out runs/no_guidance --set guidance.enabled=false
python run_pipeline.py --seeds 12 3 --out runs/dev                  # fewer seeds
python run_pipeline.py --steps genes go_kegg --out runs/brca        # downstream of an existing run
python tests/test_pipeline.py                                       # checks + small end-to-end run (CPU)
```

Steps: `train` (stages 1–6a), `genes` (gene selection), `go_kegg` (enrichment). Rerunning into the same
`--out` loads finished folds instead of retraining them. If the settings differ from those the folds were
trained with, the run stops instead of mixing them.

## Configuration

Every option, with its default and a comment, is in [configs/brca.yaml](configs/brca.yaml). Unknown keys and
invalid values are rejected on load. The image, guidance and calibration defaults are the best guided setup
measured so far; the training schedule (AdamW + cosine, then SGD SWA, 100 epochs) is new and not yet validated
against the reference runs:

- **Images:** `reserve_zero` on.
- **Guidance mask:** Grad-CAM top-22% of the gene pixels, gene pixels only, one mask per class, top-50% genes by ANOVA protected.
- **Penalty:** the LumA-vs-LumB pair objective, λ = 1000.
- **Training:** AdamW (lr 1e-3, weight decay 0.01) with CosineAnnealingLR for 75 epochs, then 25 SWA epochs with
  SGD at constant lr 0.01 (SWALR); fold model = the averaged weights with BatchNorm re-estimated.
- **Calibration:** Beta, without thresholds.

## Outputs (`runs/<name>/`)

| File | Content |
|---|---|
| `summary.md`, `summary.json` | mean ± std over fold models and seeds, multi-seed ensemble, guidance mask check |
| `fold_metrics.csv`, `seed_metrics.csv`, `oof_predictions_all_seeds.csv`, `mask_check.csv` | per fold / per seed / per sample |
| `seed_<s>/fold_<f>/` | `predictions.csv`, `history.csv`, `fold_result.json`, `model.pt` (SWA weights), `gene_attributions.npz`, `layout.npz`, `gene_coordinates_<omics>.csv`, `guidance_mask.npy` |
| `gene_selection_top<K>/` | `label_<subtype>_top_<K>_<sign>.csv`, `report.txt` (PAM50, stability), `background_features.csv`, `go_kegg/` |
| `run.log`, `config_resolved.yaml` | full log, exact settings |

## Where to change what

| To change | Edit |
|---|---|
| input files, label column | `data:` in the config |
| image layout or value normalisation | `imaging.py` (`fit_layout`, `normalise_per_sample`) |
| architecture, new head | `models.py` (`build_model`), plus a choice in `config.py` `CHOICES` |
| training loop, optimisers, schedules | `training.py` (`train_fold`, `adamw_phase`, `swa_phase`) |
| mask rule | `guidance.py` (`MaskBuilder`) |
| penalised objective | `guidance.py` (`_objective`, `penalty_objective`) |
| gene selection / report | `genes.py` |
| summary tables | `experiment.py` (`aggregate`, `summary_markdown`) |

## Notes

- **GPU memory:** a guided step with the `gap` head needs ~12 GB at batch 16, so run experiments one after
  another on a 16 GB card. `--gpu-memory-fraction` (default 0.9) caps PyTorch. Without the cap, Windows spills
  into system RAM and training becomes 50–100× slower.
- **Gene head with guidance:** the `gene` head runs three trunks per image, and the guidance penalty adds a
  second forward pass. A guided step needs ~2 GB per image (`gap`: 0.76 GB), so batch 16 does not fit on a
  16 GB card; the largest that fits is `train.batch_size=6` (12 GB, ~4× slower per image). Without guidance,
  batch 16 needs 5.7 GB.
- **Validation:** the validation split (16%) is used for monitoring, the guidance mask (`mask_source: val`) and
  calibration. Nothing is selected on it.
- **Run time:** 100 epochs take about twice as long as the 50 of the reference runs.
- **Relation to `new_method_gradcam_refactored`:** this code is a cleaned-up version of it. Before the training
  schedule changed (that version: SGD with warm-up + cosine for 50 epochs, SWA of epochs 31-50 on the same
  trajectory), the two gave bit-identical results with the same settings (checked on synthetic data for both
  heads and both mask methods). Images, splits, models, guidance, calibration and gene selection are unchanged;
  results now differ because of the AdamW / SWA schedule. For the `gene` head, multi-threaded CPU / GPU runs
  differ from run to run at float precision.
