# Grad-CAM-guided per-omics gene-readout CNN, in the seven phases of DeepCamCS

Multi-omics subtype classification (TCGA BRCA, PAM50: Basal, Her2, LumA, LumB) and biomarker discovery.

1. Each sample's mRNA, methylation and CNV profiles are drawn as a 3-channel image.
2. An EfficientNet-B1 with one trunk per omics reads the class evidence at every gene's pixel.
3. From epoch 19, a Grad-CAM-guided input-gradient penalty discourages the network from using genes for the
   wrong reasons.
4. Integrated Gradients then rank the genes per subtype, and the selected genes go through PAM50, Hallmark and
   GO / KEGG analysis.

This folder holds only the method of `omics_gradcam_pipeline/runs/brca_gene_guided`, organised in the
phases of `DeepCamCS-main`. On the same data and settings it gives the same results as that run (see
[Verification](#verification)).

## Phases

| Phase | Module (`gene_guided/`) | DeepCamCS | What it does | Output |
|---|---|---|---|---|
| 1 | `phase1_images.py` | `produceIMG.py` | **Omics → images**, per seed × fold.<br>Layout: fitted on the fold's training split only (UMAP of the features, rotation to the minimum bounding rectangle, snowfall); one pixel per gene.<br>Pixels: per-sample p5–p95 clip, min-max / log1p / min-max, 0 reserved for empty pixels, uint8.<br>Also the ANOVA ranks on the training split. | `1_images/` |
| 2 | `phase2_train.py` | `train.py` | **Train and evaluate the guided CNN**, then Integrated Gradients (64 steps) of every test sample at every gene's pixel, then the summary over all folds and seeds. | `2_training/` |
| 3 | `phase3_choose_genes.py` | `ChooseGenes.py` | **Top-K genes** per subtype × omics × sign, by the median attribution over the samples predicted as that subtype. Also how stable the lists are across seeds. | `3_selected_genes_top<K>/` |
| 4 | `phase4_merge_genes.py` | `merge_genes_omics.py` | **Union of the selected (gene, omics) pairs**, and their overlap with PAM50. | `4_merged_genes/` |
| 5 | `phase5_filter_omics.py` | `filter_omics_by_unique_genes.py` | **Original omics matrices restricted to the selected genes**, plus the labels. | `5_filtered_omics/` |
| 6 | `phase6_pathway_analysis.py` | `pathway_analysis.py` | **MSigDB Hallmark over-representation** of the pooled list and of each subtype's positive list. | `6_pathway/` |
| 7 | `phase7_go_kegg.py` + `scripts/go_kegg_enrichment.R` | `go_kegg_enrichment.R` | **GO BP and KEGG per subtype** (clusterProfiler) against the measured genes. | `7_go_kegg/` |

Each phase reads the files of the previous phases from the run directory, so any phase can be rerun on its
own. Phases 1 and 2 skip finished folds, so an interrupted run resumes where it stopped.

## The method (fixed in the code)

- **Model:** EfficientNet-B1 (timm, ImageNet weights), one trunk per omics: stem and stages 1–5, input 1 channel.
  - A 1×1 convolution turns the stage-3 (30×30) and stage-5 (15×15) feature maps into class-evidence maps.
  - These maps are read at every gene's pixel by bilinear interpolation, then pass dropout 0.6 and a gene-weighted
    sum. The classifier never sees input pixels.
- **Training:**
  - Focal loss (γ 2) with inverse-frequency class weights; class-aware augmentation (noise, CutMix / MixUp / SMOTE
    within a class).
  - SGD (Nesterov, lr 0.01, weight decay 1e-3) with a 10-epoch warm-up then cosine over 50 epochs, batch 16,
    backbone frozen for the first 3 epochs.
  - Fold model: the average of the weights after epochs 31–50 of the same run (SWA), with BatchNorm statistics
    re-estimated.
  - Raw softmax probabilities, no calibration. Nothing is selected on the validation split.
- **Guidance:**
  - **Mask:** Grad-CAM of the predicted class on the validation split, from the last kept stage. Built after epoch 18,
    rebuilt every 3 epochs from epoch 21. For each sample it takes the top 22% of its gene pixels by "wrong-reason"
    score: evidence against the true class if the prediction is right, evidence for the wrong class if not. One mask
    per class keeps the pixels picked in more than 20% of that class's samples; gene pixels only. The top 50% of each
    omics' genes by ANOVA F on the training split are never masked.
  - **Penalty**, from epoch 19: λ = 1000 × the mean over samples of ‖∇ₓS ⊙ m‖² / |m|.
    - S is log p(LumA|x) − log p(LumB|x) for LumA and LumB samples, and log p(y|x) for the others.
    - It is computed on 4 random samples of each batch, in training mode, with BatchNorm running statistics left
      untouched.
    - Its gradient is exact, from one forward-mode (JVP) pass; double backward is about 40× slower for depthwise
      convolutions. The focal loss uses the whole batch.
- **Evaluation:** repeated stratified CV with 6 seeds × 5 outer folds (20% test). The rest of each fold is split
  64% train / 16% validation. Every sample is tested once per seed.

## Setup

```
pip install -r requirements.txt
```

R ≥ 4.0 is needed for phase 7. Once, in R:

```
install.packages("BiocManager")
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "enrichplot", "ggplot2"))
```

Without R, phase 7 is skipped with a note. The data location is described in `data/README.md`.

## Usage

```
python run_pipeline.py                                   # config.yaml, phases 1-7
python run_pipeline.py --phases 3 4 5 6 7                # downstream of a finished training
python run_pipeline.py --seeds 12 3 --out runs/dev       # two seeds, another run directory
python run_pipeline.py --set guidance.enabled=false --out runs/no_guidance     # the same model without guidance
python tests/test_pipeline.py                            # 9 checks incl. an end-to-end run of phases 1-6 (CPU)
```

- **Settings:** every setting is in `config.yaml`, documented there. `--set section.key=value` overrides any of them.
- **Rejected settings:** unknown keys and invalid values are refused.
- **Mixed runs:** a run directory refuses to mix folds made with different settings.

## Run directory (`runs/<name>/`)

| Path | Content |
|---|---|
| `config_resolved.yaml`, `run.log`, `dataset_info.json` | the settings, the log, the class counts |
| `1_images/dataset.npz`, `seed_<s>/splits.json` | sample IDs, labels, feature names; the train / val / test indices |
| `1_images/seed_<s>/fold_<f>/` | `images.npz` (uint8, ~12 MB), `layout.npz`, `gene_coordinates_<omics>.csv`, `anova_rank.npz` |
| `2_training/summary.md` | test metrics: mean ± std over the 30 fold models, and pooled out-of-fold per seed; also the multi-seed ensemble and the mask check |
| `2_training/` (other files) | `fold_metrics.csv`, `seed_metrics.csv`, `oof_predictions_all_seeds.csv`, `mask_check.csv` |
| `2_training/seed_<s>/fold_<f>/` | `predictions.csv`, `history.csv`, `fold_result.json`, `model.pt`, `gene_attributions.npz`, `guidance_mask.npy` |
| `3_selected_genes_top400/` | `label_<subtype>_top_400_<sign>.csv`, `all_selected_features.csv`, `selection_stability_across_seeds.csv`, `background_features.csv`, `report.txt` |
| `4_merged_genes/` | `unique_genes_omics.csv`, `pam50_overlap.csv`, `report.txt` |
| `5_filtered_omics/` | `<omics file>_selected.csv`, `labels.csv` |
| `6_pathway/` | `MSigDB_Hallmark_2020_<list>.csv` (every tested set), `summary.csv` |
| `7_go_kegg/` | `GO_BP_<subtype>_positive.csv`, `KEGG_<subtype>_positive.csv`, dot plots, `go_kegg_enrichment_summary.csv` |

## Differences from DeepCamCS

| | DeepCamCS-main | Here |
|---|---|---|
| Image layout fitted on | all samples, test included | the training split of each fold |
| Pixel scaling | the lowest gene value equals the empty background | p5–p95 clip; 0 reserved for empty pixels |
| Model | EfficientNet-B1, global average pooling | one trunk per omics, readout at every gene's pixel |
| Guidance | one Grad-CAM mask; penalty on ∇ₓ Σ_c log p_c, double backward | per-class gene-only mask, ANOVA-protected genes; LumA/LumB pair penalty; exact JVP gradient on 4 samples per batch |
| Fold model | best validation F1 | SWA of epochs 31–50 (no selection on the validation split) |
| Evaluation | one 10% holdout, one seed | 6 seeds × 5 outer folds, every sample tested once per seed |
| Phase 4 | PAM50 overlap over all omics | per gene-level omics and subtype; methylation probes and CNV segments are not genes |
| Phase 6 | Enrichr web service (the gene list is uploaded), genome background | local hypergeometric test against the measured genes; only the gene-set library is downloaded, once, into `cache/` |
| Phase 7 | genome background | the measured genes as the universe |

## Verification

The settings in `config.yaml` are those of `omics_gradcam_pipeline/runs/brca_gene_guided`.

- **Phase 1:** the layouts and gene coordinates of all 5 folds of seed 12 are identical to that run's.
- **Phases 3–7:** `runs/brca_gene_guided_imported/` holds that run's 30 folds of attributions, with `2_training/`
  imported; see `IMPORTED.txt` there. Phase 3 reproduces its gene lists and stability table exactly, and phase 7 its
  GO / KEGG counts. Phases 4–6 are new for that run:
  - all 13 PAM50 genes present in the mRNA data were selected;
  - Hallmark: *Estrogen Response Early / Late* is enriched in every subtype's list. The pooled list adds
    *Epithelial–Mesenchymal Transition*, *Interferon Gamma Response* and *UV Response Down*.
- **Phase 2:** checked on the GPU against fold 1 of seed 12 (see the note at the end of this file).

That run's results (30 folds): accuracy 85.23 ± 3.44, macro F1 84.30 ± 3.69, ROC-AUC 0.966. The 6-seed
ensemble reaches 87.45 / 87.25.

## Run time and memory (RTX-class 16 GB GPU)

- **Phase 1:** about 35 s per fold (UMAP).
- **Phase 2:** about 13 minutes per fold, so about 6.5 hours for 30 folds. Guided epochs take about 21 s and
  unguided ones about 3 s.
  - The guided step peaks at 13.8 GB. 6 penalised samples per batch do not fit.
  - `--gpu-memory-fraction` (default 0.9) caps PyTorch. Without the cap, Windows spills into system RAM and
    training becomes 50–100× slower.
- **Phases 3–7:** about 1 minute together.
