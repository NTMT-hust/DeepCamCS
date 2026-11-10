# DeepCamCS: Guided Learning with Grad-CAM for Multi-Omics Cancer Subtype Classification

An end-to-end deep learning and explainable AI (XAI) framework that transforms high-dimensional multi-omics data (mRNA, DNA Methylation, CNV) into structured multi-channel images, classifies cancer subtypes using EfficientNet-B1 with Focal Loss and Explanation-Guided Learning, and extracts biologically actionable biomarker genes and functional pathways.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Repository Structure](#-repository-structure)
- [Methodology Highlights](#-methodology-highlights)
  - [1. Multi-Omics to Image Transformation (Cart2Pixel & Snowfall)](#1-multi-omics-to-image-transformation-cart2pixel--snowfall)
  - [2. Model Architecture & Focal Loss](#2-model-architecture--focal-loss)
  - [3. Explanation-Guided Learning via Grad-CAM](#3-explanation-guided-learning-via-grad-cam)
  - [4. Post-Hoc Attribution & Biomarker Discovery](#4-post-hoc-attribution--biomarker-discovery)
- [Installation & Environment Setup](#-installation--environment-setup)
- [End-to-End Execution Guide](#-end-to-end-execution-guide)
  - [Step 1: Generate Multi-Omics Images](#step-1-generate-multi-omics-images)
  - [Step 2: Train with Stratified K-Fold CV & Guided Learning](#step-2-train-with-stratified-k-fold-cv--guided-learning)
  - [Step 3: Select Top Attributed Biomarker Genes](#step-3-select-top-attributed-biomarker-genes)
  - [Step 4: Merge Omics & Evaluate PAM50 Overlap](#step-4-merge-omics--evaluate-pam50-overlap)
  - [Step 5: Filter Raw Omics Matrices](#step-5-filter-raw-omics-matrices)
  - [Step 6: Biological Pathway Enrichment Analysis](#step-6-biological-pathway-enrichment-analysis)
  - [Step 7: GO & KEGG Functional Enrichment Analysis](#step-7-go--kegg-functional-enrichment-analysis)
- [Command-Line Arguments Reference](#-command-line-arguments-reference)
- [Expected Output Structure](#-expected-output-structure)

---

## 🔬 Overview

High-throughput multi-omics profiling offers comprehensive molecular views of cancer subtypes, but faces challenges from high dimensionality, sample scarcity, severe class imbalance, and "black-box" predictions. 

**DeepCamCS** addresses these challenges by:
1. **Representing Multi-Omics as Images:** Projects multi-omics modalities into 2D spatial layouts where each channel encodes a distinct modality (Channel 0: mRNA, Channel 1: DNA Methylation, Channel 2: Copy Number Variation).
2. **Integrating Explanation Feedback during Training:** Employs Grad-CAM feedback masks to penalize gradients in regions driving erroneous predictions ("Right for the Right Reasons" guided learning).
3. **Addressing Class Imbalance:** Uses multi-class Focal Loss and inverse class weighting to ensure robust performance on rare cancer subtypes.
4. **Translating Spatial Explanations to Biomarkers:** Reverse-maps high-attribution pixels to individual gene symbols across all omics channels, validating them against established cancer signatures (such as PAM50 for breast cancer) and biological hallmark pathways.

---

## 🏗 Pipeline Architecture

<p align="center">
  <img src="architecture.png" alt="DeepCamCS Framework Architecture" width="95%"/>
</p>

*Figure 1: DeepCamCS framework architecture. **(A)** Tabular multi-omics profiles ($x_n$: mRNA, DNA Methylation, CNV) are transformed into structured 2D spatial multi-channel images ($X_n$). **(B)** An EfficientNet-B1 convolutional neural network extracts deep features from the last convolutional layer to classify cancer subtypes ($Y_n \to$ Softmax). **(C)** Gradient-based explanations (Grad-CAM) compute feature attributions from the last conv layer, generating explained attribution heatmaps and driving a guided-learning loss feedback loop to regularize model attention.*


---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `produceIMG.py` | Converts raw omics matrices (mRNA, Methylation, CNV) into 3-channel 2D spatial images using UMAP, minimum bounding rotated rectangle, and the fast Snowfall collision-resolution algorithm. Also exports coordinate lookup CSVs. |
| `Dataset.py` | `OmicsImageDataset`: loads the generated images and keeps only the selected omics channels (single/dual-omics ablations). |
| `models.py` | Backbone factory: any convolutional `timm` model (EfficientNet, ResNet, ResNet26d, ...) or a configurable U-Net classifier, all exposing `forward_features` / `forward_head` for Grad-CAM. |
| `FocalLoss.py` | Multi-class Focal Loss with optional class weights ($\alpha$) and focusing parameter ($\gamma$). With $\gamma=0$ it is class-weighted cross entropy. |
| `guidance.py` | Explanation-guided learning: guidance-mask construction (Grad-CAM or sparse IG, top-k, frequency consensus, ANOVA-protected genes) and the input-gradient penalty. |
| `additional_function.py` | Dataset folder discovery, class weights, and evaluation metrics (confusion matrix, precision/recall/F1/MCC/AUC, sensitivity/specificity). |
| `StratifiedKFoldCrossValidation.py` | K-fold engine: SGD + stochastic weight averaging, mask refresh from the validation split, holdout evaluation of the SWA model, and IG records. || `train.py` | Main pipeline CLI entry point: executes stratified cross-validation, generates performance curve figures, writes test result JSON, and triggers pixel-to-gene explanation mapping. |
| `ProcessHeatMapResult.py` | Processes attribution heatmaps: computes class-average heatmaps, identifies top 10% critical pixels, and maps pixel coordinates back to biological gene names across each omics modality. |
| `ChooseGenes.py` | Ranks and selects the top $K$ positive and negative attributed genes per cancer subtype based on median attribution scores. |
| `merge_genes_omics.py` | Aggregates selected genes across modalities, deduplicates them, and benchmarks overlap against the breast cancer PAM50 gold-standard gene panel. |
| `filter_omics_by_unique_genes.py` | Subsets original raw omics matrices (mRNA, CNV, Methylation) using the finalized list of unique biomarker genes. |
| `pathway_analysis.py` | Performs pathway enrichment analysis on extracted biomarker genes using `gseapy` / Enrichr (e.g., `MSigDB_Hallmark_2020`). |
| `go_kegg_enrichment.R` | Executes subtype-specific (or pooled) Gene Ontology (GO Biological Process) and KEGG pathway overrepresentation analysis via `clusterProfiler` with Benjamini-Hochberg FDR correction and dotplot visualizations. |

---

## 💡 Methodology Highlights

### 1. Multi-Omics to Image Transformation (Cart2Pixel & Snowfall)
Raw genomics data lack inherent spatial relationships. In `produceIMG.py`:
- High-dimensional gene features across samples are projected into 2D space via cosine metric **UMAP**.
- Coordinates are aligned using a **minimum rotated bounding rectangle** to maximize spatial utilization.
- Overlapping gene positions are resolved onto a discrete grid using the **Snowfall** breadth-first search (BFS) collision-handling algorithm.
- Gene expression/methylation/CNV values are mapped to pixels using `ConvPixel` with min-max and $\log(1 + x)$ scaling.
- mRNA (Red), Methylation (Green), and CNV (Blue) are stacked into a 3-channel image (default $224 \times 224$), preserving gene coordinate consistency across all samples.

### 2. Model Architecture, Loss & Optimisation
- Backbone (`--backbone`): EfficientNet-B1 by default; any convolutional `timm` name (`resnet26d`, `resnet50`, ...) or `unet`. The dropout/linear head comes from `timm` (`drop_rate`).
- Channels (`--channels`): any subset of `mRNA Methylation CNV`; the input stem adapts to the channel count.
- Loss (`--loss focal|ce`): **Focal Loss** $-\alpha_t (1 - p_t)^\gamma \log(p_t)$ or cross entropy (the same loss with $\gamma=0$). Class weighting $\alpha_t$ (inverse class frequency) is shared by both and can be disabled with `--no-class_weights`.
- Optimiser: **SGD with Nesterov momentum**, cosine decay to `--swa_lr`, then **stochastic weight averaging** from `--swa_start_epoch`. The SWA model (BatchNorm statistics recomputed) is the final model of each fold; no checkpoint is selected on validation data.
- Split: 20% stratified holdout (`--holdout_fraction`), then stratified K-fold on the remainder. The validation fold is used only to build the guidance mask (and to produce IG explanation records); the holdout is evaluated once per fold with the SWA model.

### 3. Explanation-Guided Learning
Enabled when `--lambda1 > 0` (`--lambda1 0` gives the unguided baseline). From epoch `--mask_start_epoch`, and every `--mask_refresh_every` epochs after, a mask of "wrong-reason" pixels is built on the validation split:
1. Per-sample attribution for the predicted class (`--mask_method gradcam` or `ig`).
2. Sign flip by correctness: `-score` if the prediction is correct, `+score` if wrong.
3. Per-sample top 22% gene pixels (pixels with value 0 are background and excluded).
4. Keep pixels selected in more than 20% of samples of a true class (class-conditional mask).
5. Remove the top 50% of features per channel by ANOVA F-statistic (computed on the training split).

The penalty is added to the classification loss: $\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_1 \cdot \frac{1}{B}\sum_b \frac{\lVert \nabla_{x_b} \sum_c \log p_c \odot m_{y_b} \rVert_2^2}{|m_{y_b}|}$.
### 4. Post-Hoc Attribution & Biomarker Discovery
- Pixel attributions from the SWA models across folds are aggregated per predicted class.
- The `(pixel_x, pixel_y)` coordinates are cross-referenced with `gene_coordinates_{mRNA,Methylation,CNV}.csv` to assign attribution scores to individual genes.
- Biomarkers are ranked to identify subtype-specific drivers, validated against PAM50, and enriched for biological pathways using GSEA/Enrichr.

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Python 3.9+
- R 4.0+ (for Step 7 GO & KEGG enrichment analysis)
- CUDA-enabled GPU (recommended for EfficientNet training and Grad-CAM extraction)

### Install Dependencies

**Python packages:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm opencv-python numpy pandas matplotlib seaborn scikit-learn scipy shapely umap-learn imageio tqdm gseapy
```

**R packages (for Step 7):**
> *Note:* `go_kegg_enrichment.R` automatically checks and installs missing CRAN and Bioconductor packages on first run. You can also install them ahead of time:
```R
install.packages(c("dplyr", "ggplot2", "optparse", "BiocManager"))
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "enrichplot"))
```

---

## 🚀 End-to-End Execution Guide

### Step 1: Generate Multi-Omics Images

Transform your aligned multi-omics matrices into 3-channel 2D images:

```bash
python produceIMG.py
```

> **Note:** Configure `base_dir`, `out_dir`, and input filenames in `produceIMG.py` (or modify to fit your data layout). This outputs:
> - `OutputData/train_val/{class_label}/sample_*.png`
> - `OutputData/test/{class_label}/sample_*.png`
> - `OutputData/gene_coordinates_{mRNA, Methylation, CNV}.csv`

---

### Step 2: Train with Stratified K-Fold CV & Guided Learning

Train with 5-fold cross-validation, SWA, holdout evaluation, and IG attribution generation. Ablations are command-line switches, for example `--channels mRNA --backbone resnet26d --loss ce --lambda1 0` for a single-omics ResNet26d cross-entropy baseline without guidance:

```bash
python train.py \
    --dataset_path "OutputData/dataset" \
    --coord_base "OutputData" \
    --output_dir "OutputData/Explanation_IG" \
    --k_folds 5 \
    --num_epochs 70 \
    --batch_size 32 \
    --lr 0.01 \
    --weight_decay 0.001 \
    --dropout_rate 0.5 \
    --focal_gamma 2.0 \
    --lambda1 10 \
    --results_plot "OutputData/imbalanced_kfold_comprehensive_results.png" \
    --test_metrics_plot "OutputData/test_metrics_by_fold.png" \
    --test_results_json "fold_test_results.json"
```

**Key Outputs from Step 2:**
- `swa_model_fold{0..4}.pth`: SWA model per fold.
- `confusion_matrix_fold{1..5}_test.png`: Test set confusion matrices.
- `imbalanced_kfold_comprehensive_results.png`: Training loss and metric trajectories.
- `Explanation/mean_heatmap_class_{label}.png`: Class-average attribution maps.
- `Explanation/predicted_label_{label}_attribute_scores.csv`: Per-gene attribution scores.
- `Explanation/all_classes_attribute_scores.csv`: Master attribution table.

---

### Step 3: Select Top Attributed Biomarker Genes

Extract the top $K$ positive and negative attributing genes per cancer subtype:

```bash
python ChooseGenes.py \
    --base_dir "Explanation" \
    --k 400 \
    --output_dir "selected_genes_400"
```

**Outputs:**
- `selected_genes_400/predicted_label_{cls}_top_400_positive.csv`
- `selected_genes_400/predicted_label_{cls}_top_400_negative.csv`

---

### Step 4: Merge Omics & Evaluate PAM50 Overlap

Merge all top-ranked genes across modalities, deduplicate them, and evaluate overlap with the PAM50 breast cancer gene signature:

```bash
python merge_genes_omics.py \
    --folder "selected_genes_400" \
    --output_dir "Top"
```

**Outputs & Terminal Summary:**
- `Top/unique_genes_omics.csv`: Deduplicated `(unique_genes, omics_type)` pairs.
- Terminal log reporting total unique genes and detailed PAM50 overlap list.

---

### Step 5: Filter Raw Omics Matrices

Subset the original omics data matrices down to the extracted candidate biomarker genes:

```bash
python filter_omics_by_unique_genes.py \
    --unique_genes_file "Top/unique_genes_omics.csv" \
    --cnv_file "path/to/BRCA_CNV_aligned.csv" \
    --mrna_file "path/to/BRCA_mRNA_aligned.csv" \
    --methy_file "path/to/BRCA_Methy_aligned.csv" \
    --output_dir "Top"
```

**Outputs:**
- `Top/BRCA_CNV_aligned_filtered.csv`
- `Top/BRCA_mRNA_aligned_filtered.csv`
- `Top/BRCA_Methy_aligned_filtered.csv`

---

### Step 6: Biological Pathway Enrichment Analysis

Perform functional pathway enrichment analysis using Enrichr and MSigDB Hallmark gene sets:

```bash
python pathway_analysis.py \
    --gene_file "Top/unique_genes_omics.csv" \
    --gene_column "unique_genes" \
    --gene_set "MSigDB_Hallmark_2020" \
    --output_file "hallmark_pathway_overlap.csv"
```

**Outputs:**
- `hallmark_pathway_overlap.csv`: Table containing Term, overlap gene counts, adjusted p-values, and gene member lists.

---

### Step 7: GO & KEGG Functional Enrichment Analysis

Perform subtype-specific Gene Ontology (GO Biological Process) and KEGG pathway overrepresentation analysis using `clusterProfiler` with Benjamini-Hochberg FDR correction ($P_{\text{adj}} \le 0.05$).

Unlike Step 6 (which tests a single pooled list of genes across all classes), Step 7 runs **per subtype** directly on Step 3's top positive attributing genes (`predicted_label_{cls}_top_{k}_positive.csv`) to preserve subtype-specific biological signals:

```bash
# Default mode: Subtype-specific enrichment on top K positive genes
Rscript go_kegg_enrichment.R \
    --gene_dir "selected_genes_400" \
    --k 400 \
    --gene_column "gene_name" \
    --gene_set "BP" \
    --pvalue_cutoff 0.05 \
    --output_dir "GO_KEGG_Enrichment"
```

You can also run in `single_file` mode on the pooled gene list from Step 4 for comparison:

```bash
# Single file mode: Pooled enrichment across all subtypes
Rscript go_kegg_enrichment.R \
    --mode single_file \
    --gene_file "Top/unique_genes_omics.csv" \
    --gene_column "unique_genes" \
    --output_dir "GO_KEGG_Enrichment"
```

**Outputs:**
- `GO_KEGG_Enrichment/GO_BP_{subtype}.csv`: Significant GO Biological Process terms ($P_{\text{adj}} \le 0.05$).
- `GO_KEGG_Enrichment/KEGG_{subtype}.csv`: Significant KEGG pathways ($P_{\text{adj}} \le 0.05$).
- `GO_KEGG_Enrichment/GO_BP_{subtype}_dotplot.png`: Dotplots showing top enriched GO terms.
- `GO_KEGG_Enrichment/KEGG_{subtype}_dotplot.png`: Dotplots showing top enriched KEGG pathways.
- `GO_KEGG_Enrichment/go_kegg_enrichment_summary.csv`: Master summary table recording mapped genes and significant term counts across subtypes.

---

## 📋 Command-Line Arguments Reference

### `train.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--dataset_path` | `str` | `...` | Root directory containing image class subfolders |
| `--coord_base` | `str` | `...` | Directory containing `gene_coordinates_*.csv` files |
| `--output_dir` | `str` | `Explanation` | Directory where heatmaps and attribution tables are saved |
| `--results_plot` | `str` | `...results.png` | Output path for the training-curve plot |
| `--test_metrics_plot` | `str` | `test_metrics_by_fold.png` | Output path for held-out test metrics by fold |
| `--test_results_json` | `str` | `fold_test_results.txt` | Path to save JSON evaluation results |
| `--k_folds` | `int` | `5` | Number of folds for Stratified Cross-Validation |
| `--num_epochs` | `int` | `70` | Total training epochs per fold |
| `--batch_size` | `int` | `32` | Training and validation batch size |
| `--lr` | `float` | `0.01` | Initial SGD learning rate |
| `--weight_decay` | `float` | `0.001` | SGD weight decay |
| `--momentum` | `float` | `0.9` | SGD momentum |
| `--swa_lr` | `float` | `0.005` | Constant learning rate during SWA |
| `--swa_start_epoch` | `int` | `40` | Epoch index where weight averaging starts |
| `--channels` | `list` | `mRNA Methylation CNV` | Omics channels to train on |
| `--backbone` | `str` | `efficientnet_b1` | `unet` or any convolutional timm model |
| `--pretrained` | `flag` | `True` | ImageNet weights for timm backbones (`--no-pretrained` to disable) |
| `--unet_width` / `--unet_depth` | `int` | `32` / `4` | U-Net size |
| `--loss` | `str` | `focal` | `focal` or `ce` |
| `--class_weights` | `flag` | `True` | Inverse-frequency class weights (`--no-class_weights` to disable) |
| `--holdout_fraction` | `float` | `0.2` | Held-out test fraction |
| `--mask_method` | `str` | `gradcam` | `gradcam` or `ig` |
| `--mask_start_epoch` / `--mask_refresh_every` | `int` | `40` / `5` | Mask schedule |
| `--checkpoint_dir` | `str` | `.` | Where SWA checkpoints are saved |
| `--dropout_rate` | `float` | `0.5` | Dropout probability before classification layer |
| `--focal_gamma` | `float` | `2.0` | Focusing parameter $\gamma$ for Focal Loss |
| `--lambda1` | `float` | `5` | Penalty weight for explanation-guided loss (`0` disables it) |
| `--random_seed` | `int` | `42` | Seed for data splits and model initialization |
| `--show_plot` | `flag` | `False` | Display the results plot after training |

### `go_kegg_enrichment.R`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--mode` | `str` | `per_subtype` | Enrichment mode: `per_subtype` (processes each subtype file) or `single_file` |
| `--gene_dir` | `str` | `selected_genes_400` | Input directory containing `predicted_label_*_top_{k}_positive.csv` |
| `--k` | `int` | `400` | Number of top genes matching filenames in `ChooseGenes.py` |
| `--gene_file` | `str` | `Top/unique_genes_omics.csv` | Path to single gene CSV (used when `--mode single_file`) |
| `--gene_column` | `str` | `gene_name` | Name of the column containing gene symbols |
| `--gene_set` | `str` | `BP` | Gene Ontology domain: `BP` (Biological Process), `MF` (Molecular Function), or `CC` (Cellular Component) |
| `--organism` | `str` | `hsa` | KEGG organism code (`hsa` for Homo sapiens) |
| `--pvalue_cutoff`| `float` | `0.05` | Benjamini-Hochberg adjusted P-value significance threshold |
| `--output_dir` | `str` | `GO_KEGG_Enrichment` | Directory where enrichment tables, dotplots, and summary are saved |

---

## 📊 Expected Output Structure

```
.
├── Explanation/
│   ├── all_classes_attribute_scores.csv
│   ├── critical_pixels_class_*.csv
│   ├── mean_heatmap_class_*.png
│   ├── predicted_label_*_attribute_scores.csv
│   └── {cnv,methylation,mrna}_class_*.csv
├── GradCAM/
│   └── Fold_*/
│       └── sample_*_gradcam.png
├── selected_genes_400/
│   ├── predicted_label_*_top_400_negative.csv
│   └── predicted_label_*_top_400_positive.csv
├── Top/
│   ├── unique_genes_omics.csv
│   ├── BRCA_CNV_aligned_filtered.csv
│   ├── BRCA_mRNA_aligned_filtered.csv
│   └── BRCA_Methy_aligned_filtered.csv
├── GO_KEGG_Enrichment/
│   ├── GO_BP_*.csv
│   ├── GO_BP_*_dotplot.png
│   ├── KEGG_*.csv
│   ├── KEGG_*_dotplot.png
│   └── go_kegg_enrichment_summary.csv
├── swa_model_fold*.pth
├── confusion_matrix_fold*_test.png
├── hallmark_pathway_overlap.csv
├── imbalanced_kfold_comprehensive_results.png
└── fold_test_results.json
```
