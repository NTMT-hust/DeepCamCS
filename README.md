# DeepCamCS

DeepCamCS turns aligned cancer omics matrices into RGB images, trains an EfficientNet-B1 subtype classifier, and maps model explanations back to genes for downstream pathway analysis. The fold-first pipeline supports mRNA, DNA methylation, and copy-number variation (CNV), including single-modality and pairwise runs.

## Pipeline overview

For each stratified outer fold, the pipeline:

1. Holds out one fold as the test partition.
2. Splits the remaining samples into training and validation sets (80/20 by default).
3. Fits one UMAP gene layout per selected modality using only that fold's training-sample profiles. The fixed gene coordinates are then used to render train, validation, and test samples. Validation/test measurements do not fit or alter the layout.
4. Trains on the training images. Validation Grad-CAM continues to provide the explanation-guided feedback mask; the best checkpoint is selected by validation macro-F1.
5. Evaluates the selected checkpoint on the held-out test set after training.
6. Calculates post-hoc Integrated Gradients for validation samples, maps scores to genes, and runs the optional selection, merge, filtering, and enrichment stages.

The RGB channel positions stay fixed: mRNA is red, methylation is green, CNV is blue. Channels for unselected modalities are black. The default image size is 240 × 240. For a five-fold run, each sample is used as a test sample once; within each run the approximate proportions are 64% train, 16% validation, and 20% test.

## Input data

Set `DEEPCAM_DATA_DIR` to a directory containing the following files:

| File | Format |
| --- | --- |
| `BRCA_mRNA_aligned.csv` | Genes/features in rows, sample IDs in columns; first column contains gene names |
| `BRCA_Methy_aligned.csv` | Same orientation and sample IDs as the selected data |
| `BRCA_CNV_aligned.csv` | Same orientation and sample IDs as the selected data |
| `BRCA_label_num.csv` | Either `Sample_ID,Label` (header names are not required), or a one-column label file in the same order as the first selected matrix |

Only the selected omics matrices are loaded. Samples are intersected across selected matrices and labels. Labels are treated as class-folder names.

## Install

Use Python 3.9 or newer. Install the Python packages declared in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

PyTorch should match your CPU or CUDA environment; install the appropriate PyTorch build if the requirements installation does not provide it. The GO/KEGG stage requires R and installs its missing CRAN/Bioconductor packages when first run. It also needs network access for package setup and KEGG queries. The Hallmark stage uses Enrichr through `gseapy` and requires network access.

## Run the full pipeline

```bash
export DEEPCAM_DATA_DIR="/path/to/aligned_csvs"
export DEEPCAM_OUTPUT_DIR="/path/to/results"
export OMICS="mRNA,Methylation,CNV"
bash run_pipeline.sh
```

Select one modality or a pair by setting `OMICS`, for example:

```bash
export OMICS="mRNA"
# or
export OMICS="mRNA,CNV"
bash run_pipeline.sh
```

Valid names are `mRNA`, `Methylation`, and `CNV`. The default is all three.

### Run each modality experiment

The following Bash commands run the full pipeline for all seven single- and multi-omics combinations. Replace the input path with your data directory. Each run has its own output folder so folds, checkpoints, and downstream results stay separate.

```bash
export DEEPCAM_DATA_DIR="/path/to/aligned_csvs"

OMICS=mRNA DEEPCAM_OUTPUT_DIR="./results/mRNA" bash run_pipeline.sh
OMICS=Methylation DEEPCAM_OUTPUT_DIR="./results/Methylation" bash run_pipeline.sh
OMICS=CNV DEEPCAM_OUTPUT_DIR="./results/CNV" bash run_pipeline.sh

OMICS=mRNA,Methylation DEEPCAM_OUTPUT_DIR="./results/mRNA_Methylation" bash run_pipeline.sh
OMICS=mRNA,CNV DEEPCAM_OUTPUT_DIR="./results/mRNA_CNV" bash run_pipeline.sh
OMICS=Methylation,CNV DEEPCAM_OUTPUT_DIR="./results/Methylation_CNV" bash run_pipeline.sh

OMICS=mRNA,Methylation,CNV DEEPCAM_OUTPUT_DIR="./results/all_omics" bash run_pipeline.sh
```

To run only image generation and model training for an experiment, disable the downstream analysis stages:

```bash
OMICS=mRNA DEEPCAM_OUTPUT_DIR="./results/mRNA" \
  RUN_GENE_SELECTION=0 RUN_MERGE=0 RUN_FILTER=0 RUN_HALLMARK=0 RUN_GO_KEGG=0 \
  bash run_pipeline.sh
```

Set `K_FOLDS`, `NUM_EPOCHS`, or other configuration variables before these commands to use the same settings for every experiment. For separate settings, set the values inline on the specific command.

## Configuration

These settings are environment variables read by `run_pipeline.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `K_FOLDS` | `5` | Number of stratified outer folds; each class needs enough samples for the requested fold count |
| `VAL_FRACTION` | `0.2` | Validation share of each outer training partition |
| `OMICS` | `mRNA,Methylation,CNV` | One or more comma-separated modalities |
| `NUM_EPOCHS` | `100` | Training epochs per fold |
| `BATCH_SIZE` | `24` | Training batch size |
| `LEARNING_RATE` | `0.0001` | AdamW learning rate |
| `DROPOUT_RATE` | `0.5` | Classifier dropout |
| `FOCAL_GAMMA` | `2.0` | Focal-loss focusing parameter |
| `LAMBDA1` | `10` | Explanation-guided loss weight |
| `LAMBDA2` | `0.001` | AdamW weight decay |
| `LOSS` | `focal` | `focal` or `cross_entropy` |
| `GUIDED_LEARNING` | `1` | Set to `0` to disable the explanation-guided penalty |
| `PRETRAINED` | `1` | Set to `0` to initialize EfficientNet without pretrained weights |
| `TOP_K` | `100` | Positive/negative genes selected per class and modality |

Each pipeline stage can be turned on or off with `1` or `0`:

| Variable | Stage |
| --- | --- |
| `RUN_GENERATION` | Create fold splits, UMAP coordinates, and images |
| `RUN_TRAINING` | Train and evaluate each fold |
| `RUN_GENE_SELECTION` | Aggregate attribution tables and select top genes |
| `RUN_MERGE` | Merge selected genes and report PAM50 overlap |
| `RUN_FILTER` | Filter the source matrices to selected genes |
| `RUN_HALLMARK` | Run pooled MSigDB Hallmark enrichment |
| `RUN_GO_KEGG` | Run subtype-specific GO and KEGG enrichment |

All stage switches default to `1`. When skipping a stage, its expected outputs must already exist. For example, to rerun only training from previously generated folds, set `RUN_GENERATION=0` and `RUN_TRAINING=1`. `PYTHON` and `RSCRIPT` can select non-default executable names.

To launch the pipeline in the background and save stdout and errors to a timestamped `.txt` log, set `RUN_IN_BACKGROUND=1`. The script prints the process ID and log path when it starts:

```bash
RUN_IN_BACKGROUND=1 bash run_pipeline.sh
```

The default log is written under `DEEPCAM_OUTPUT_DIR` (or `OutputData/` if unset). Choose another path with `PIPELINE_LOG_FILE=/path/to/pipeline.txt`. Follow the log with `tail -f /path/to/pipeline.txt`. Foreground execution remains the default; use `RUN_IN_BACKGROUND=0` to make it explicit.

## Fold outputs

The output directory is organized by outer fold:

```text
results/
├── Fold_1/
│   ├── train/<class>/*.png
│   ├── val/<class>/*.png
│   ├── test/<class>/*.png
│   ├── gene_coordinates_<modality>.csv
│   ├── sample_splits.csv
│   ├── Explanation_IG/
│   │   ├── ig_mean_heatmap_*.png
│   │   ├── predicted_label_*_attribute_scores.csv
│   │   └── all_classes_attribute_scores.csv
│   └── Training/
│       ├── best_model_fold0.pth
│       ├── confusion_matrix_fold1_test.png
│       ├── GradCAM/Fold_1/
│       ├── learning_curves.png
│       ├── test_metrics.png
│       └── test_results.json
├── Fold_2/
│   └── ...
├── selected_genes_100/
├── Top/
├── GO_KEGG_Enrichment/
└── hallmark_pathway_overlap.csv
```

`sample_splits.csv` records each sample's label and train/validation/test assignment. Coordinate files are specific to their fold because each fold fits its own training-only UMAP layout. `ChooseGenes.py` combines per-fold attribution CSVs by taking the median score for each class, modality, and gene before selecting top positive and negative scores.

## Individual stages

The full workflow is normally run through `run_pipeline.sh`. The main scripts can also be invoked individually:

```bash
python produceIMG.py --data-dir "$DEEPCAM_DATA_DIR" --output-dir "$DEEPCAM_OUTPUT_DIR" \
  --omics mRNA,CNV --k-folds 5 --val-fraction 0.2
```

To train a generated fold directly:

```bash
python train.py --dataset_path "$DEEPCAM_OUTPUT_DIR/Fold_1" \
  --coord_base "$DEEPCAM_OUTPUT_DIR/Fold_1" \
  --output_dir "$DEEPCAM_OUTPUT_DIR/Fold_1/Explanation_IG" \
  --artifact_dir "$DEEPCAM_OUTPUT_DIR/Fold_1/Training" \
  --num_epochs 100 --batch_size 24 --lr 0.0001
```

`train.py` detects `train/`, `val/`, and `test/` under `--dataset_path` and uses those partitions. For a legacy dataset containing only class folders, it retains the older internal cross-validation path.

Downstream commands:

```bash
python ChooseGenes.py --base_dir "$DEEPCAM_OUTPUT_DIR" --k 100 \
  --output_dir "$DEEPCAM_OUTPUT_DIR/selected_genes_100"
python merge_genes_omics.py --folder "$DEEPCAM_OUTPUT_DIR/selected_genes_100" \
  --output_dir "$DEEPCAM_OUTPUT_DIR/Top"
python filter_omics_by_unique_genes.py \
  --unique_genes_file "$DEEPCAM_OUTPUT_DIR/Top/unique_genes_omics.csv" \
  --cnv_file "$DEEPCAM_DATA_DIR/BRCA_CNV_aligned.csv" \
  --mrna_file "$DEEPCAM_DATA_DIR/BRCA_mRNA_aligned.csv" \
  --methy_file "$DEEPCAM_DATA_DIR/BRCA_Methy_aligned.csv" \
  --output_dir "$DEEPCAM_OUTPUT_DIR/Top"
```

The pathway scripts accept their own options via `--help`. `pathway_analysis.py` runs pooled Enrichr/MSigDB Hallmark enrichment. `go_kegg_enrichment.R` defaults to per-subtype enrichment using the selected positive gene files; it also supports `--mode single_file` for a pooled list.

## Model and explanation notes

- The classifier is `timm` EfficientNet-B1 with a dropout and linear classification head. All backbone parameters are trained.
- The default loss is class-weighted multi-class Focal Loss; weighted cross-entropy is available with `LOSS=cross_entropy`.
- During training, validation Grad-CAM is used to update the explanation-guided mask after the 40-epoch warmup and every five epochs thereafter. The loss penalty can be disabled with `GUIDED_LEARNING=0`.
- Checkpoint selection uses validation macro-F1. The held-out test partition is evaluated only after training completes, using the selected checkpoint.
- Post-hoc gene attribution uses Sparse Integrated Gradients on validation samples, grouped by predicted class and mapped through the fold-specific coordinate CSVs. Grad-CAM and Integrated Gradients serve different roles in the implementation.
- `merge_genes_omics.py` reports overlap with the PAM50 breast cancer panel. Its gene list is pooled across selected modalities and subtypes after gene selection.

## Repository files

| File | Purpose |
| --- | --- |
| `produceIMG.py` | Stratified outer folds, inner train/validation split, training-only UMAP coordinates, and fold images |
| `train.py` | Training CLI, fold metrics plots/JSON, and post-hoc IG-to-gene outputs |
| `StratifiedKFoldCrossValidation.py` | Training loop, validation-based checkpoint selection, and final fold test evaluation |
| `EfficientNetB1Classifier.py` | EfficientNet-B1 classifier |
| `GradCAM.py` | Validation Grad-CAM and guidance-mask generation |
| `sparse_integrated_gradients.py` | Post-hoc Integrated Gradients implementation |
| `FocalLoss.py` | Multi-class Focal Loss |
| `Dataset.py` | Image dataset loader |
| `ProcessHeatMapResult.py` | Heatmap aggregation and pixel-to-gene attribution mapping |
| `ChooseGenes.py` | Per-class, per-modality top positive/negative gene selection |
| `merge_genes_omics.py` | Gene merge and PAM50 overlap report |
| `filter_omics_by_unique_genes.py` | Filter source omics matrices by selected genes |
| `pathway_analysis.py` | Enrichr/MSigDB Hallmark enrichment |
| `go_kegg_enrichment.R` | GO and KEGG overrepresentation analysis |
| `additional_function.py` | Data loading, metrics, class weights, and plotting helpers |
| `run_pipeline.sh` | Environment-configured end-to-end runner |
| `plot_metrics_from_log.py` | Plot metrics from training logs |
| `architecture.png` | Framework overview figure |
