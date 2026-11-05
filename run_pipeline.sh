#!/usr/bin/env bash
set -Eeuo pipefail

# Run the complete DeepCamCS pipeline from aligned matrices through enrichment.
# Example (Git Bash):
#   export DEEPCAM_DATA_DIR="/c/data/brca-aligned"
#   export DEEPCAM_OUTPUT_DIR="/c/results/DeepCamCS"
#   bash run_pipeline.sh
# Run in the current terminal instead of the background:
#   RUN_IN_BACKGROUND=0 bash run_pipeline.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${DEEPCAM_DATA_DIR:?Set DEEPCAM_DATA_DIR to the folder containing the aligned BRCA CSV files}"
DEEPCAM_OUTPUT_DIR="${DEEPCAM_OUTPUT_DIR:-$ROOT_DIR/OutputData}"

if [[ "${RUN_IN_BACKGROUND:-1}" == "1" && "${DEEPCAM_PIPELINE_BACKGROUND_CHILD:-0}" != "1" ]]; then
    if ! command -v nohup >/dev/null 2>&1; then
        printf 'Required command not found: nohup\n' >&2
        exit 1
    fi

    mkdir -p "$DEEPCAM_OUTPUT_DIR"
    LOG_FILE="${PIPELINE_LOG_FILE:-$DEEPCAM_OUTPUT_DIR/pipeline_$(date +%Y%m%d_%H%M%S).txt}"
    nohup env DEEPCAM_PIPELINE_BACKGROUND_CHILD=1 bash "$ROOT_DIR/run_pipeline.sh" "$@" \
        >"$LOG_FILE" 2>&1 </dev/null &
    PIPELINE_PID=$!

    printf 'Pipeline started in background.\nPID: %s\nLog: %s\n' "$PIPELINE_PID" "$LOG_FILE"
    exit 0
fi

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"

TOP_K="${TOP_K:-100}"
K_FOLDS="${K_FOLDS:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
DROPOUT_RATE="${DROPOUT_RATE:-0.5}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
LAMBDA1="${LAMBDA1:-10}"
LAMBDA2="${LAMBDA2:-0.001}"

EXPLANATION_DIR="${EXPLANATION_DIR:-$DEEPCAM_OUTPUT_DIR/Explanation_IG}"
SELECTED_GENES_DIR="${SELECTED_GENES_DIR:-$DEEPCAM_OUTPUT_DIR/selected_genes_$TOP_K}"
TOP_DIR="${TOP_DIR:-$DEEPCAM_OUTPUT_DIR/Top}"
GO_KEGG_DIR="${GO_KEGG_DIR:-$DEEPCAM_OUTPUT_DIR/GO_KEGG_Enrichment}"
HALLMARK_RESULTS="${HALLMARK_RESULTS:-$DEEPCAM_OUTPUT_DIR/hallmark_pathway_overlap.csv}"

require_match() {
    local pattern="$1"
    if ! compgen -G "$pattern" >/dev/null; then
        printf 'Expected pipeline output was not created: %s\n' "$pattern" >&2
        exit 1
    fi
}

for executable in "$PYTHON" "$RSCRIPT"; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$executable" >&2
        exit 1
    fi
done

required_inputs=(
    "$DEEPCAM_DATA_DIR/BRCA_mRNA_aligned.csv"
    "$DEEPCAM_DATA_DIR/BRCA_Methy_aligned.csv"
    "$DEEPCAM_DATA_DIR/BRCA_CNV_aligned.csv"
    "$DEEPCAM_DATA_DIR/BRCA_label_num.csv"
)
for input_file in "${required_inputs[@]}"; do
    if [[ ! -f "$input_file" ]]; then
        printf 'Required input file not found: %s\n' "$input_file" >&2
        exit 1
    fi
done

mkdir -p "$DEEPCAM_OUTPUT_DIR" "$EXPLANATION_DIR" "$SELECTED_GENES_DIR" "$TOP_DIR" "$GO_KEGG_DIR"
export DEEPCAM_DATA_DIR DEEPCAM_OUTPUT_DIR

printf '\n[1/7] Transform omics matrices into images and gene coordinates\n'
"$PYTHON" produceIMG.py
require_match "$DEEPCAM_OUTPUT_DIR/dataset/*/*.png"
for coordinate_file in gene_coordinates_mRNA.csv gene_coordinates_Methylation.csv gene_coordinates_CNV.csv; do
    if [[ ! -s "$DEEPCAM_OUTPUT_DIR/$coordinate_file" ]]; then
        printf 'Expected coordinate file was not created: %s\n' "$DEEPCAM_OUTPUT_DIR/$coordinate_file" >&2
        exit 1
    fi
done

printf '\n[2/7] Train and evaluate EfficientNet; calculate validation-fold IG scores\n'
"$PYTHON" train.py \
    --dataset_path "$DEEPCAM_OUTPUT_DIR/dataset" \
    --coord_base "$DEEPCAM_OUTPUT_DIR" \
    --output_dir "$EXPLANATION_DIR" \
    --results_plot "$DEEPCAM_OUTPUT_DIR/imbalanced_kfold_results.png" \
    --test_metrics_plot "$DEEPCAM_OUTPUT_DIR/test_metrics_by_fold.png" \
    --test_results_json "$DEEPCAM_OUTPUT_DIR/fold_test_results.json" \
    --k_folds "$K_FOLDS" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --dropout_rate "$DROPOUT_RATE" \
    --focal_gamma "$FOCAL_GAMMA" \
    --lambda1 "$LAMBDA1" \
    --lambda2 "$LAMBDA2"
require_match "$EXPLANATION_DIR/predicted_label_*_attribute_scores.csv"

printf '\n[3/7] Select top positive and negative attributed genes\n'
"$PYTHON" ChooseGenes.py \
    --base_dir "$EXPLANATION_DIR" \
    --k "$TOP_K" \
    --output_dir "$SELECTED_GENES_DIR"
require_match "$SELECTED_GENES_DIR/predicted_label_*_top_${TOP_K}_positive.csv"

printf '\n[4/7] Merge selected genes and report PAM50 overlap\n'
"$PYTHON" merge_genes_omics.py \
    --folder "$SELECTED_GENES_DIR" \
    --output_dir "$TOP_DIR"
if [[ ! -s "$TOP_DIR/unique_genes_omics.csv" ]]; then
    printf 'Merged unique gene file was not created: %s\n' "$TOP_DIR/unique_genes_omics.csv" >&2
    exit 1
fi

printf '\n[5/7] Filter the original omics matrices to selected genes\n'
"$PYTHON" filter_omics_by_unique_genes.py \
    --unique_genes_file "$TOP_DIR/unique_genes_omics.csv" \
    --cnv_file "$DEEPCAM_DATA_DIR/BRCA_CNV_aligned.csv" \
    --mrna_file "$DEEPCAM_DATA_DIR/BRCA_mRNA_aligned.csv" \
    --methy_file "$DEEPCAM_DATA_DIR/BRCA_Methy_aligned.csv" \
    --output_dir "$TOP_DIR"
for filtered_file in \
    BRCA_CNV_aligned_filtered.csv \
    BRCA_mRNA_aligned_filtered.csv \
    BRCA_Methy_aligned_filtered.csv; do
    if [[ ! -s "$TOP_DIR/$filtered_file" ]]; then
        printf 'Expected filtered matrix was not created: %s\n' "$TOP_DIR/$filtered_file" >&2
        exit 1
    fi
done

printf '\n[6/7] Run pooled MSigDB Hallmark enrichment\n'
"$PYTHON" pathway_analysis.py \
    --gene_file "$TOP_DIR/unique_genes_omics.csv" \
    --gene_column unique_genes \
    --gene_set MSigDB_Hallmark_2020 \
    --output_file "$HALLMARK_RESULTS"
if [[ ! -s "$HALLMARK_RESULTS" ]]; then
    printf 'Hallmark enrichment output was not created: %s\n' "$HALLMARK_RESULTS" >&2
    exit 1
fi

printf '\n[7/7] Run subtype-specific GO Biological Process and KEGG enrichment\n'
"$RSCRIPT" go_kegg_enrichment.R \
    --mode per_subtype \
    --gene_dir "$SELECTED_GENES_DIR" \
    --k "$TOP_K" \
    --gene_column gene_name \
    --gene_set BP \
    --pvalue_cutoff 0.05 \
    --output_dir "$GO_KEGG_DIR"
if [[ ! -s "$GO_KEGG_DIR/go_kegg_enrichment_summary.csv" ]]; then
    printf 'GO/KEGG summary was not created: %s\n' "$GO_KEGG_DIR/go_kegg_enrichment_summary.csv" >&2
    exit 1
fi

printf '\nPipeline finished. Results are under: %s\n' "$DEEPCAM_OUTPUT_DIR"
