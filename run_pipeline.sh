#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
: "${DEEPCAM_DATA_DIR:?Set DEEPCAM_DATA_DIR to aligned CSV inputs}"
DEEPCAM_OUTPUT_DIR="${DEEPCAM_OUTPUT_DIR:-$ROOT_DIR/OutputData}"

# Background mode is optional. The child re-enters this script in foreground
# mode and sends both stdout and stderr to one persistent text log.
if [[ "${RUN_IN_BACKGROUND:-0}" == "1" && "${DEEPCAM_PIPELINE_BACKGROUND_CHILD:-0}" != "1" ]]; then
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
OMICS="${OMICS:-mRNA,Methylation,CNV}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
LOSS="${LOSS:-focal}"
GUIDED_LEARNING="${GUIDED_LEARNING:-1}"
PRETRAINED="${PRETRAINED:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_GENE_SELECTION="${RUN_GENE_SELECTION:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_FILTER="${RUN_FILTER:-1}"
RUN_HALLMARK="${RUN_HALLMARK:-1}"
RUN_GO_KEGG="${RUN_GO_KEGG:-1}"
EXPLANATION_DIR="${EXPLANATION_DIR:-$DEEPCAM_OUTPUT_DIR}"
SELECTED_GENES_DIR="${SELECTED_GENES_DIR:-$DEEPCAM_OUTPUT_DIR/selected_genes_$TOP_K}"
TOP_DIR="${TOP_DIR:-$DEEPCAM_OUTPUT_DIR/Top}"
GO_KEGG_DIR="${GO_KEGG_DIR:-$DEEPCAM_OUTPUT_DIR/GO_KEGG_Enrichment}"
mkdir -p "$DEEPCAM_OUTPUT_DIR"
export DEEPCAM_DATA_DIR DEEPCAM_OUTPUT_DIR OMICS K_FOLDS VAL_FRACTION

if [[ "$RUN_GENERATION" == 1 ]]; then
  printf '\n[1/7] Generate stratified folds and fold-specific image datasets\n'
  "$PYTHON" produceIMG.py --data-dir "$DEEPCAM_DATA_DIR" --output-dir "$DEEPCAM_OUTPUT_DIR" --omics "$OMICS" --k-folds "$K_FOLDS" --val-fraction "$VAL_FRACTION"
fi

if [[ "$RUN_TRAINING" == 1 ]]; then
  printf '\n[2/7] Train each fold and evaluate its untouched held-out test set\n'
  for ((fold=1; fold<=K_FOLDS; fold++)); do
    FOLD_DIR="$DEEPCAM_OUTPUT_DIR/Fold_$fold"
    args=(--dataset_path "$FOLD_DIR" --coord_base "$FOLD_DIR" --output_dir "$FOLD_DIR/Explanation_IG" --artifact_dir "$FOLD_DIR/Training" --results_plot "$FOLD_DIR/Training/learning_curves.png" --test_metrics_plot "$FOLD_DIR/Training/test_metrics.png" --test_results_json "$FOLD_DIR/Training/test_results.json" --k_folds 5 --num_epochs "$NUM_EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LEARNING_RATE" --dropout_rate "$DROPOUT_RATE" --focal_gamma "$FOCAL_GAMMA" --lambda1 "$LAMBDA1" --lambda2 "$LAMBDA2" --loss "$LOSS")
    [[ "$GUIDED_LEARNING" == 1 ]] || args+=(--no-guided_learning)
    [[ "$PRETRAINED" == 1 ]] || args+=(--no-pretrained)
    "$PYTHON" train.py "${args[@]}"
  done
fi

if [[ "$RUN_GENE_SELECTION" == 1 ]]; then
  printf '\n[3/7] Select top attributed genes across folds\n'
  "$PYTHON" ChooseGenes.py --base_dir "$EXPLANATION_DIR" --k "$TOP_K" --output_dir "$SELECTED_GENES_DIR"
fi
if [[ "$RUN_MERGE" == 1 ]]; then
  printf '\n[4/7] Merge selected genes and calculate PAM50 overlap\n'
  "$PYTHON" merge_genes_omics.py --folder "$SELECTED_GENES_DIR" --output_dir "$TOP_DIR"
fi
if [[ "$RUN_FILTER" == 1 ]]; then
  printf '\n[5/7] Filter input matrices to selected genes\n'
  "$PYTHON" filter_omics_by_unique_genes.py --unique_genes_file "$TOP_DIR/unique_genes_omics.csv" \
    --cnv_file "$DEEPCAM_DATA_DIR/BRCA_CNV_aligned.csv" --mrna_file "$DEEPCAM_DATA_DIR/BRCA_mRNA_aligned.csv" \
    --methy_file "$DEEPCAM_DATA_DIR/BRCA_Methy_aligned.csv" --output_dir "$TOP_DIR"
fi
if [[ "$RUN_HALLMARK" == 1 ]]; then
  printf '\n[6/7] Run pooled Hallmark enrichment\n'
  "$PYTHON" pathway_analysis.py --gene_file "$TOP_DIR/unique_genes_omics.csv" --gene_column unique_genes --gene_set MSigDB_Hallmark_2020 --output_file "$DEEPCAM_OUTPUT_DIR/hallmark_pathway_overlap.csv"
fi
if [[ "$RUN_GO_KEGG" == 1 ]]; then
  printf '\n[7/7] Run subtype-specific GO and KEGG enrichment\n'
  "$RSCRIPT" go_kegg_enrichment.R --mode per_subtype --gene_dir "$SELECTED_GENES_DIR" --k "$TOP_K" --gene_column gene_name --gene_set BP --pvalue_cutoff 0.05 --output_dir "$GO_KEGG_DIR"
fi
printf '\nPipeline finished. Results are under: %s\n' "$DEEPCAM_OUTPUT_DIR"
