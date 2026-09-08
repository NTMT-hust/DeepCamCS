#!/usr/bin/env Rscript
# =============================================================================
# go_kegg_enrichment.R  --  Step 7: GO & KEGG Functional Enrichment
#
# Complements Step 6 (pathway_analysis.py, Enrichr/MSigDB Hallmark on the
# single globally-merged gene list from merge_genes_omics.py) with the
# functional enrichment method DeepCamCS's paper actually describes: GO
# Biological Process + KEGG pathway overrepresentation via the
# clusterProfiler R package (Yu et al., 2012), Benjamini-Hochberg FDR
# correction, significance at adjusted P-value <= 0.05.
#
# Unlike Step 6, this runs PER SUBTYPE (not on one pooled gene list across
# all classes) -- consuming Step 3's output directly:
#   selected_genes_400/predicted_label_{cls}_top_{k}_positive.csv
#   columns: predicted_label, omics_type, gene_name, attribute_score
# This matches how the paper reports enrichment (HER2, Luminal B, and
# Normal-like each show DIFFERENT significant GO/KEGG terms) -- pooling
# every subtype's genes together before enrichment, as Step 6 does, would
# wash out that subtype-specific signal.
#
# Two modes:
#   --mode per_subtype (default): loop over every
#     predicted_label_*_top_{k}_positive.csv in --gene_dir, one enrichment
#     run per subtype.
#   --mode single_file: run once on one CSV (e.g. Step 4's
#     Top/unique_genes_omics.csv) for parity with pathway_analysis.py's
#     interface, if you want a single pooled-gene-list run instead/as well.
#
# Usage (mirrors this repo's Python CLI style):
#   Rscript go_kegg_enrichment.R \
#       --gene_dir selected_genes_400 \
#       --k 400 \
#       --gene_column gene_name \
#       --output_dir GO_KEGG_Enrichment
#
#   Rscript go_kegg_enrichment.R --mode single_file \
#       --gene_file Top/unique_genes_omics.csv \
#       --gene_column unique_genes \
#       --output_dir GO_KEGG_Enrichment
#
# Note on "two-sided tests": clusterProfiler's enrichGO/enrichKEGG implement
# a one-sided hypergeometric overrepresentation test -- the only test these
# functions expose, and standard practice for this kind of analysis. This is
# used as-is rather than reinterpreted.
# =============================================================================

# ---------------------------------------------------------------------- #
# 0. Setup                                                                #
# ---------------------------------------------------------------------- #
required_bioc <- c("clusterProfiler", "org.Hs.eg.db", "enrichplot")
required_cran <- c("dplyr", "ggplot2", "optparse")

installed <- rownames(installed.packages())
missing_bioc <- setdiff(required_bioc, installed)
missing_cran <- setdiff(required_cran, installed)

if (length(missing_cran) > 0) install.packages(missing_cran, repos = "https://cloud.r-project.org")
if (length(missing_bioc) > 0) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install(missing_bioc, update = FALSE, ask = FALSE)
}

suppressPackageStartupMessages({
  library(optparse)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(enrichplot)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------- #
# 1. CLI arguments (mirrors this repo's argparse-style scripts)           #
# ---------------------------------------------------------------------- #
option_list <- list(
  make_option("--mode", type = "character", default = "per_subtype",
              help = "'per_subtype' (default) or 'single_file' [default %default]"),
  make_option("--gene_dir", type = "character", default = "selected_genes_400",
              help = "Directory of Step 3's per-subtype gene CSVs [default %default]"),
  make_option("--k", type = "integer", default = 400,
              help = "K used in ChooseGenes.py's output filenames [default %default]"),
  make_option("--gene_file", type = "character", default = "Top/unique_genes_omics.csv",
              help = "Single gene CSV, used only when --mode single_file [default %default]"),
  make_option("--gene_column", type = "character", default = "gene_name",
              help = "Column holding gene symbols [default %default]"),
  make_option("--gene_set", type = "character", default = "BP",
              help = "GO ontology: BP, MF, or CC [default %default]"),
  make_option("--organism", type = "character", default = "hsa",
              help = "KEGG organism code [default %default]"),
  make_option("--pvalue_cutoff", type = "double", default = 0.05,
              help = "Adjusted P-value significance threshold [default %default]"),
  make_option("--output_dir", type = "character", default = "GO_KEGG_Enrichment",
              help = "Directory to save enrichment results [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

ORG_DB          <- org.Hs.eg.db
KEGG_ORGANISM   <- opt$organism
GO_ONTOLOGY     <- opt$gene_set
P_ADJUST_METHOD <- "BH"
SIGNIFICANCE    <- opt$pvalue_cutoff

`%+%` <- function(a, b) paste0(a, b)

# ---------------------------------------------------------------------- #
# 2. Gene symbol -> Entrez ID mapping                                     #
# ---------------------------------------------------------------------- #
map_to_entrez <- function(gene_symbols) {
  gene_symbols <- unique(toupper(trimws(as.character(gene_symbols))))
  gene_symbols <- gene_symbols[gene_symbols != "" & !is.na(gene_symbols)]
  mapping <- suppressWarnings(
    bitr(gene_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = ORG_DB)
  )
  unmapped <- setdiff(gene_symbols, mapping$SYMBOL)
  if (length(unmapped) > 0) {
    message(sprintf("  [note] %d/%d gene symbols could not be mapped to Entrez IDs "
                     %+% "and were dropped: %s",
                     length(unmapped), length(gene_symbols),
                     paste(head(unmapped, 10), collapse = ", ")))
  }
  mapping
}

# ---------------------------------------------------------------------- #
# 3. GO / KEGG overrepresentation                                         #
# ---------------------------------------------------------------------- #
run_go <- function(entrez_ids) {
  ego <- enrichGO(
    gene = entrez_ids, OrgDb = ORG_DB, ont = GO_ONTOLOGY,
    pAdjustMethod = P_ADJUST_METHOD, pvalueCutoff = 1, qvalueCutoff = 1, readable = TRUE
  )
  if (is.null(ego) || nrow(as.data.frame(ego)) == 0) return(list(full = ego, sig = data.frame()))
  df <- as.data.frame(ego)
  list(full = ego, sig = df %>% filter(p.adjust <= SIGNIFICANCE) %>% arrange(p.adjust))
}

run_kegg <- function(entrez_ids) {
  ekegg <- tryCatch(
    enrichKEGG(gene = entrez_ids, organism = KEGG_ORGANISM,
               pAdjustMethod = P_ADJUST_METHOD, pvalueCutoff = 1, qvalueCutoff = 1),
    error = function(e) {
      message("  [warn] enrichKEGG failed (often a transient KEGG API issue): ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(ekegg) || nrow(as.data.frame(ekegg)) == 0) return(list(full = ekegg, sig = data.frame()))
  df <- as.data.frame(ekegg)
  list(full = ekegg, sig = df %>% filter(p.adjust <= SIGNIFICANCE) %>% arrange(p.adjust))
}

# ---------------------------------------------------------------------- #
# 4. Per-gene-set orchestration (one subtype, or the single pooled file)  #
# ---------------------------------------------------------------------- #
run_enrichment_for_geneset <- function(gene_symbols, label, out_dir) {
  message(sprintf("\n=== %s ===", label))
  message(sprintf("  %d input genes", length(unique(gene_symbols))))

  mapping <- map_to_entrez(gene_symbols)
  if (nrow(mapping) == 0) {
    message("  [skip] no genes mapped to Entrez IDs.")
    return(data.frame(label = label, n_input_genes = length(unique(gene_symbols)),
                       n_mapped = 0, n_go_sig = 0, n_kegg_sig = 0))
  }
  entrez_ids <- mapping$ENTREZID

  go_res <- run_go(entrez_ids)
  kegg_res <- run_kegg(entrez_ids)

  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  write.csv(go_res$sig, file.path(out_dir, sprintf("GO_%s_%s.csv", GO_ONTOLOGY, label)), row.names = FALSE)
  write.csv(kegg_res$sig, file.path(out_dir, sprintf("KEGG_%s.csv", label)), row.names = FALSE)

  if (nrow(go_res$sig) > 0) {
    p <- dotplot(go_res$full, showCategory = min(15, nrow(go_res$sig))) +
      ggtitle(sprintf("GO %s — %s", GO_ONTOLOGY, label))
    ggsave(file.path(out_dir, sprintf("GO_%s_%s_dotplot.png", GO_ONTOLOGY, label)),
           p, width = 8, height = 6, dpi = 200)
  }
  if (!is.null(kegg_res$full) && nrow(kegg_res$sig) > 0) {
    p <- dotplot(kegg_res$full, showCategory = min(15, nrow(kegg_res$sig))) +
      ggtitle(sprintf("KEGG — %s", label))
    ggsave(file.path(out_dir, sprintf("KEGG_%s_dotplot.png", label)),
           p, width = 8, height = 6, dpi = 200)
  }

  message(sprintf("  GO %s significant terms (p.adj<=%.2f): %d | KEGG significant pathways: %d",
                   GO_ONTOLOGY, SIGNIFICANCE, nrow(go_res$sig), nrow(kegg_res$sig)))

  data.frame(
    label = label,
    n_input_genes = length(unique(gene_symbols)),
    n_mapped = length(entrez_ids),
    n_go_sig = nrow(go_res$sig),
    n_kegg_sig = nrow(kegg_res$sig),
    top_go_terms = paste(head(go_res$sig$Description, 5), collapse = "; "),
    top_kegg_terms = paste(head(kegg_res$sig$Description, 5), collapse = "; ")
  )
}

# ---------------------------------------------------------------------- #
# 5a. per_subtype mode: one run per predicted_label_*_top_{k}_positive.csv #
# ---------------------------------------------------------------------- #
run_per_subtype <- function(gene_dir, k, gene_column, out_dir) {
  pattern <- sprintf("^predicted_label_.*_top_%d_positive\\.csv$", k)
  files <- list.files(gene_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) {
    stop(sprintf("No files matching '%s' found in %s. Run ChooseGenes.py first "
                  %+% "(check --k matches what you used there).", pattern, gene_dir))
  }
  subtype_labels <- sub(sprintf("^predicted_label_(.*)_top_%d_positive\\.csv$", k), "\\1", basename(files))

  summary_rows <- lapply(seq_along(files), function(i) {
    df <- read.csv(files[i], stringsAsFactors = FALSE)
    if (!(gene_column %in% names(df))) {
      stop(sprintf("Column '%s' not found in %s. Available columns: %s",
                    gene_column, files[i], paste(names(df), collapse = ", ")))
    }
    run_enrichment_for_geneset(df[[gene_column]], subtype_labels[i], out_dir)
  })
  bind_rows(summary_rows)
}

# ---------------------------------------------------------------------- #
# 5b. single_file mode: one run on one pooled gene CSV                    #
# ---------------------------------------------------------------------- #
run_single_file <- function(gene_file, gene_column, out_dir) {
  if (!file.exists(gene_file)) stop(sprintf("Gene file not found: '%s'", gene_file))
  df <- read.csv(gene_file, stringsAsFactors = FALSE)
  if (!(gene_column %in% names(df))) {
    message(sprintf("  [note] column '%s' not found; falling back to first column '%s'",
                     gene_column, names(df)[1]))
    gene_column <- names(df)[1]
  }
  label <- tools::file_path_sans_ext(basename(gene_file))
  run_enrichment_for_geneset(df[[gene_column]], label, out_dir)
}

# ---------------------------------------------------------------------- #
# Entry point                                                              #
# ---------------------------------------------------------------------- #
summary_df <- if (opt$mode == "single_file") {
  run_single_file(opt$gene_file, opt$gene_column, opt$output_dir)
} else {
  run_per_subtype(opt$gene_dir, opt$k, opt$gene_column, opt$output_dir)
}

write.csv(summary_df, file.path(opt$output_dir, "go_kegg_enrichment_summary.csv"), row.names = FALSE)

message("\n" %+% strrep("=", 60))
message("GO/KEGG FUNCTIONAL ENRICHMENT SUMMARY")
message(strrep("=", 60))
print(summary_df[, c("label", "n_input_genes", "n_mapped", "n_go_sig", "n_kegg_sig")])
message(sprintf("\n\U0001F389 Completed GO/KEGG enrichment. Outputs saved to '%s'.", opt$output_dir))
