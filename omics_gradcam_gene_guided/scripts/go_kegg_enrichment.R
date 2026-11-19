#!/usr/bin/env Rscript
# =============================================================================
# GO / KEGG over-representation analysis per subtype
# (port of DeepCamCS go_kegg_enrichment.R, adapted to this pipeline)
#
# Input: the gene-selection folder written by phase 3 (gene_guided/phase3_choose_genes.py)
#   label_<subtype>_top_<K>_<positive|negative>.csv   selected features per subtype
#   background_features.csv                          every measured feature (gene universe)
#
# Differences from DeepCamCS:
#   * the universe (background) is the set of measured genes, not the whole genome.
#     Testing a list drawn from ~3,200 measured genes against the genome inflates
#     significance for any term enriched among breast-tissue genes in general.
#   * Entrez IDs are taken from "SYMBOL|ENTREZ" feature IDs when present (no loss
#     from symbol mapping); plain symbols are mapped with bitr().
#   * only gene-level omics are used (default mRNA): methylation probes (cg...) and
#     CNV segments (chr...) are not genes.
#
# Usage:
#   Rscript scripts/go_kegg_enrichment.R --dir runs/brca/gene_selection_top400
#   Rscript scripts/go_kegg_enrichment.R --dir <dir> --signs positive,negative --ont BP --no-kegg
#
# Options (all optional except --dir):
#   --dir DIR             gene-selection folder
#   --k K                 K in the file names (default: detected from the folder)
#   --signs LIST          positive | negative | positive,negative           (default positive)
#   --omics LIST          omics_type rows to use, comma separated            (default mRNA)
#   --ont ONT             GO ontology: BP | MF | CC                          (default BP)
#   --padj X              Benjamini-Hochberg adjusted p-value cutoff         (default 0.05)
#   --min-gs N            smallest gene set tested                           (default 10)
#   --max-gs N            largest gene set tested                            (default 500)
#   --top-n N             terms shown in the dot plots                       (default 15)
#   --no-kegg             skip KEGG. KEGG needs internet: clusterProfiler downloads the
#                         pathway annotation from the KEGG REST API (the gene list is not sent).
#                         GO uses the local org.Hs.eg.db annotation.
#   --out DIR             output folder                                      (default <dir>/go_kegg)
#   --install-missing     install missing CRAN / Bioconductor packages
#
# Outputs (in --out):
#   GO_<ONT>_<subtype>_<sign>.csv, KEGG_<subtype>_<sign>.csv     all tested terms, sorted by
#                                                                p.adjust, column `significant`
#   GO_<ONT>_<subtype>_<sign>_dotplot.png, KEGG_..._dotplot.png   significant terms only
#   go_kegg_enrichment_summary.csv                                one row per subtype x sign
# =============================================================================

VALUE_OPTIONS <- c("dir", "k", "signs", "omics", "ont", "padj", "min_gs", "max_gs", "top_n", "out")

parse_args <- function(argv) {
  opts <- list(dir = NULL, k = NA, signs = "positive", omics = "mRNA", ont = "BP", padj = 0.05,
               min_gs = 10, max_gs = 500, top_n = 15, out = NULL,
               kegg = TRUE, install_missing = FALSE)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    if (!startsWith(a, "--")) stop("Unexpected argument: ", a, call. = FALSE)
    a <- substring(a, 3L)
    if (grepl("=", a, fixed = TRUE)) {
      key <- sub("=.*$", "", a)
      val <- sub("^[^=]*=", "", a)
    } else {
      key <- a
      val <- NULL
    }
    key <- gsub("-", "_", key, fixed = TRUE)
    if (key == "no_kegg") {
      opts$kegg <- FALSE
    } else if (key == "install_missing") {
      opts$install_missing <- TRUE
    } else if (key %in% VALUE_OPTIONS) {
      if (is.null(val)) {
        if (i == length(argv)) stop("Missing value for --", gsub("_", "-", key), call. = FALSE)
        i <- i + 1L
        val <- argv[[i]]
      }
      opts[[key]] <- val
    } else {
      stop("Unknown option --", gsub("_", "-", key), call. = FALSE)
    }
    i <- i + 1L
  }
  if (is.null(opts$dir)) stop("--dir is required", call. = FALSE)
  if (!dir.exists(opts$dir)) stop("Folder not found: ", opts$dir, call. = FALSE)
  opts$k <- suppressWarnings(as.integer(opts$k))
  opts$padj <- as.numeric(opts$padj)
  opts$min_gs <- as.integer(opts$min_gs)
  opts$max_gs <- as.integer(opts$max_gs)
  opts$top_n <- as.integer(opts$top_n)
  opts$signs <- trimws(strsplit(as.character(opts$signs), ",", fixed = TRUE)[[1]])
  opts$omics <- trimws(strsplit(as.character(opts$omics), ",", fixed = TRUE)[[1]])
  opts$ont <- toupper(opts$ont)
  if (!all(opts$signs %in% c("positive", "negative"))) {
    stop("--signs must be positive, negative or positive,negative", call. = FALSE)
  }
  if (!(opts$ont %in% c("BP", "MF", "CC"))) stop("--ont must be BP, MF or CC", call. = FALSE)
  if (is.na(opts$padj) || opts$padj <= 0 || opts$padj > 1) stop("--padj must be in (0, 1]", call. = FALSE)
  if (is.null(opts$out)) opts$out <- file.path(opts$dir, "go_kegg")
  opts
}

ensure_packages <- function(install_missing) {
  pkgs <- c("clusterProfiler", "org.Hs.eg.db", "enrichplot", "ggplot2")
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) return(invisible(TRUE))
  if (!install_missing) {
    stop("Missing R packages: ", paste(missing, collapse = ", "), "\n",
         "Install them once with:\n",
         "  install.packages(\"BiocManager\")\n",
         "  BiocManager::install(c(", paste0("\"", missing, "\"", collapse = ", "), "))\n",
         "or re-run with --install-missing", call. = FALSE)
  }
  repos <- "https://cloud.r-project.org"
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = repos)
  BiocManager::install(missing, ask = FALSE, update = FALSE)
  still <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE)]
  if (length(still) > 0) stop("Could not install: ", paste(still, collapse = ", "), call. = FALSE)
  invisible(TRUE)
}

# "SYMBOL|ENTREZ" -> ENTREZ; plain symbols -> bitr(); anything else is dropped
features_to_entrez <- function(features) {
  f <- trimws(as.character(features))
  f <- f[!is.na(f) & f != ""]
  has_id <- grepl("\\|[0-9]+$", f)
  entrez <- rep(NA_character_, length(f))
  entrez[has_id] <- sub("^.*\\|", "", f[has_id])
  symbol <- toupper(trimws(sub("\\|.*$", "", f)))
  need <- !has_id & symbol != "" & symbol != "?"
  if (any(need)) {
    map <- tryCatch(
      suppressMessages(suppressWarnings(
        clusterProfiler::bitr(unique(symbol[need]), fromType = "SYMBOL", toType = "ENTREZID",
                              OrgDb = org.Hs.eg.db))),
      error = function(e) data.frame(SYMBOL = character(0), ENTREZID = character(0)))
    map <- map[!duplicated(map$SYMBOL), , drop = FALSE]
    entrez[need] <- map$ENTREZID[match(symbol[need], map$SYMBOL)]
  }
  list(entrez = unique(entrez[!is.na(entrez)]), n_input = length(f), n_mapped = sum(!is.na(entrez)))
}

run_go <- function(genes, universe, opts) {
  args <- list(gene = genes, OrgDb = org.Hs.eg.db, keyType = "ENTREZID", ont = opts$ont,
               pAdjustMethod = "BH", pvalueCutoff = 1, qvalueCutoff = 1,
               minGSSize = opts$min_gs, maxGSSize = opts$max_gs, readable = TRUE)
  if (!is.null(universe)) args$universe <- universe
  tryCatch(do.call(clusterProfiler::enrichGO, args),
           error = function(e) {
             message("  [warn] enrichGO failed: ", conditionMessage(e))
             NULL
           })
}

run_kegg <- function(genes, universe, opts) {
  args <- list(gene = genes, organism = "hsa", keyType = "kegg", pAdjustMethod = "BH",
               pvalueCutoff = 1, qvalueCutoff = 1, minGSSize = opts$min_gs, maxGSSize = opts$max_gs)
  if (!is.null(universe)) args$universe <- universe
  res <- tryCatch(do.call(clusterProfiler::enrichKEGG, args),
                  error = function(e) {
                    message("  [warn] enrichKEGG failed (KEGG needs internet access): ", conditionMessage(e))
                    NULL
                  })
  if (!is.null(res) && nrow(as.data.frame(res)) > 0) {
    res <- tryCatch(clusterProfiler::setReadable(res, OrgDb = org.Hs.eg.db, keyType = "ENTREZID"),
                    error = function(e) res)
  }
  res
}

result_table <- function(res, padj) {
  if (is.null(res)) return(NULL)
  df <- as.data.frame(res)
  if (nrow(df) == 0) return(df)
  df <- df[order(df$p.adjust, df$pvalue), , drop = FALSE]
  df$significant <- df$p.adjust <= padj
  df
}

save_dotplot <- function(res, n_sig, title, path, top_n) {
  if (is.null(res) || n_sig == 0) return(invisible(FALSE))
  n_show <- min(top_n, n_sig)
  ok <- tryCatch({
    p <- enrichplot::dotplot(res, showCategory = n_show, title = title)
    ggplot2::ggsave(path, p, width = 8, height = 2 + 0.32 * n_show, dpi = 150, limitsize = FALSE)
    TRUE
  }, error = function(e) {
    message("  [warn] dot plot failed: ", conditionMessage(e))
    FALSE
  })
  invisible(ok)
}

n_rows <- function(df) if (is.null(df)) NA_integer_ else nrow(df)
n_significant <- function(df) if (is.null(df) || nrow(df) == 0) 0L else sum(df$significant)
top_term <- function(df) if (is.null(df) || nrow(df) == 0) NA_character_ else df$Description[1]
top_padj <- function(df) if (is.null(df) || nrow(df) == 0) NA_real_ else df$p.adjust[1]

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  ensure_packages(opts$install_missing)
  suppressPackageStartupMessages({
    library(clusterProfiler)
    library(org.Hs.eg.db)
    library(enrichplot)
    library(ggplot2)
  })
  dir.create(opts$out, recursive = TRUE, showWarnings = FALSE)

  # ---- selected-feature files ---------------------------------------------------------
  pattern <- "^label_(.+)_top_([0-9]+)_(positive|negative)\\.csv$"
  files <- list.files(opts$dir, pattern = pattern)
  if (length(files) == 0) {
    stop("No label_<subtype>_top_<K>_<sign>.csv files in ", opts$dir,
         " (run the genes step first: python run_pipeline.py --steps genes)", call. = FALSE)
  }
  info <- data.frame(file = files, subtype = sub(pattern, "\\1", files),
                     k = as.integer(sub(pattern, "\\2", files)), sign = sub(pattern, "\\3", files),
                     stringsAsFactors = FALSE)
  if (is.na(opts$k)) {
    ks <- unique(info$k)
    if (length(ks) > 1) stop("Several K values found (", paste(ks, collapse = ", "), "); pass --k", call. = FALSE)
    opts$k <- ks
  }
  info <- info[info$k == opts$k & info$sign %in% opts$signs, , drop = FALSE]
  if (nrow(info) == 0) stop("No files for K = ", opts$k, " and signs ", paste(opts$signs, collapse = ","),
                            call. = FALSE)
  info <- info[order(info$subtype, info$sign), , drop = FALSE]

  # ---- universe: every measured gene of the selected omics ------------------------------
  bg_path <- file.path(opts$dir, "background_features.csv")
  universe <- NULL
  if (file.exists(bg_path)) {
    bg <- read.csv(bg_path, stringsAsFactors = FALSE, check.names = FALSE)
    bg <- bg[bg$omics_type %in% opts$omics, , drop = FALSE]
    universe <- features_to_entrez(bg$gene_name)$entrez
    if (length(universe) == 0) {
      stop("No gene IDs could be derived from the ", paste(opts$omics, collapse = "+"),
           " background features (are these gene-level omics?)", call. = FALSE)
    }
    message(sprintf("Universe: %d Entrez genes from %d measured %s features", length(universe), nrow(bg),
                    paste(opts$omics, collapse = "+")))
  } else {
    message("[warn] ", bg_path, " not found: testing against all annotated genes, which overstates ",
            "significance. Re-run the genes step (python run_pipeline.py --steps genes) to create it.")
  }
  message(sprintf("GO %s%s | BH-adjusted p <= %g | gene-set size %d-%d | K = %d | signs: %s",
                  opts$ont, if (opts$kegg) " + KEGG" else "", opts$padj, opts$min_gs, opts$max_gs,
                  opts$k, paste(opts$signs, collapse = ",")))

  # ---- per subtype x sign ----------------------------------------------------------------
  rows <- list()
  for (r in seq_len(nrow(info))) {
    subtype <- info$subtype[r]
    sgn <- info$sign[r]
    tag <- paste0(subtype, "_", sgn)
    sel <- read.csv(file.path(opts$dir, info$file[r]), stringsAsFactors = FALSE, check.names = FALSE)
    sel <- sel[sel$omics_type %in% opts$omics, , drop = FALSE]
    ids <- features_to_entrez(sel$gene_name)
    genes <- ids$entrez
    if (!is.null(universe)) genes <- intersect(genes, universe)
    message(sprintf("[%s | %s] %d features -> %d Entrez genes", subtype, sgn, nrow(sel), length(genes)))

    go <- if (length(genes) > 0) run_go(genes, universe, opts) else NULL
    go_df <- result_table(go, opts$padj)
    if (!is.null(go_df)) {
      write.csv(go_df, file.path(opts$out, sprintf("GO_%s_%s.csv", opts$ont, tag)), row.names = FALSE)
    }
    save_dotplot(go, n_significant(go_df), sprintf("GO %s - %s (%s)", opts$ont, subtype, sgn),
                 file.path(opts$out, sprintf("GO_%s_%s_dotplot.png", opts$ont, tag)), opts$top_n)

    kegg_df <- NULL
    if (opts$kegg && length(genes) > 0) {
      kegg <- run_kegg(genes, universe, opts)
      kegg_df <- result_table(kegg, opts$padj)
      if (!is.null(kegg_df)) {
        write.csv(kegg_df, file.path(opts$out, sprintf("KEGG_%s.csv", tag)), row.names = FALSE)
      }
      save_dotplot(kegg, n_significant(kegg_df), sprintf("KEGG - %s (%s)", subtype, sgn),
                   file.path(opts$out, sprintf("KEGG_%s_dotplot.png", tag)), opts$top_n)
    }
    message(sprintf("    GO: %d significant of %s tested%s", n_significant(go_df), n_rows(go_df),
                    if (opts$kegg) sprintf(" | KEGG: %d significant of %s tested",
                                           n_significant(kegg_df), n_rows(kegg_df)) else ""))

    rows[[length(rows) + 1L]] <- data.frame(
      subtype = subtype, sign = sgn, n_features = nrow(sel), n_mapped_genes = length(genes),
      universe_size = if (is.null(universe)) NA_integer_ else length(universe),
      GO_ontology = opts$ont, GO_tested = n_rows(go_df), GO_significant = n_significant(go_df),
      GO_top_term = top_term(go_df), GO_top_padj = top_padj(go_df),
      KEGG_tested = if (opts$kegg) n_rows(kegg_df) else NA_integer_,
      KEGG_significant = if (opts$kegg) n_significant(kegg_df) else NA_integer_,
      KEGG_top_term = if (opts$kegg) top_term(kegg_df) else NA_character_,
      KEGG_top_padj = if (opts$kegg) top_padj(kegg_df) else NA_real_,
      stringsAsFactors = FALSE)
  }
  summary <- do.call(rbind, rows)
  out_csv <- file.path(opts$out, "go_kegg_enrichment_summary.csv")
  write.csv(summary, out_csv, row.names = FALSE)
  print(summary[, c("subtype", "sign", "n_mapped_genes", "GO_significant", "KEGG_significant")], row.names = FALSE)
  message("Saved results to ", normalizePath(opts$out))
}

main()
