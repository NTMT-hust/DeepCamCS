# Data folder

Put the input files here. The file names are set in `configs/brca.yaml` (`data:` section).

| file | content |
|---|---|
| `mRNA_filtered.csv` | mRNA features |
| `methylation_filtered.csv` | methylation probes |
| `CNA_filtered.csv` | copy-number segments |
| `labels.csv` | one row per sample: `patient_id`, `PAM50` |

Omics files: rows = features, columns = sample IDs, first column = feature ID. For example:

```
gene_id,TCGA-A1-A0SD,TCGA-A1-A0SE,...
ESR1|2099,1.23,-0.40,...
```

- Samples are matched to the labels by ID, not by position. Only samples present in every file are used.
- Missing values (NaN) are replaced by 0.
- Feature IDs of the form `SYMBOL|ENTREZ` are used directly by the PAM50 check and by GO / KEGG enrichment.

The pipeline does not rescale the tables: each image is normalised per sample when it is drawn.
