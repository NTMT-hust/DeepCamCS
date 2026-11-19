# Data

`config.yaml` reads the data from `../omics_gradcam_pipeline/data` (`data.data_dir`). To use this folder
instead, copy the files here and set `data_dir: data`.

| File | Content |
|---|---|
| `mRNA_filtered.csv` | features × samples; feature IDs `SYMBOL|ENTREZ` (e.g. `ESR1|2099`) |
| `methylation_filtered.csv` | features × samples; CpG probe IDs (`cg...`) |
| `CNA_filtered.csv` | features × samples; CNV segment IDs (`chr...`) |
| `labels.csv` | `patient_id`, `PAM50` |

The first column of every omics file is the feature ID, the other columns are sample IDs. Only the samples in
the labels file and in every omics file are used.
