# Results - 6 seeds × 5 folds (30 models)

Outer stratified K-fold (test), inner stratified train / validation split. Seeds: 12, 3, 44, 63, 16, 1.
Model: efficientnet_b1 (ImageNet-pretrained), head = gene readout of feature levels [2, 3], one trunk per omics; focal loss, class-aware augmentation; SGD (lr 0.01, 10-epoch warm-up, cosine to epoch 50); fold model = average of the weights after epochs 31-50 of that same run (SWA, no restart). Images: snowfall layout fitted on the training split, 0 reserved for empty pixels. Guidance: gradcam mask on the val split (top 22% wrong-reason gene pixels, frequency > 0.2; gene pixels only; one mask per class; top 50% genes by ANOVA never masked); penalty on ∇ₓ [log p(LumA|x) − log p(LumB|x)] for LumA/LumB samples, ∇ₓ log p(y|x) otherwise, λ = 1000, from epoch 19, on 4 random samples per batch.

## Test metrics, mean ± std over fold models

| metric | raw (argmax) | uncalibrated |
|---|---|---|
| Accuracy (%) | 85.23 ± 3.44 | 85.23 ± 3.44 |
| Precision (macro) (%) | 84.45 ± 3.45 | 84.45 ± 3.45 |
| Recall (macro) (%) | 85.05 ± 4.42 | 85.05 ± 4.42 |
| F1 (macro) (%) | 84.30 ± 3.69 | 84.30 ± 3.69 |
| MCC | 0.7765 ± 0.0520 | 0.7765 ± 0.0520 |
| ROC-AUC (OvR, macro) | 0.9664 ± 0.0113 | 0.9664 ± 0.0113 |
| PR-AUC (macro) | 0.9104 ± 0.0259 | 0.9104 ± 0.0259 |
| Sensitivity (macro) (%) | 85.05 ± 4.42 | 85.05 ± 4.42 |
| Specificity (macro) (%) | 94.42 ± 1.38 | 94.42 ± 1.38 |
| **additional** | | |
| Balanced accuracy (%) | 85.05 ± 4.42 | 85.05 ± 4.42 |
| F1 (weighted) (%) | 85.44 ± 3.37 | 85.44 ± 3.37 |
| Cohen's kappa | 0.7740 ± 0.0526 | 0.7740 ± 0.0526 |
| NLL | 0.4050 ± 0.0493 | 0.4050 ± 0.0493 |
| Brier | 0.2333 ± 0.0345 | 0.2333 ± 0.0345 |
| ECE | 0.1106 ± 0.0224 | 0.1106 ± 0.0224 |

## Pooled out-of-fold test metrics, mean ± std over seeds

| metric | raw (argmax) | uncalibrated |
|---|---|---|
| Accuracy (%) | 85.23 ± 0.87 | 85.23 ± 0.87 |
| Precision (macro) (%) | 83.87 ± 1.18 | 83.87 ± 1.18 |
| Recall (macro) (%) | 85.08 ± 1.20 | 85.08 ± 1.20 |
| F1 (macro) (%) | 84.35 ± 1.13 | 84.35 ± 1.13 |
| MCC | 0.7753 ± 0.0131 | 0.7753 ± 0.0131 |
| ROC-AUC (OvR, macro) | 0.9660 ± 0.0030 | 0.9660 ± 0.0030 |
| PR-AUC (macro) | 0.9043 ± 0.0090 | 0.9043 ± 0.0090 |
| Sensitivity (macro) (%) | 85.08 ± 1.20 | 85.08 ± 1.20 |
| Specificity (macro) (%) | 94.42 ± 0.34 | 94.42 ± 0.34 |
| **additional** | | |
| Balanced accuracy (%) | 85.08 ± 1.20 | 85.08 ± 1.20 |
| F1 (weighted) (%) | 85.46 ± 0.86 | 85.46 ± 0.86 |
| Cohen's kappa | 0.7740 ± 0.0132 | 0.7740 ± 0.0132 |
| NLL | 0.4050 ± 0.0112 | 0.4050 ± 0.0112 |
| Brier | 0.2333 ± 0.0080 | 0.2333 ± 0.0080 |
| ECE | 0.0981 ± 0.0044 | 0.0981 ± 0.0044 |

## Multi-seed ensemble (mean of each sample's out-of-fold probabilities over seeds)

| metric | raw | uncalibrated |
|---|---|---|
| Accuracy (%) | 87.45 | 87.45 |
| Precision (macro) (%) | 86.64 | 86.64 |
| Recall (macro) (%) | 87.94 | 87.94 |
| F1 (macro) (%) | 87.25 | 87.25 |
| MCC | 0.8073 | 0.8073 |
| ROC-AUC (OvR, macro) | 0.9753 | 0.9753 |
| PR-AUC (macro) | 0.9285 | 0.9285 |
| Sensitivity (macro) (%) | 87.94 | 87.94 |
| Specificity (macro) (%) | 95.14 | 95.14 |
| NLL | 0.3859 | 0.3859 |
| ECE | 0.1291 | 0.1291 |

## Per-seed pooled out-of-fold test (raw)

| seed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | MCC | ROC-AUC (OvR, macro) | PR-AUC (macro) | Sensitivity (macro) | Specificity (macro) |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 84.03 | 82.14 | 83.04 | 82.58 | 0.7537 | 0.9614 | 0.8910 | 83.04 | 93.78 |
| 3 | 85.42 | 84.50 | 85.53 | 84.88 | 0.7790 | 0.9664 | 0.9085 | 85.53 | 94.54 |
| 44 | 86.57 | 85.40 | 86.68 | 85.99 | 0.7933 | 0.9654 | 0.8994 | 86.68 | 94.75 |
| 63 | 84.92 | 82.96 | 85.28 | 83.87 | 0.7743 | 0.9701 | 0.9168 | 85.28 | 94.56 |
| 16 | 84.79 | 83.70 | 85.30 | 84.32 | 0.7707 | 0.9646 | 0.9014 | 85.30 | 94.38 |
| 1 | 85.68 | 84.51 | 84.66 | 84.49 | 0.7809 | 0.9683 | 0.9089 | 84.66 | 94.53 |

## Guidance mask check (final mask of each fold model, mean)

Share of each omics' features inside the mask, by ANOVA-F rank on the training split. A mask of wrong reasons should cover the top-ranked features much less than the rest.

| omics | top 50 | 51-200 | 201-1000 | >1000 | all |
|---|---|---|---|---|---|
| mRNA | 0% | 0% | 0% | 19% | 13% |
| Methylation | 0% | 0% | 0% | 25% | 17% |
| CNV | 0% | 0% | 0% | 33% | 22% |
