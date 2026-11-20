# Results - 6 seeds × 5 folds (30 models)

Outer stratified K-fold (test), inner stratified train / validation split. Seeds: 12, 3, 44, 63, 16, 1.
Model: efficientnet_b1 (ImageNet-pretrained), one trunk per omics, gene readout of feature levels [2, 3]; focal loss, class-aware augmentation; SGD (lr 0.01, 10-epoch warm-up, cosine to epoch 50); fold model = average of the weights after epochs 31-50 of that same run (SWA). Images: snowfall layout fitted on the training split, 0 reserved for empty pixels. Raw softmax probabilities. Guidance: Grad-CAM mask on the validation split (top 22% wrong-reason gene pixels, frequency > 0.2; gene pixels only; one mask per class; top 50% genes by ANOVA never masked); penalty on ∇ₓ [log p(LumA|x) − log p(LumB|x)] for LumA/LumB samples, ∇ₓ log p(y|x) otherwise, λ = 1000, from epoch 19, on 4 random samples per batch.

## Test metrics

| metric | mean ± std over fold models | pooled out-of-fold, mean ± std over seeds |
|---|---|---|
| Accuracy (%) | 84.79 ± 3.77 | 84.79 ± 1.03 |
| Precision (macro) (%) | 84.57 ± 4.01 | 83.95 ± 0.81 |
| Recall (macro) (%) | 84.77 ± 4.18 | 84.79 ± 1.34 |
| F1 (macro) (%) | 84.25 ± 3.89 | 84.24 ± 0.98 |
| MCC | 0.7700 ± 0.0562 | 0.7685 ± 0.0155 |
| ROC-AUC (OvR, macro) | 0.9640 ± 0.0129 | 0.9635 ± 0.0048 |
| PR-AUC (macro) | 0.9030 ± 0.0316 | 0.8969 ± 0.0110 |
| Sensitivity (macro) (%) | 84.77 ± 4.18 | 84.79 ± 1.34 |
| Specificity (macro) (%) | 94.21 ± 1.47 | 94.21 ± 0.42 |
| **additional** | | |
| Balanced accuracy (%) | 84.77 ± 4.18 | 84.79 ± 1.34 |
| F1 (weighted) (%) | 85.02 ± 3.70 | 85.04 ± 1.00 |
| Cohen's kappa | 0.7673 ± 0.0572 | 0.7672 ± 0.0157 |
| NLL | 0.4182 ± 0.0500 | 0.4181 ± 0.0192 |
| Brier | 0.2415 ± 0.0352 | 0.2415 ± 0.0136 |
| ECE | 0.1152 ± 0.0253 | 0.1007 ± 0.0023 |

## Multi-seed ensemble (mean of each sample's out-of-fold probabilities over seeds)

| metric | value |
|---|---|
| Accuracy (%) | 87.20 |
| Precision (macro) (%) | 86.33 |
| Recall (macro) (%) | 87.26 |
| F1 (macro) (%) | 86.73 |
| MCC | 0.8044 |
| ROC-AUC (OvR, macro) | 0.9741 |
| PR-AUC (macro) | 0.9248 |
| Sensitivity (macro) (%) | 87.26 |
| Specificity (macro) (%) | 95.11 |
| NLL | 0.3982 |
| ECE | 0.1371 |

## Per-seed pooled out-of-fold test

| seed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | MCC | ROC-AUC (OvR, macro) | PR-AUC (macro) | Sensitivity (macro) | Specificity (macro) |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 85.55 | 83.88 | 86.19 | 84.92 | 0.7803 | 0.9631 | 0.8899 | 86.19 | 94.56 |
| 3 | 85.55 | 84.48 | 86.35 | 85.28 | 0.7809 | 0.9687 | 0.9129 | 86.35 | 94.57 |
| 44 | 85.93 | 85.33 | 85.22 | 85.17 | 0.7848 | 0.9682 | 0.9018 | 85.22 | 94.59 |
| 63 | 84.41 | 83.32 | 83.93 | 83.49 | 0.7629 | 0.9644 | 0.9011 | 83.93 | 94.13 |
| 16 | 83.40 | 83.27 | 84.03 | 83.43 | 0.7501 | 0.9607 | 0.8945 | 84.03 | 93.76 |
| 1 | 83.90 | 83.44 | 83.01 | 83.16 | 0.7521 | 0.9561 | 0.8811 | 83.01 | 93.68 |

## Guidance mask check (final mask of each fold model, mean)

Share of each omics' features inside the mask, by ANOVA-F rank on the training split. A mask of wrong reasons should cover the top-ranked features much less than the rest.

| omics | top 50 | 51-200 | 201-1000 | >1000 | all |
|---|---|---|---|---|---|
| mRNA | 0% | 0% | 0% | 21% | 14% |
| Methylation | 0% | 0% | 0% | 26% | 18% |
| CNV | 0% | 0% | 0% | 31% | 21% |
