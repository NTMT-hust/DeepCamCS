"""Attribution-guided multi-omics image classification.

Stage  Module                         What it does
1      data, splits                   load + align the omics tables by sample ID; repeated stratified CV
2      imaging                        omics -> image (UMAP layout + snowfall, per-sample normalisation)
3      augmentation, models,          EfficientNet-B1 (ImageNet), focal loss, class-aware augmentation,
       losses, training               SWA fold model (epochs 31-50)
4      guidance (+ attribution)       Grad-CAM / IG mask of "wrong reasons" + input-gradient penalty (JVP)
5      calibration                    Beta / temperature calibration on the validation split
6      attribution, genes             IG gene attributions -> top-K genes, PAM50, stability; GO/KEGG in R
       experiment                     runs stages 2-6 for every seed x fold, resume, summary report
"""

__version__ = "3.0.0"
