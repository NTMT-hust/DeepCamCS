"""Create stratified fold datasets using a training-only UMAP gene layout."""
import argparse
import os
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import umap
from scipy.ndimage import zoom
from shapely.geometry import MultiPoint
from sklearn.model_selection import StratifiedKFold, train_test_split

FILES = {
    "mRNA": "BRCA_mRNA_aligned.csv",
    "Methylation": "BRCA_Methy_aligned.csv",
    "CNV": "BRCA_CNV_aligned.csv",
}
CHANNEL = {"mRNA": 0, "Methylation": 1, "CNV": 2}


def snowfall(xp, yp, size):
    occupied = set()
    outx, outy = [], []
    for x, y in zip(xp, yp):
        x, y = int(np.clip(x, 1, size)), int(np.clip(y, 1, size))
        if (x, y) in occupied:
            queue, seen = deque([(x, y)]), {(x, y)}
            while queue:
                cx, cy = queue.popleft()
                if (cx, cy) not in occupied:
                    x, y = cx, cy
                    break
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    point = (cx + dx, cy + dy)
                    if 1 <= point[0] <= size and 1 <= point[1] <= size and point not in seen:
                        seen.add(point)
                        queue.append(point)
        occupied.add((x, y))
        outx.append(x)
        outy.append(y)
    return np.asarray(outx), np.asarray(outy)


def fit_gene_layout(train_gene_profiles, size, seed):
    """Fit gene coordinates using training samples only; coordinates are reused for val/test."""
    if train_gene_profiles.shape[0] < 3:
        raise ValueError("UMAP requires at least three genes/features")
    n_neighbors = max(2, min(30, train_gene_profiles.shape[0] - 1))
    embedding = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
                          metric="cosine", init="spectral", random_state=seed).fit_transform(train_gene_profiles)
    x, y = embedding[:, 0], embedding[:, 1]
    rect = np.asarray(MultiPoint(np.column_stack((x, y))).convex_hull.minimum_rotated_rectangle.exterior.coords)
    theta = np.arctan2(rect[1, 1] - rect[0, 1], rect[1, 0] - rect[0, 0])
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ np.vstack((x, y))
    spanx, spany = np.ptp(rot[0]), np.ptp(rot[1])
    xp = np.ones(len(x), dtype=int) if spanx == 0 else np.rint(1 + (size - 1) * (rot[0] - rot[0].min()) / spanx).astype(int)
    yp = np.ones(len(y), dtype=int) if spany == 0 else np.rint(1 + (size - 1) * (rot[1] - rot[1].min()) / spany).astype(int)
    return snowfall(xp, yp, size)


def paint(values, xp, yp, size):
    image = np.zeros((size, size), dtype=np.float32)
    counts = np.zeros((size, size), dtype=np.float32)
    np.add.at(image, (xp - 1, yp - 1), values)
    np.add.at(counts, (xp - 1, yp - 1), 1)
    occupied = counts > 0
    image[occupied] /= counts[occupied]
    pixels = image[occupied]
    pixels = pixels - pixels.min()
    if pixels.max() > 0:
        pixels = pixels / pixels.max()
    pixels = np.log1p(pixels)
    if pixels.max() > 0:
        pixels = pixels / pixels.max()
    image[occupied] = pixels
    return zoom(image, (size / image.shape[0], size / image.shape[1]), order=1)


def load_labels(path, samples):
    frame = pd.read_csv(path)
    if frame.shape[1] == 1:
        if len(frame) != len(samples):
            raise ValueError("One-column labels must have one row per sample in the first selected matrix")
        frame.columns = ["Label"]
        frame.insert(0, "Sample_ID", samples)
    else:
        frame = frame.iloc[:, :2].copy()
        frame.columns = ["Sample_ID", "Label"]
    frame["Sample_ID"] = frame["Sample_ID"].astype(str).str.strip()
    if frame["Label"].dtype == object:
        frame["Label"] = frame["Label"].astype(str).str.strip()
    return frame.drop_duplicates("Sample_ID").set_index("Sample_ID")["Label"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.environ.get("DEEPCAM_DATA_DIR", "/kaggle/input/brca-aligned"))
    p.add_argument("--output-dir", default=os.environ.get("DEEPCAM_OUTPUT_DIR", "OutputData"))
    p.add_argument("--omics", default=os.environ.get("OMICS", "mRNA,Methylation,CNV"), help="Comma-separated: mRNA,Methylation,CNV")
    p.add_argument("--k-folds", type=int, default=int(os.environ.get("K_FOLDS", "5")))
    p.add_argument("--val-fraction", type=float, default=float(os.environ.get("VAL_FRACTION", "0.2")))
    p.add_argument("--image-size", type=int, default=240)
    p.add_argument("--seed", type=int, default=44)
    args = p.parse_args()
    modalities = [x.strip() for x in args.omics.split(",") if x.strip()]
    if not modalities or len(set(modalities)) != len(modalities) or set(modalities) - set(FILES):
        p.error("--omics must be one or more unique values from mRNA,Methylation,CNV")
    matrices = {m: pd.read_csv(Path(args.data_dir) / FILES[m], index_col=0) for m in modalities}
    samples = matrices[modalities[0]].columns.astype(str)
    for matrix in matrices.values():
        samples = samples.intersection(matrix.columns.astype(str))
    labels = load_labels(Path(args.data_dir) / "BRCA_label_num.csv", matrices[modalities[0]].columns.astype(str))
    samples = np.asarray([s for s in samples if s in labels.index], dtype=str)
    y = labels.loc[samples].astype(str).to_numpy()
    matrices = {m: df.loc[:, samples].astype(np.float32) for m, df in matrices.items()}
    if args.k_folds < 2:
        p.error("--k-folds must be at least 2")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (trainval_ids, test_ids) in enumerate(splitter.split(samples, y), start=1):
        train_ids, val_ids = train_test_split(trainval_ids, test_size=args.val_fraction,
                                               random_state=args.seed + fold, stratify=y[trainval_ids])
        fold_dir = out / f"Fold_{fold}"
        if fold_dir.exists():
            import shutil
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
        layouts = {}
        for modality, matrix in matrices.items():
            # Each gene is represented by its training-set profile. Fit UMAP once for this fold;
            # validation/test values are rendered at these fixed gene coordinates.
            gene_profiles = matrix.iloc[:, train_ids].to_numpy()
            xp, yp = fit_gene_layout(gene_profiles, args.image_size, args.seed)
            layouts[modality] = (xp, yp)
            coords = pd.DataFrame({"gene_name": matrix.index, "pixel_x": yp - 1, "pixel_y": xp - 1})
            coords.to_csv(fold_dir / f"gene_coordinates_{modality}.csv", index=False)
        split_names = {i: s for s, ids in split_ids.items() for i in ids}
        for split, ids in split_ids.items():
            for label in np.unique(y):
                (fold_dir / split / str(label)).mkdir(parents=True, exist_ok=True)
            for idx in ids:
                image = np.zeros((args.image_size, args.image_size, 3), dtype=np.float32)
                for modality, matrix in matrices.items():
                    xp, yp = layouts[modality]
                    image[:, :, CHANNEL[modality]] = paint(matrix.iloc[:, idx].to_numpy(), xp, yp, args.image_size)
                sample, label = samples[idx], y[idx]
                target = fold_dir / split / label
                target.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(target / f"{sample}.png", np.uint8(np.clip(image, 0, 1) * 255))
        pd.DataFrame({"Sample_ID": samples, "Label": y,
                      "split": [split_names.get(i, "") for i in range(len(samples))]}).to_csv(fold_dir / "sample_splits.csv", index=False)
        print(f"Fold {fold}/{args.k_folds}: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}; modalities={','.join(modalities)}")
    print(f"Fold datasets and fold-specific gene coordinates saved under {out}")


if __name__ == "__main__":
    main()
