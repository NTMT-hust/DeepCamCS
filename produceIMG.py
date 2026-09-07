from collections import deque
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from shapely.geometry import MultiPoint
from sklearn.model_selection import train_test_split
import umap


# ===================== SNOWFALL =====================

def bfs_find_empty(x, y, occ, max_a, max_b):
    visited = set()
    q = deque([(x, y)])
    while q:
        cx, cy = q.popleft()
        if (cx, cy) not in occ:
            return cx, cy
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx <= max_a and 1 <= ny <= max_b and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return x, y


def snowfall_fast(xp, yp, max_a=120, max_b=120):
    xp_new = xp.copy()
    yp_new = yp.copy()
    occupied = set()
    for i in range(len(xp)):
        x = min(max(1, xp_new[i]), max_a)
        y = min(max(1, yp_new[i]), max_b)
        if (x, y) in occupied:
            x, y = bfs_find_empty(x, y, occupied, max_a, max_b)
        xp_new[i] = x
        yp_new[i] = y
        occupied.add((x, y))
    return xp_new, yp_new, max(xp_new), max(yp_new)


# ===================== CONV PIXEL =====================

def ConvPixel(feature_vec, xp, yp, a, b):
    """Convert feature vector to pixel image with black background."""
    m = np.zeros((a, b), dtype=float)
    mask = np.zeros((a, b), dtype=bool)

    for j in range(len(feature_vec)):
        m[xp[j] - 1, yp[j] - 1] = feature_vec[j]
        mask[xp[j] - 1, yp[j] - 1] = True

    # Handle duplicate coordinates
    coords = np.vstack((xp, yp)).T
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    for pid in np.unique(inv):
        idx = np.where(inv == pid)[0]
        if len(idx) > 1:
            m[xp[idx[0]] - 1, yp[idx[0]] - 1] = np.mean(feature_vec[idx])

    if mask.any():
        gene_values = m[mask]
        gene_values = gene_values - gene_values.min()
        if gene_values.max() > 0:
            gene_values = gene_values / gene_values.max()

        gene_values = np.log1p(gene_values)
        gene_values = gene_values - gene_values.min()
        if gene_values.max() > 0:
            gene_values = gene_values / gene_values.max()

        m[mask] = gene_values

    return m


# ===================== BOUNDING RECT =====================

def min_bounding_rect(x, y):
    hull = MultiPoint(np.column_stack([x, y])).convex_hull
    rect = hull.minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)
    return coords[:, 0], coords[:, 1]


# ===================== CART2PIXEL =====================

def Cart2Pixel(q, max_a=120, max_b=120, random_state=42):
    """Map features to pixel coordinates using UMAP and rotation."""
    print(f"  UMAP embedding for {q.shape[0]} features...")
    y_embed = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, q.shape[0] - 1),
        min_dist=0.3,
        metric="cosine",
        init="spectral",
        random_state=random_state,
    ).fit_transform(q)

    x, y = y_embed[:, 0], y_embed[:, 1]
    xrect, yrect = min_bounding_rect(x, y)

    theta = np.arctan2(yrect[1] - yrect[0], xrect[1] - xrect[0])
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])

    z = rotation_matrix @ np.vstack([x, y])
    zx, zy = z[0], z[1]

    xp = np.round(1 + max_a * (zx - zx.min()) / (zx.max() - zx.min())).astype(int)
    yp = np.round(1 - max_b * (zy - zy.max()) / (zy.max() - zy.min())).astype(int)

    xp, yp, a, b = snowfall_fast(xp, yp, max_a, max_b)
    return xp, yp, a, b


# ===================== RESIZE IMAGE =====================

def resize_to_target(img, target_size=(120, 120)):
    """Resize image to target size using scipy zoom."""
    zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)


def main():
    base_dir = "/kaggle/input/brca-aligned"
    target_size = 224
    out_dir = "/kaggle/working/OutputData"

    print("Loading data...")
    mrna = pd.read_csv(f"{base_dir}/BRCA_mRNA_aligned.csv", index_col=0)
    methy = pd.read_csv(f"{base_dir}/BRCA_Methy_aligned.csv", index_col=0)
    cnv = pd.read_csv(f"{base_dir}/BRCA_CNV_aligned.csv", index_col=0)
    labels = pd.read_csv(f"{base_dir}/BRCA_label_num.csv").iloc[:, 0].values

    common_samples = mrna.columns.intersection(methy.columns).intersection(cnv.columns)
    print(f"Common samples: {len(common_samples)}")

    mrna = mrna[common_samples]
    methy = methy[common_samples]
    cnv = cnv[common_samples]

    if len(labels) != len(common_samples):
        print(f"⚠️ Warning: Labels ({len(labels)}) != Samples ({len(common_samples)})")
        labels = labels[:len(common_samples)]

    sample_names = np.array(common_samples)
    all_indices = np.arange(len(common_samples))

    print("\nSplitting dataset into train_val (90%) and test (10%) stratified by subtype...")
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.10,
        stratify=labels,
        random_state=42,
    )

    qm = mrna.values.astype(np.float32)
    qme = methy.values.astype(np.float32)
    qc = cnv.values.astype(np.float32)

    qm_trainval = qm[:, train_val_idx]
    qme_trainval = qme[:, train_val_idx]
    qc_trainval = qc[:, train_val_idx]

    print("\nFitting UMAP & Snowfall pixel coordinates on TRAIN_VAL set only...")
    xp_m, yp_m, a_m, b_m = Cart2Pixel(qm_trainval, max_a=target_size, max_b=target_size, random_state=42)
    xp_me, yp_me, a_me, b_me = Cart2Pixel(qme_trainval, max_a=target_size, max_b=target_size, random_state=42)
    xp_c, yp_c, a_c, b_c = Cart2Pixel(qc_trainval, max_a=target_size, max_b=target_size, random_state=42)

    os.makedirs(out_dir, exist_ok=True)
    for split in ["train_val", "test"]:
        for c in np.unique(labels):
            os.makedirs(f"{out_dir}/{split}/{c}", exist_ok=True)

    def save_split(split, indices, y):
        print(f"\nGenerating {split} images ({len(indices)} samples)...")
        for i, (idx, lbl) in enumerate(zip(indices, y)):
            img_m = resize_to_target(ConvPixel(qm[:, idx], xp_m, yp_m, a_m, b_m), (target_size, target_size))
            img_me = resize_to_target(ConvPixel(qme[:, idx], xp_me, yp_me, a_me, b_me), (target_size, target_size))
            img_c = resize_to_target(ConvPixel(qc[:, idx], xp_c, yp_c, a_c, b_c), (target_size, target_size))

            img = np.stack([img_m, img_me, img_c], axis=-1)
            imageio.imwrite(
                f"{out_dir}/{split}/{lbl}/sample_{idx}.png",
                (img * 255).astype(np.uint8),
            )

    save_split("train_val", train_val_idx, labels[train_val_idx])
    save_split("test", test_idx, labels[test_idx])

    pd.DataFrame({"gene_name": mrna.index, "pixel_x": xp_m, "pixel_y": yp_m}).to_csv(f"{out_dir}/gene_coordinates_mRNA.csv", index=False)
    pd.DataFrame({"gene_name": methy.index, "pixel_x": xp_me, "pixel_y": yp_me}).to_csv(f"{out_dir}/gene_coordinates_Methylation.csv", index=False)
    pd.DataFrame({"gene_name": cnv.index, "pixel_x": xp_c, "pixel_y": yp_c}).to_csv(f"{out_dir}/gene_coordinates_CNV.csv", index=False)

    pd.DataFrame({"sample_index": train_val_idx, "sample_name": sample_names[train_val_idx], "label": labels[train_val_idx]}).to_csv(f"{out_dir}/train_val_samples.csv", index=False)
    pd.DataFrame({"sample_index": test_idx, "sample_name": sample_names[test_idx], "label": labels[test_idx]}).to_csv(f"{out_dir}/test_samples.csv", index=False)

    print(f"\n✅ DONE: Images saved to '{out_dir}/train_val/' and '{out_dir}/test/'")


if __name__ == "__main__":
    main()