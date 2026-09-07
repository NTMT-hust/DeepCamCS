import csv
import os
import cv2
import numpy as np
import pandas as pd


def calculate_mean(all_heatmaps):
    """Computes mean heatmap per predicted class across collected heatmaps."""
    class_sum = {}
    class_count = {}

    for heatmap, cls in all_heatmaps:
        if cls not in class_sum:
            class_sum[cls] = heatmap.copy().astype(np.float32)
            class_count[cls] = 1
        else:
            class_sum[cls] += heatmap.astype(np.float32)
            class_count[cls] += 1

    return {cls: class_sum[cls] / class_count[cls] for cls in class_sum}


def calculate_pixel_attribute_score(heatmap, method="raw"):
    """
    Calculate attribute score for each pixel from a heatmap.

    Args:
        heatmap (np.ndarray): 2D array of shape (H, W).
        method (str): "raw" for raw values, "normalized" for [-1, 1] scaled values.
    """
    attr_score = heatmap.copy().astype(np.float32)
    if method == "normalized":
        max_abs = np.max(np.abs(attr_score))
        if max_abs > 0:
            attr_score = attr_score / max_abs
    return attr_score


def visualize_mean_heatmap(mean_heatmap, save_path):
    """Safely visualizes 2D heatmaps using JET colormap."""
    h = mean_heatmap.copy().astype(np.float32)
    h_min, h_max = h.min(), h.max()
    if h_max > h_min:
        h_norm = (h - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(h)

    heatmap_uint8 = np.uint8(255 * h_norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap_color)


def find_critical_pixel(heatmap, threshold):
    """Finds pixel coordinates where attribute score >= threshold."""
    coords = np.argwhere(heatmap >= threshold)
    return [[int(x), int(y), float(heatmap[y, x])] for y, x in coords]


def find_genes_attribute_scores(attr_map, coord_filepath, omics_type, predicted_label, threshold=None):
    """
    Map 2D pixel attribute scores to individual genes using coordinate lookup.

    Produces DataFrame with columns required by ChooseGenes.py:
        ['gene_name', 'predicted_label', 'omics_type', 'attribute_score']
    """
    if not os.path.exists(coord_filepath):
        print(f"⚠️ Coordinate file not found: {coord_filepath}")
        return pd.DataFrame(columns=["gene_name", "predicted_label", "omics_type", "attribute_score"])

    coords_df = pd.read_csv(coord_filepath)
    h, w = attr_map.shape[:2]
    records = []

    for _, row in coords_df.iterrows():
        x = int(row["pixel_x"])
        y = int(row["pixel_y"])

        if 0 <= x < w and 0 <= y < h:
            score = float(attr_map[y, x])
            if threshold is None or score >= threshold:
                records.append({
                    "gene_name": row["gene_name"],
                    "predicted_label": predicted_label,
                    "omics_type": omics_type,
                    "attribute_score": score,
                })

    return pd.DataFrame(records)


def save_to_csv(rows, csv_path, headers=None):
    """Saves rows to CSV file."""
    if headers is None:
        headers = ["x", "y", "value"] if len(rows) > 0 and len(rows[0]) == 3 else ["x", "y", "gene_name", "attribute_score"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
