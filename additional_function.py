from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
import torch


def load_dataset_from_folder(dataset_path):
    """Load dataset from folder structure with class subfolders."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {dataset_path}")

    class_names = [folder.name for folder in class_folders]
    num_classes = len(class_names)

    print(f"\nFound {num_classes} classes: {class_names}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    image_paths = []
    labels = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for class_folder in class_folders:
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]

        class_images = [
            str(img_path)
            for img_path in class_folder.iterdir()
            if img_path.suffix.lower() in image_extensions
        ]

        print(f"  Class '{class_name}' (label {class_idx}): {len(class_images)} images")

        if len(class_images) == 0:
            print(f"    ⚠️  Warning: No images found in {class_folder}")

        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    print(f"\nTotal images loaded: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")

    return image_paths, labels, class_names, num_classes


def calculate_class_weights(labels, num_classes):
    """Calculate class weights inversely proportional to class frequencies."""
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts)
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


def get_sample_weights(labels, class_counts):
    """Get sample weights for WeightedRandomSampler."""
    return 1.0 / class_counts[labels]


def calculate_comprehensive_metrics(y_true, y_pred, y_prob, num_classes, class_names=None):
    """
    Calculate comprehensive evaluation metrics.

    Returns:
        Dictionary containing all metrics.
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = 100.0 * np.sum(y_true == y_pred) / len(y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    # Precision, Recall, F1-Score
    metrics["precision_macro"] = 100.0 * precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = 100.0 * recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["precision_weighted"] = 100.0 * precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = 100.0 * recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = 100.0 * f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Matthews Correlation Coefficient
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # AUC metrics
    if num_classes == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        metrics["pr_auc"] = average_precision_score(y_true, y_prob[:, 1])
    else:
        # Multiclass One-vs-Rest / One-vs-One
        metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")

        # Multiclass PR-AUC
        pr_auc_scores = []
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_prob[:, i]
            pr_auc_scores.append(average_precision_score(y_true_binary, y_prob_binary))
        metrics["pr_auc"] = np.mean(pr_auc_scores)

    # Sensitivity and Specificity
    sensitivities = []
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        sens = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)

    metrics["sensitivity"] = np.mean(sensitivities)
    metrics["specificity"] = np.mean(specificities)
    metrics["per_class_sensitivity"] = sensitivities
    metrics["per_class_specificity"] = specificities

    # Per-class classification details
    if class_names is not None:
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        metrics["classification_report"] = class_report

        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if class_name in class_report:
                per_class_metrics[f"class_{i}"] = {
                    "precision": class_report[class_name]["precision"] * 100,
                    "recall": class_report[class_name]["recall"] * 100,
                    "f1": class_report[class_name]["f1-score"] * 100,
                    "support": class_report[class_name]["support"],
                }
        metrics["per_class_details"] = per_class_metrics

    metrics["ari"] = None
    metrics["dbi"] = None
    metrics["silhouette"] = None

    return metrics


def print_detailed_metrics(metrics, class_names=None, fold_num=None):
    """Print detailed metrics in a formatted way."""
    header = f"FOLD {fold_num} - DETAILED METRICS" if fold_num else "DETAILED METRICS"
    print(f"\n{'='*60}\n{header}\n{'='*60}")

    print("\n📊 Basic Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.2f}%")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.2f}%")
    print(f"  Precision (Weight): {metrics['precision_weighted']:.2f}%")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.2f}%")
    print(f"  Recall (Weight):    {metrics['recall_weighted']:.2f}%")
    print(f"  F1-Score (Macro):   {metrics['f1_macro']:.2f}%")
    print(f"  F1-Score (Weight):  {metrics['f1_weighted']:.2f}%")
    print(f"  MCC:                {metrics['mcc']:.4f}")

    print("\n📈 AUC Metrics:")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    else:
        print(f"  ROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
        print(f"  ROC-AUC (OvO):      {metrics['roc_auc_ovo']:.4f}")
    print(f"  PR-AUC:             {metrics['pr_auc']:.4f}")

    print("\n🎯 Sensitivity & Specificity:")
    print(f"  Sensitivity:        {metrics['sensitivity']:.2f}%")
    print(f"  Specificity:        {metrics['specificity']:.2f}%")

    if "per_class_details" in metrics and class_names:
        print("\n📋 Per-Class Metrics:")
        for i, class_name in enumerate(class_names):
            if f"class_{i}" in metrics["per_class_details"]:
                class_metrics = metrics["per_class_details"][f"class_{i}"]
                print(f"  {class_name}:")
                print(f"    Precision: {class_metrics['precision']:.2f}%")
                print(f"    Recall:    {class_metrics['recall']:.2f}%")
                print(f"    F1-Score:  {class_metrics['f1']:.2f}%")
                print(f"    Support:   {class_metrics['support']}")

    print("\n📊 Confusion Matrix:")
    print_confidence_matrix(metrics["confusion_matrix"], class_names)

    if metrics.get("ari") is not None:
        print("\n🔍 Cluster Validation Metrics:")
        print(f"  Adjusted Rand Index (ARI): {metrics['ari']:.4f}")
        print(f"  Davies-Bouldin Index (DBI): {metrics['dbi']:.4f}")
        print(f"  Silhouette Score: {metrics['silhouette']:.4f}")


def print_confidence_matrix(cm, class_names=None):
    """Print confusion matrix in absolute and percentage formats."""
    cm_df = pd.DataFrame(cm)
    if class_names:
        cm_df.columns = class_names
        cm_df.index = class_names

    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    print("\nAbsolute values:")
    print(cm_df.to_string())

    print("\nPercentage (%):")
    for i in range(len(cm)):
        row = [f"{cm_percent[i, j]:.1f}%" for j in range(len(cm))]
        prefix = f"{class_names[i]:15}" if class_names else f"Class {i:2}        "
        print(f"{prefix} {', '.join(row)}")


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(10, 8))

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_cluster_metrics(model, dataloader, device, num_classes):
    """Calculate cluster validation metrics (ARI, DBI, Silhouette)."""
    from sklearn.cluster import KMeans

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            batch_features = model.extract_features(inputs)
            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())

    if not features:
        return None, None, None

    features = np.vstack(features)
    labels = np.array(labels)

    if len(features) > num_classes:
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(features)

        ari = adjusted_rand_score(labels, predicted_labels)
        dbi = davies_bouldin_score(features, predicted_labels)
        silhouette = silhouette_score(features, predicted_labels)

        return ari, dbi, silhouette

    return None, None, None