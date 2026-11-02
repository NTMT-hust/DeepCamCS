from collections import defaultdict
import gc
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from additional_function import (
    calculate_class_weights,
    calculate_comprehensive_metrics,
    load_dataset_from_folder,
    plot_confusion_matrix,
    print_detailed_metrics,
)
from EfficientNetB1Classifier import EfficientNetB1Classifier
from FocalLoss import FocalLoss
from GradCAM import build_mask_from_heatmap, generate_epoch_gradcam
from Dataset import ImbalancedImageDataset
from sparse_integrated_gradients import SparseIntegratedGradients

class StratifiedKFoldCrossValidation:
    def __init__(
        self,
        dataset_path,
        k_folds=5,
        num_epochs=20,
        batch_size=32,
        lr=0.0001,
        weight_decay=1e-4,
        dropout_rate=0.3,
        focal_gamma=2.0,
        random_seed=42,
        lambda1=10
    ):
        self.dataset_path = dataset_path
        self.k_folds = k_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.focal_gamma = focal_gamma
        self.random_seed = random_seed
        self.lambda1 = lambda1

        self._set_seed(random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = None

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        # 1. Load Dataset
        image_paths, labels, class_names, num_classes = load_dataset_from_folder(self.dataset_path)
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        self._set_seed(self.random_seed)

        # 2. Split Holdout Test Set (10% stratified)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=self.random_seed)
        trainval_idx, test_idx = next(sss.split(image_paths, labels))

        test_image_paths = image_paths[test_idx]
        test_labels_arr = labels[test_idx]
        image_paths = image_paths[trainval_idx]
        labels = labels[trainval_idx]

        print(f'\n{"="*60}')
        print(f"Using device: {self.device}")
        print(f"Random seed: {self.random_seed}")
        print(f'{"="*60}')


        test_class_counts = np.bincount(labels)
        ds_test = ImbalancedImageDataset(
            test_image_paths, test_labels_arr, test_class_counts, transform=ToTensor()
        )
        test_loader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False)

        # 4. Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)
        all_fold_models = []
        all_ig_record = []
        fold_results = []
        fold_test_results = {}

        for fold, (train_ids, val_ids) in enumerate(skf.split(image_paths, labels)):
            print(f'\n{"="*20} FOLD {fold + 1}/{self.k_folds} {"="*20}')

            # A fold must build its own guidance mask after the warmup period.
            self.mask = None

            train_labels = labels[train_ids]
            train_class_counts = np.bincount(train_labels, minlength=num_classes)

            ds_train = ImbalancedImageDataset(
                image_paths[train_ids],
                train_labels,
                train_class_counts,
                transform=ToTensor()
            )
            ds_val = ImbalancedImageDataset(
                image_paths[val_ids],
                labels[val_ids],
                train_class_counts,
                transform=ToTensor()
            )
            train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

            model = EfficientNetB1Classifier(num_classes=num_classes, dropout_rate=self.dropout_rate).to(self.device)
            model = nn.DataParallel(model)
            model.to(self.device)

            class_weights = calculate_class_weights(train_labels, num_classes).to(self.device)
            criterion = FocalLoss(num_classes, alpha=class_weights, gamma=self.focal_gamma)
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
            best_val_f1 = -1
            best_metrics = {}
            fold_history = defaultdict(list)

            for epoch in range(self.num_epochs):
                print(f"\nEpoch {epoch+1}/{self.num_epochs}", end="")

                print()

                train_loss, train_labels_epoch, train_preds, train_probs = train_epoch(
                    model=model,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    mask=self.mask,
                    lambda1=self.lambda1,
                )
                train_metrics = calculate_comprehensive_metrics(
                    train_labels_epoch, train_preds, train_probs, num_classes, class_names
                )

                # Generate GradCAM from validation set
                global_heatmap, epoch_raw_heatmaps = generate_epoch_gradcam(
                    model=model,
                    dataloader=val_loader,
                    device=self.device,
                    epoch=epoch,
                    class_names=class_names,
                    output_path=Path(f"GradCAM/Fold_{fold+1}"),
                )

                # Complete 9 warmup epochs, then refresh after every 3 epochs
                # (completed epochs 9, 12, 15, ...; zero-based indices 8, 11, 14, ...).
                completed_epochs = epoch + 1
                if (
                    completed_epochs >= 9
                    and (completed_epochs - 9) % 3 == 0
                    and global_heatmap is not None
                ):
                    self.mask = build_mask_from_heatmap(global_heatmap, self.device)
                    print("Updated explanation mask.")

                # Validate
                val_loss, val_labels_epoch, val_preds, val_probs = validate_epoch(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=self.device,
                )
                val_metrics = calculate_comprehensive_metrics(
                    val_labels_epoch, val_preds, val_probs, num_classes, class_names
                )
                
                # test
                test_loss, test_labels_epoch, test_preds, test_probs = validate_epoch(
                    model=model,
                    dataloader=test_loader,
                    criterion=criterion,
                    device=self.device
                )
                test_metrics = calculate_comprehensive_metrics(
                    test_labels_epoch, test_preds, test_probs, num_classes, class_names
                )

                scheduler.step()

                # Record history
                fold_history["train_loss"].append(train_loss)
                fold_history["train_acc"].append(train_metrics["accuracy"])
                fold_history["train_auc"].append(train_metrics.get("roc_auc", train_metrics.get("roc_auc_ovr", 0)))
                fold_history["train_sens"].append(train_metrics["sensitivity"])
                fold_history["train_spec"].append(train_metrics["specificity"])
                fold_history["train_f1"].append(train_metrics["f1_macro"])

                fold_history["val_loss"].append(val_loss)
                fold_history["val_acc"].append(val_metrics["accuracy"])
                fold_history["val_auc"].append(val_metrics.get("roc_auc", val_metrics.get("roc_auc_ovr", 0)))
                fold_history["val_sens"].append(val_metrics["sensitivity"])
                fold_history["val_spec"].append(val_metrics["specificity"])
                fold_history["val_f1"].append(val_metrics["f1_macro"])

                print(
                    f'Train - Loss: {train_loss:.4f} | Acc: {train_metrics["accuracy"]:.2f}% | '
                    f'AUC: {train_metrics.get("roc_auc", train_metrics.get("roc_auc_ovr", 0)):.4f} | '
                    f'F1: {train_metrics["f1_macro"]:.2f}%'
                )
                print(
                    f'Val   - Loss: {val_loss:.4f} | Acc: {val_metrics["accuracy"]:.2f}% | '
                    f'AUC: {val_metrics.get("roc_auc", val_metrics.get("roc_auc_ovr", 0)):.4f} | '
                    f'F1: {val_metrics["f1_macro"]:.2f}%'
                )

                curr_f1 = val_metrics.get("f1_macro", 0)
                if curr_f1 > best_val_f1:
                    best_val_f1 = curr_f1
                    best_metrics = val_metrics
                    torch.save(model.state_dict(), f"best_model_fold{fold}.pth")

            # Evaluation on test set
            print(f"\nFold {fold + 1} training complete. Evaluating best model on TEST SET...")
            model.load_state_dict(torch.load(f"best_model_fold{fold}.pth", map_location=self.device))
            model.to(self.device)
            model.eval()

            # Explain this fold's validation samples with its best checkpoint.
            # These samples are within the training partition; the separate holdout stays untouched.
            fold_ig_records = generate_ig_records(
                model=model,
                dataloader=val_loader,
                device=self.device,
                class_names=class_names,
                steps=8,
            )
            all_ig_record.extend(fold_ig_records)

            raw_test_probs_list, raw_test_labels_list = [], []
            with torch.no_grad():
                for t_inputs, t_labels in tqdm(test_loader, desc="  Test Evaluation", leave=False):
                    t_inputs = t_inputs.to(self.device)
                    t_logits = model(t_inputs)
                    t_probs = torch.softmax(t_logits, dim=1)
                    raw_test_probs_list.append(t_probs.cpu().numpy())
                    raw_test_labels_list.extend(t_labels.numpy())

            raw_test_probs = np.concatenate(raw_test_probs_list, axis=0)
            raw_test_labels = np.array(raw_test_labels_list)
            raw_test_preds = np.argmax(raw_test_probs, axis=1)

            test_metrics_before = calculate_comprehensive_metrics(
                raw_test_labels, raw_test_preds, raw_test_probs, num_classes, class_names
            )
            print(
                f"  [Fold {fold+1}] TEST — "
                f'Acc: {test_metrics_before["accuracy"]:.2f}% | '
                f'AUC: {test_metrics_before.get("roc_auc", test_metrics_before.get("roc_auc_ovr", 0)):.4f} | '
                f'F1:  {test_metrics_before["f1_macro"]:.2f}%'
            )

            plot_confusion_matrix(
                test_metrics_before["confusion_matrix"],
                class_names,
                title=f"Fold {fold+1} — Test Confusion Matrix",
                save_path=f"confusion_matrix_fold{fold+1}_test.png",
            )

            fold_test_results[fold] = {
                "test_metrics": test_metrics_before,
            }
            all_fold_models.append(model)
            fold_results.append({
                "fold": fold + 1,
                "metrics": best_metrics,
                "test_metrics": fold_test_results[fold],
                "history": dict(fold_history),
            })
            print_detailed_metrics(best_metrics, class_names, fold + 1)

            del optimizer
            gc.collect()
            torch.cuda.empty_cache()

        return fold_results, all_fold_models, [], class_names, {}, all_ig_record, fold_test_results


def train_epoch(model, dataloader, criterion, optimizer, device, mask, lambda1):
    """Executes a single training epoch with optional explanation guided loss."""
    model.to(device)
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        model.train()
        inputs, labels = inputs.to(device), labels.to(device).long()
        inputs.requires_grad_(True)

        optimizer.zero_grad()
        outputs = model(inputs)

        focall_loss = criterion(outputs, labels)

        # Right Reasons guided loss
        if mask is not None:
            probs = torch.softmax(outputs, dim=1)
            s = torch.log(probs + 1e-8).sum(dim=1)

            grads = torch.autograd.grad(
                outputs=s.sum(),
                inputs=inputs,
                create_graph=True,
            )[0]

            batch_mask = mask.repeat(inputs.size(0), inputs.size(1), 1, 1)
            grad_penalty = ((batch_mask * grads) ** 2).mean()
            loss = focall_loss + lambda1 * grad_penalty
        else:
            loss = focall_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def generate_ig_records(model, dataloader, device, class_names, steps=8):
    """Return (3xHxW attribution, predicted class name) records for a dataset."""
    model.eval()
    ig = SparseIntegratedGradients(model, device, steps=steps)
    records = []

    for inputs, _ in tqdm(dataloader, desc="Integrated Gradients", leave=False):
        inputs = inputs.to(device)
        with torch.no_grad():
            predicted = model(inputs).argmax(dim=1)

        for i in range(inputs.size(0)):
            class_idx = int(predicted[i].item())
            attr = ig.attribute(inputs[i:i + 1], class_idx)
            records.append((attr[0].detach().cpu().numpy(), class_names[class_idx]))

    return records


def validate_epoch(model, dataloader, criterion, device):
    """Executes a validation pass over the dataloader."""
    model.to(device)
    model.eval()

    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)
