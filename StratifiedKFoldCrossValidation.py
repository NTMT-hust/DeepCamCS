from collections import defaultdict
import gc
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm

from additional_function import (
    calculate_class_weights,
    calculate_comprehensive_metrics,
    load_dataset_from_folder,
    plot_confusion_matrix,
    print_detailed_metrics,
)
from Dataset import OMICS_CHANNELS, OmicsImageDataset
from FocalLoss import FocalLoss
from guidance import FeatureRelevance, MaskBuilder, guided_penalty
from models import build_model, unwrap_model
from sparse_integrated_gradients import SparseIntegratedGradients


class StratifiedKFoldCrossValidation:
    """K-fold training with SGD + SWA.

    The validation split of each fold is used only to build the guidance mask (and for the
    explanation records); it never selects a checkpoint. The final model of each fold is the
    SWA average, evaluated once on the held-out test split.
    """

    def __init__(
        self,
        dataset_path,
        channels=tuple(OMICS_CHANNELS),
        backbone="efficientnet_b1",
        pretrained=True,
        unet_width=32,
        unet_depth=4,
        loss="focal",
        class_weights=True,
        focal_gamma=2.0,
        k_folds=5,
        num_epochs=70,
        batch_size=32,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-3,
        swa_lr=0.005,
        swa_start_epoch=40,
        dropout_rate=0.5,
        lambda1=10,
        mask_method="gradcam",
        mask_start_epoch=40,
        mask_refresh_every=5,
        holdout_fraction=0.2,
        checkpoint_dir=".",
        random_seed=12,
    ):
        if loss not in {"focal", "ce"}:
            raise ValueError("loss must be 'focal' or 'ce'")
        if not channels or any(c not in OMICS_CHANNELS for c in channels):
            raise ValueError(f"channels must be a non-empty subset of {list(OMICS_CHANNELS)}")
        if num_epochs <= swa_start_epoch:
            raise ValueError(f"num_epochs ({num_epochs}) must exceed swa_start_epoch ({swa_start_epoch}).")
        if lambda1 > 0 and num_epochs <= mask_start_epoch:
            raise ValueError(f"num_epochs ({num_epochs}) must exceed mask_start_epoch ({mask_start_epoch}).")

        self.dataset_path = dataset_path
        self.channels = list(channels)
        self.backbone = backbone
        self.pretrained = pretrained
        self.unet_width = unet_width
        self.unet_depth = unet_depth
        self.loss = loss
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.k_folds = k_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.swa_lr = swa_lr
        self.swa_start_epoch = swa_start_epoch
        self.dropout_rate = dropout_rate
        self.lambda1 = lambda1  # 0 disables explanation-guided learning
        self.mask_method = mask_method
        self.mask_start_epoch = mask_start_epoch
        self.mask_refresh_every = mask_refresh_every
        self.holdout_fraction = holdout_fraction
        self.checkpoint_dir = Path(checkpoint_dir)
        self.random_seed = random_seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(random_seed)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _make_loader(self, paths, labels, shuffle):
        dataset = OmicsImageDataset(paths, labels, self.channels)
        return dataset, DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _build_criterion(self, train_labels, num_classes):
        alpha = None
        if self.class_weights:
            alpha = calculate_class_weights(train_labels, num_classes).to(self.device)
        # Cross entropy is focal loss with gamma = 0, so both share the same class weighting and scale.
        gamma = self.focal_gamma if self.loss == "focal" else 0.0
        return FocalLoss(num_classes, alpha=alpha, gamma=gamma)

    def run(self):
        image_paths, labels, class_names, num_classes = load_dataset_from_folder(self.dataset_path)
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        self._set_seed(self.random_seed)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.holdout_fraction, random_state=self.random_seed)
        trainval_idx, test_idx = next(sss.split(image_paths, labels))
        _, test_loader = self._make_loader(image_paths[test_idx], labels[test_idx], shuffle=False)
        image_paths, labels = image_paths[trainval_idx], labels[trainval_idx]

        print(f'\n{"="*60}')
        print(f"Device: {self.device} | seed: {self.random_seed}")
        print(f"Channels: {self.channels} | backbone: {self.backbone} | loss: {self.loss}")
        print(f"Holdout: {self.holdout_fraction:.0%} ({len(test_idx)} samples)")
        print(f'{"="*60}')

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)
        fold_results, all_ig_records, fold_test_results = [], [], {}

        for fold, (train_ids, val_ids) in enumerate(skf.split(image_paths, labels)):
            print(f'\n{"="*20} FOLD {fold + 1}/{self.k_folds} {"="*20}')
            result, ig_records = self._run_fold(
                fold,
                image_paths[train_ids], labels[train_ids],
                image_paths[val_ids], labels[val_ids],
                test_loader, class_names, num_classes,
            )
            fold_results.append(result)
            fold_test_results[fold] = {"test_metrics": result["test_metrics"]}
            all_ig_records.extend(ig_records)

            gc.collect()
            torch.cuda.empty_cache()

        return fold_results, class_names, all_ig_records, fold_test_results

    def _run_fold(self, fold, train_paths, train_labels, val_paths, val_labels,
                  test_loader, class_names, num_classes):
        ds_train, train_loader = self._make_loader(train_paths, train_labels, shuffle=True)
        _, val_loader = self._make_loader(val_paths, val_labels, shuffle=False)

        model = build_model(
            self.backbone, num_classes, len(self.channels), self.dropout_rate,
            self.pretrained, self.unet_width, self.unet_depth,
        )
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(self.device)

        criterion = self._build_criterion(train_labels, num_classes)
        optimizer = optim.SGD(
            model.parameters(), lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay, nesterov=True,
        )
        # Cosine decay down to the SWA learning rate, then a constant SWA learning rate.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.swa_start_epoch, eta_min=self.swa_lr
        )
        swa_scheduler = None
        swa_model = AveragedModel(model)

        mask_builder, mask = None, None
        if self.lambda1 > 0:
            mask_builder = MaskBuilder(
                num_classes=num_classes,
                relevance=FeatureRelevance.from_training_split(ds_train, protect_top_fraction=0.5),
                method=self.mask_method,
            )

        history = defaultdict(list)
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            if epoch == self.swa_start_epoch:
                swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)

            loss, y_true, y_pred, y_prob = train_epoch(
                model, train_loader, criterion, optimizer, self.device, mask, self.lambda1
            )
            metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob, num_classes, class_names)
            history["train_loss"].append(loss)
            history["train_acc"].append(metrics["accuracy"])
            history["train_f1"].append(metrics["f1_macro"])
            history["train_auc"].append(metrics.get("roc_auc", metrics.get("roc_auc_ovr", 0)))
            print(
                f'Train - Loss: {loss:.4f} | Acc: {metrics["accuracy"]:.2f}% | '
                f'AUC: {history["train_auc"][-1]:.4f} | F1: {metrics["f1_macro"]:.2f}%'
            )

            # The mask (built from the validation split) takes effect from the next epoch.
            completed = epoch + 1
            if (
                mask_builder is not None
                and completed >= self.mask_start_epoch
                and (completed - self.mask_start_epoch) % self.mask_refresh_every == 0
            ):
                mask = mask_builder.build(model, val_loader, self.device)
                print(f"Updated explanation mask (active pixels per class: "
                      f"{mask.flatten(1).sum(dim=1).int().tolist()}).")

            if epoch >= self.swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        # Final model = SWA average; recompute BatchNorm statistics for it.
        update_bn(train_loader, swa_model, device=self.device)
        swa_model.eval()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(unwrap_model(swa_model).state_dict(), self.checkpoint_dir / f"swa_model_fold{fold}.pth")

        print(f"\nFold {fold + 1} training complete. Evaluating SWA model on TEST SET...")
        y_true, y_pred, y_prob = predict(swa_model, test_loader, self.device)
        test_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob, num_classes, class_names)
        print(
            f"  [Fold {fold + 1}] TEST — Acc: {test_metrics['accuracy']:.2f}% | "
            f"AUC: {test_metrics.get('roc_auc', test_metrics.get('roc_auc_ovr', 0)):.4f} | "
            f"F1:  {test_metrics['f1_macro']:.2f}%"
        )
        plot_confusion_matrix(
            test_metrics["confusion_matrix"], class_names,
            title=f"Fold {fold + 1} — Test Confusion Matrix",
            save_path=f"confusion_matrix_fold{fold + 1}_test.png",
        )
        print_detailed_metrics(test_metrics, class_names, fold + 1)

        # Explain the fold's validation samples with the SWA model.
        ig_records = generate_ig_records(swa_model, val_loader, self.device, class_names, steps=8)

        return {"fold": fold + 1, "test_metrics": test_metrics, "history": dict(history)}, ig_records


def train_epoch(model, dataloader, criterion, optimizer, device, mask, lambda1):
    """One training epoch, with the input-gradient penalty once a mask exists."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()

        guided = mask is not None
        inputs.requires_grad_(guided)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if guided:
            s = torch.log_softmax(outputs, dim=1).sum(dim=1)
            grads = torch.autograd.grad(s.sum(), inputs, create_graph=True)[0]
            loss = loss + lambda1 * guided_penalty(grads, mask, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    return running_loss / len(dataloader), np.array(all_labels), np.array(all_preds), np.array(all_probs)


def predict(model, dataloader, device):
    """Return (labels, predictions, probabilities) for a dataloader."""
    model.eval()
    labels_out, probs_out = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluation", leave=False):
            probs = torch.softmax(model(inputs.to(device)), dim=1)
            probs_out.append(probs.cpu().numpy())
            labels_out.append(labels.numpy())
    probs_out = np.concatenate(probs_out)
    return np.concatenate(labels_out), probs_out.argmax(axis=1), probs_out


def generate_ig_records(model, dataloader, device, class_names, steps=8):
    """Return (CxHxW attribution, predicted class name) records for a dataset."""
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
