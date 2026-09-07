import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ===================== GRAD-CAM CORE =====================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap_raw = torch.sum(weights * self.activations, dim=1).squeeze()

        heatmap = torch.clamp(heatmap_raw, min=0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmap_raw = self.upsample_heatmap(heatmap_raw, input_tensor)
        return heatmap.cpu().numpy(), class_idx, heatmap_raw.cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def upsample_heatmap(self, heatmap, input_tensor):
        """
        heatmap: (H_cam, W_cam) tensor
        input_tensor: (1, C, H, W)
        """
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H_cam, W_cam)
        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        return heatmap.squeeze()  # (H, W)


def save_gradcam_image(img_path, heatmap, pred_class, true_class, output_dir, alpha=0.4):
    """Overlays a Grad-CAM heatmap on the original image and saves it."""
    raw_img = cv2.imread(str(img_path))
    if raw_img is None:
        print(f"Error: Could not load image {img_path}")
        return None

    height, width, _ = raw_img.shape
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_norm = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(raw_img, 1 - alpha, heatmap_color, alpha, 0)
    label_text = f"Pred: {pred_class} | True: {true_class}"
    cv2.putText(
        overlay,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    sample_name = Path(img_path).stem
    save_path = Path(output_dir) / f"{sample_name}_gradcam.png"
    cv2.imwrite(str(save_path), overlay)
    return heatmap_resized


def generate_epoch_gradcam(model, dataloader, device, epoch, class_names, output_path, save_limit=20):
    """Generate Grad-CAM heatmaps for an epoch and compute global difference heatmap."""
    model.to(device)
    model.eval()

    backbone = model.module.backbone if hasattr(model, "module") else model.backbone
    if hasattr(backbone, "conv_head"):
        target_layer = backbone.conv_head
    elif hasattr(backbone, "blocks"):
        target_layer = backbone.blocks[-1]
    elif hasattr(backbone, "layer4"):
        target_layer = backbone.layer4[-1]
    else:
        raise ValueError("Unsupported backbone for Grad-CAM")

    cam_extractor = GradCAM(model.module if hasattr(model, "module") else model, target_layer)
    os.makedirs(output_path, exist_ok=True)

    pos_heatmaps = []
    neg_heatmaps = []
    raw_heatmaps = []

    for i in range(min(len(dataloader.dataset), save_limit)):
        img_tensor, label = dataloader.dataset[i]
        img_path = dataloader.dataset.image_paths[i]

        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        with torch.enable_grad():
            heatmap, pred_idx, raw_heatmap = cam_extractor.generate(input_tensor, class_idx=None)

        pred_class = class_names[pred_idx]
        true_class = class_names[label]

        raw_heatmaps.append((raw_heatmap, pred_class))

        save_gradcam_image(
            img_path=img_path,
            heatmap=heatmap,
            pred_class=pred_class,
            true_class=true_class,
            output_dir=output_path,
        )

        if pred_class == true_class:
            pos_heatmaps.append(raw_heatmap)
        else:
            neg_heatmaps.append(raw_heatmap)

    cam_extractor.remove_hooks()
    model.zero_grad()
    torch.cuda.empty_cache()

    fallback_shape = None
    if len(pos_heatmaps) > 0:
        fallback_shape = pos_heatmaps[0].shape
    elif len(neg_heatmaps) > 0:
        fallback_shape = neg_heatmaps[0].shape

    pos_heatmap = safe_mean_stack(pos_heatmaps, fallback_shape)
    neg_heatmap = safe_mean_stack(neg_heatmaps, fallback_shape)

    if pos_heatmap is None:
        return None, raw_heatmaps

    bin_pos_heatmap = (pos_heatmap >= 0).astype(np.float32)
    bin_neg_heatmap = (neg_heatmap <= 0).astype(np.float32)

    global_heatmap = bin_neg_heatmap - bin_pos_heatmap
    return global_heatmap, raw_heatmaps


def build_mask_from_heatmap(global_heatmap, device):
    """Convert global heatmap to a 4D tensor mask (1, 1, H, W)."""
    mask_tensor = torch.tensor(global_heatmap, dtype=torch.float32).to(device)
    return mask_tensor.unsqueeze(0).unsqueeze(0)


def safe_mean_stack(heatmaps, fallback_shape=None):
    """Safely computes the mean across a list of heatmap arrays."""
    if len(heatmaps) == 0:
        if fallback_shape is None:
            return None
        return np.zeros(fallback_shape, dtype=np.float32)
    return np.mean(np.stack(heatmaps), axis=0)
