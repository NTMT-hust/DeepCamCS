"""Classification backbones with a shared interface.

Every model exposes ``forward_features`` (a 4D feature map) and ``forward_head``
(feature map -> logits), which Grad-CAM uses to take gradients w.r.t. the feature map.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_model(model):
    """Strip DataParallel / AveragedModel wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


class TimmClassifier(nn.Module):
    """Any convolutional timm model (efficientnet_*, resnet*, resnet26d, ...)."""

    def __init__(self, name, num_classes, in_chans, dropout_rate, pretrained):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=dropout_rate,
        )

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward_head(self, features):
        return self.backbone.forward_head(features)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetClassifier(nn.Module):
    """U-Net encoder-decoder; global-pooled full-resolution decoder features feed a linear head."""

    def __init__(self, num_classes, in_chans, dropout_rate, base_width=32, depth=4):
        super().__init__()
        widths = [base_width * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [_double_conv(in_chans, widths[0])]
            + [_double_conv(widths[i], widths[i + 1]) for i in range(depth)]
        )
        self.upsamples = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.decoders = nn.ModuleList(
            [_double_conv(widths[i] * 2, widths[i]) for i in reversed(range(depth))]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(widths[0], num_classes)

    def forward_features(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()  # the bottleneck is not a skip connection
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return x

    def forward_head(self, features):
        return self.fc(self.dropout(features.mean(dim=(2, 3))))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(name, num_classes, in_chans, dropout_rate, pretrained=True, unet_width=32, unet_depth=4):
    """``name`` is "unet" or any convolutional timm model name (e.g. resnet26d, resnet50, efficientnet_b1)."""
    if name.lower() == "unet":
        return UNetClassifier(num_classes, in_chans, dropout_rate, unet_width, unet_depth)
    return TimmClassifier(name, num_classes, in_chans, dropout_rate, pretrained)
