"""Networks for Part B.

SmallCNN          deep-learning baseline trained from scratch (no pretraining).
build_transfer    ImageNet-pretrained ResNet-50 / ResNet-18 / EfficientNet-B0 with a new 1-logit head.
set_mode          which layers are trained:
                    frozen     backbone fixed, only the new head learns (pretrained network = feature extractor)
                    partial    last stage of the backbone + head
                    finetune   all layers, backbone at a lower learning rate than the head
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 2. Small CNN from scratch
# ---------------------------------------------------------------------------
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Five VGG-style blocks (32-64-128-256-256 channels), global average pooling, dropout, 1 logit.

    About 2.3 M parameters, ~10x fewer than ResNet-50. At 384 px input the last block sees
    12 x 12 positions, so Grad-CAM maps keep useful spatial detail.
    """

    def __init__(self, widths=(32, 64, 128, 256, 256), dropout=0.3):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers.append(conv_block(c, w)); c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def gradcam_layer(self):
        return self.features[-1]

    def forward_features(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(1)


# ---------------------------------------------------------------------------
# 3. Transfer learning
# ---------------------------------------------------------------------------
class TransferNet(nn.Module):
    """Pretrained backbone + new head. `backbone` returns a pooled feature vector."""

    def __init__(self, arch, pretrained=True, dropout=0.3):
        super().__init__()
        import torchvision.models as tvm
        self.arch = arch
        if arch in ("resnet18", "resnet50"):
            weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[arch] if pretrained else None
            net = getattr(tvm, arch)(weights=weights)
            n_feat = net.fc.in_features
            net.fc = nn.Identity()
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
            self.last_stage = net.layer4                 # trained in "partial" mode; Grad-CAM target
        elif arch == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            n_feat = net.classifier[1].in_features
            self.stem = net.features[:-2]
            self.last_stage = net.features[-2:]          # last MBConv stage + final 1x1 conv; Grad-CAM on the 1x1 conv
        else:
            raise ValueError(f"unknown arch {arch}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, 1))
        self.n_features = n_feat

    def gradcam_layer(self):
        """Last convolutional block: ResNet layer4[-1], EfficientNet features[-1]."""
        return self.last_stage[-1]

    def forward_features(self, x):
        return self.pool(self.last_stage(self.stem(x))).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x)).squeeze(1)


def build_model(name, pretrained=True):
    if name == "smallcnn":
        return SmallCNN()
    return TransferNet(name, pretrained=pretrained)


def set_mode(model, mode, lr, backbone_lr_mult=0.1, weight_decay=1e-4):
    """Freeze/unfreeze layers and return optimizer parameter groups.

    Frozen layers also have their BatchNorm statistics fixed (see `keep_frozen_bn_eval`):
    otherwise the ImageNet running means would drift to fundus images even though the
    weights are "frozen", and the frozen-vs-fine-tune comparison would not be clean.
    """
    if isinstance(model, SmallCNN) or mode == "scratch":
        for p in model.parameters():
            p.requires_grad = True
        return [dict(params=list(model.parameters()), lr=lr, weight_decay=weight_decay)]

    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    groups = [dict(params=list(model.head.parameters()), lr=lr, weight_decay=weight_decay)]
    if mode == "frozen":
        model.frozen_modules = [model.stem, model.last_stage]
    elif mode == "partial":
        for p in model.last_stage.parameters():
            p.requires_grad = True
        groups.append(dict(params=list(model.last_stage.parameters()), lr=lr * backbone_lr_mult, weight_decay=weight_decay))
        model.frozen_modules = [model.stem]
    elif mode == "finetune":
        for p in model.parameters():
            p.requires_grad = True
        backbone = list(model.stem.parameters()) + list(model.last_stage.parameters())
        groups.append(dict(params=backbone, lr=lr * backbone_lr_mult, weight_decay=weight_decay))
        model.frozen_modules = []
    else:
        raise ValueError(f"unknown mode {mode}")
    return groups


def keep_frozen_bn_eval(model):
    """Call after model.train(): put BatchNorm layers inside frozen modules back into eval mode."""
    for mod in getattr(model, "frozen_modules", []):
        for m in mod.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
