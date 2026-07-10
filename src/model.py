import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64x64 -> 32x32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32x32 -> 16x16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16x16 -> 8x8
        )
   
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x
    
    import torchvision.models as models


class TransferLearningModel(nn.Module):
    def __init__(self, num_classes: int = 10, backbone_name: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Phase 1: freeze everything except the final classifier layer."""
        for name, param in self.backbone.named_parameters():
            if "fc" in name or "classifier.1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 2):
        """Phase 2: unfreeze the last n conv blocks (in addition to the classifier)."""
        if self.backbone_name == "resnet18":
            # ResNet-18's conv blocks are named layer1, layer2, layer3, layer4
            unfreeze_layers = ["layer3", "layer4"][- n_blocks:] if n_blocks <= 2 else ["layer1", "layer2", "layer3", "layer4"]
            for name, param in self.backbone.named_parameters():
                if any(name.startswith(layer) for layer in unfreeze_layers) or "fc" in name:
                    param.requires_grad = True
        else:
            raise NotImplementedError("unfreeze_last_blocks currently implemented for resnet18 only")

    def get_embedding_extractor(self):
        """
        Returns a copy of the backbone with the classifier head stripped off,
        used in 04_change_detection.ipynb to extract 512-dim embeddings.
        """
        if self.backbone_name != "resnet18":
            raise NotImplementedError("Embedding extraction currently wired for resnet18 (512-dim) only")
        extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # drop final fc layer
        return extractor