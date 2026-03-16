"""
Model definitions for model training hands-on.

Provides:
- FCNet: Simple fully connected network for Iris classification
- OversizedFCNet: Intentionally large network to demonstrate overfitting
- ResNet18TransferModel: ResNet18-based transfer learning model for CIFAR-10
"""

import torch
import torch.nn as nn


class FCNet(nn.Module):
    """Simple fully connected network for tabular classification.

    Architecture: input -> hidden layers with ReLU -> output (logits)

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer sizes.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability (0.0 means no dropout).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: list[int] | None = None,
        num_classes: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.net(x)


class OversizedFCNet(nn.Module):
    """Intentionally oversized FC network to demonstrate overfitting.

    Much larger than necessary for Iris dataset.
    Use with a small subset of data to observe overfitting behavior.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    """

    def __init__(self, input_dim: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18TransferModel(nn.Module):
    """ResNet18-based transfer learning model for image classification.

    Supports two-phase training:
    1. Head-only: backbone frozen, only the classification head is trained.
    2. Fine-tuning: all parameters unfrozen and trained with a small lr.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as models

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the original classification head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # New classification head for target dataset
        self.head = nn.Linear(in_features, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head-only training mode)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (fine-tuning mode)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        features = self.backbone(x)
        return self.head(features)

    def get_num_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
