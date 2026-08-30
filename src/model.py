import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small CNN for CIFAR-10 image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(
    architecture: str = "simple_cnn",
    num_classes: int = 10
) -> nn.Module:

    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    raise ValueError(f"Unsupported architecture: {architecture}")

# Backward-compatible name used by the project tests
SimpleClassifier = SimpleCNN