import torch

from src.model import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
