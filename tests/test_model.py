import torch

from src.model import SimpleClassifier


def test_model_forward():
    model = SimpleClassifier()

    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()