import torch
import torch.nn as nn

from src.model import create_model


def test_create_model_returns_module():
    model = create_model()
    assert isinstance(model, nn.Module)


def test_output_layer():
    model = create_model()
    assert model.fc.out_features == 10


def test_forward_pass():
    model = create_model()

    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    assert output.shape == (1, 10)