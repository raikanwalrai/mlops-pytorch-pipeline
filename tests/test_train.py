from src.train import load_config


def test_load_config():

    config = load_config()

    assert "dataset" in config
    assert "model" in config
    assert "training" in config

import torch
from src.train import get_device


def test_get_device():

    device = get_device()

    assert isinstance(device, torch.device)

import os
import torch
import tempfile

from src.train import save_checkpoint
from src.model import create_model


def test_save_checkpoint():

    model = create_model()

    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.TemporaryDirectory() as temp_dir:

        filepath = os.path.join(temp_dir, "checkpoint.pth")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            validation_loss=0.5,
            filepath=filepath,
        )

        assert os.path.exists(filepath)