from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import (
    get_transforms,
    load_dataset,
    create_dataloaders
)


def test_get_transforms_returns_compose():
    transform = get_transforms()

    assert isinstance(transform, transforms.Compose)


def test_load_dataset_sizes():
    train_dataset, test_dataset = load_dataset()

    assert len(train_dataset) == 50000
    assert len(test_dataset) == 10000


def test_create_dataloaders_returns_dataloaders():
    train_loader, test_loader = create_dataloaders()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)


def test_batch_shape():
    train_loader, _ = create_dataloaders()

    images, labels = next(iter(train_loader))

    assert images.shape == (64, 3, 224, 224)
    assert labels.shape == (64,)