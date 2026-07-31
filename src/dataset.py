from torchvision import transforms
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader



def get_transforms():
    """
    Creates the image preprocessing pipeline for CIFAR-10.

    Returns:
        torchvision.transforms.Compose:
            A composed sequence of image transformations.
    """

    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    return transform

def load_dataset(data_dir="./data"):
    """
    Downloads and loads the CIFAR-10 dataset.

    Args:
        data_dir (str): Directory where the dataset will be stored.

    Returns:
        tuple:
            (train_dataset, test_dataset)
    """

    transform = get_transforms()

    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


from torch.utils.data import DataLoader, Subset

def create_dataloaders(
    batch_size=64,
    data_dir="./data",
    num_workers=2,
    subset_size=None,
):
    """
    Creates DataLoaders for the CIFAR-10 training and test datasets.

    Args:
        batch_size (int): Number of images per batch.
        data_dir (str): Directory containing the dataset.
        num_workers (int): Number of worker processes.
        subset_size (int | None): Number of training samples to use.
                                  If None, use the full dataset.

    Returns:
        tuple:
            (train_loader, test_loader)
    """
    #print(f"Subset size parameter: {subset_size}")
    train_dataset, test_dataset = load_dataset(data_dir)
    #print(f"Original training dataset size: {len(train_dataset)}")
    if subset_size is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(subset_size, len(train_dataset)))
        )
    #print(f"Training dataset after subset: {len(train_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    #print(f"Train loader batches: {len(train_loader)}")
    #print(f"Test loader batches : {len(test_loader)}")
    return train_loader, test_loader



    

    