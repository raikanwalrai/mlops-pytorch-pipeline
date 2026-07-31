import yaml
import torch
import os
import torch.nn as nn
from src.dataset import create_dataloaders
from src.model import create_model


import torch.optim as optim

def load_config(config_path="configs/config.yaml"):
    """
    Loads training configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def get_device():
    """
    Returns the device to be used for training.

    Returns:
        torch.device
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Trains the model for one epoch.

    Args:
        model: Neural network model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU.

    Returns:
        float: Average training loss.
    """

    model.train()
    running_loss = 0.0

    #print("Entered train_one_epoch()")
    for batch_idx, (images, labels) in enumerate(dataloader):

        # if batch_idx % 100 == 0:
        #     print(f"Processing batch {batch_idx}/{len(dataloader)}")

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)

    return average_loss




def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluates the model on the validation dataset.

    Returns:
        tuple:
            (average_loss, accuracy)
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    average_loss = running_loss / len(dataloader)

    accuracy = 100 * correct / total

    return average_loss, accuracy




def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_loss,
    filepath,
):
    """
    Saves a training checkpoint.

    Args:
        model: Trained model.
        optimizer: Optimizer.
        epoch: Current epoch.
        validation_loss: Validation loss.
        filepath: Destination file.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved to: {filepath}")


def main():
    """
    Main training pipeline.
    """

    # Load configuration
    config = load_config()


    checkpoint_dir = config["checkpoint"]["save_dir"]
    checkpoint_file = config["checkpoint"]["filename"]

    checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint_file,
    )
    # Select device
    device = get_device()

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        batch_size=config["dataset"]["batch_size"],
        data_dir=config["dataset"]["data_dir"],
        num_workers=config["dataset"]["num_workers"],
        subset_size=config["dataset"].get("subset_size"),
    )

    print(f"Training batches : {len(train_loader)}")
    print(f"Testing batches  : {len(test_loader)}")

    # Create model
    model = create_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    # Move model to device
    model = model.to(device)

    print("\nModel initialized successfully.")
    print(f"Running on device: {device}")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    print("\nLoss function initialized.")
    print(f"Criterion: {criterion}")

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    print("\nOptimizer initialized.")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning Rate: {config['training']['learning_rate']}")


    # Number of training epochs
    epochs = config["training"]["epochs"]

    best_validation_loss = float("inf")

    patience = config["early_stopping"]["patience"]

    patience_counter = 0

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {validation_loss:.4f} "
            f"Validation Accuracy: {validation_accuracy:.2f}%"
        )
       
        if patience_counter >= patience:

            print("\nEarly stopping triggered.")

            break
   
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                validation_loss=validation_loss,
                filepath=checkpoint_path,
            )

            print("Best model updated.")

        else:

            patience_counter += 1

            print(
                f"No improvement for {patience_counter} epoch(s)."
            )
            


if __name__ == "__main__":
    main()