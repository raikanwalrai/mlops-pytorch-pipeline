import io

import torch
import yaml

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import create_model

# -----------------------------
# FastAPI Application
# -----------------------------
app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="Inference API for a trained ResNet18 model",
    version="1.0.0",
)
# -----------------------------
# Configuration
# -----------------------------
def load_config():
    """Load the application configuration from the YAML file."""

    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config
# -----------------------------
# Model Loading
# -----------------------------
def load_model(config):
    """Create the model and load the trained checkpoint."""

    model = create_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    checkpoint_path = (
        f'{config["checkpoint"]["save_dir"]}/'
        f'{config["checkpoint"]["filename"]}'
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model

# Image preprocessing pipeline
transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ]
)


# CIFAR-10 class labels

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image: Image.Image):
    """Preprocess an uploaded image for model inference."""

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    return image


# -----------------------------
# Application Initialization
# -----------------------------
config = load_config()
print("Configuration loaded successfully.")

model = load_model(config)
print("Model loaded successfully.")




# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/health")
def health_check():
    """Check if the API and model are available."""
    return {
        "status": "healthy",
        "model": "loaded",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the CIFAR-10 class of an uploaded image."""

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "class_id": predicted_class,
        "confidence": round(confidence, 4),
    }