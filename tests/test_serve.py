from PIL import Image

from fastapi.testclient import TestClient

from src.serve import (
    app,
    load_config,
    preprocess_image,
)

client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "model": "loaded",
    }


def test_load_config():

    config = load_config()

    assert "model" in config

    assert "dataset" in config

    assert "training" in config


def test_preprocess_image():

    image = Image.new("RGB", (96, 96))

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 3, 96, 96)