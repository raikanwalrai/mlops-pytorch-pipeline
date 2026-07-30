import torch.nn as nn
from torchvision import models


def create_model(num_classes=10, pretrained=True):
    """
    Creates and returns a ResNet18 model for image classification.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        torch.nn.Module: Configured ResNet18 model.
    """

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model