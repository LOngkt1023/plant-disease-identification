"""Model definitions for plant disease classification."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights

try:
    from .config.disease_classes import NUM_CLASSES
except (ImportError, ValueError):
    from config.disease_classes import NUM_CLASSES


def get_resnet50(num_classes: Optional[int] = None, pretrained: bool = True) -> nn.Module:
    """Build ResNet50 classifier with output dimension = num_classes."""
    final_num_classes = num_classes or NUM_CLASSES
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, final_num_classes)
    return model


def get_mobilenet_v2(num_classes: Optional[int] = None, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV2 classifier with output dimension = num_classes."""
    final_num_classes = num_classes or NUM_CLASSES
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, final_num_classes)
    return model


def build_model(model_name: str, num_classes: Optional[int] = None, pretrained: bool = True) -> nn.Module:
    """Factory for supported models."""
    name = model_name.strip().lower()
    final_num_classes = num_classes or NUM_CLASSES

    if name == "resnet50":
        return get_resnet50(num_classes=final_num_classes, pretrained=pretrained)
    if name in {"mobilenet_v2", "mobilenetv2"}:
        return get_mobilenet_v2(num_classes=final_num_classes, pretrained=pretrained)

    raise ValueError(f"Unsupported model_name='{model_name}'. Use 'resnet50' or 'mobilenet_v2'.")