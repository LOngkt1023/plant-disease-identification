"""Image preprocessing utilities for plant disease identification."""

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Tuple

def resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize image to target size using padding to maintain aspect ratio."""
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create new image with black background
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    
    # Paste thumbnail into center
    upper = (target_size[0] - img.size[0]) // 2
    left = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (upper, left))
    
    return new_img

def get_train_transforms(target_size: int = 224):
    """Standard training augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transforms(target_size: int = 224):
    """Standard validation/test transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])