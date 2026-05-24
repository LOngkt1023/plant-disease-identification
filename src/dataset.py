"""Dataset and DataLoader utilities for Plant Disease Identification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from .config.disease_classes import CLASS_NAMES
from .preprocessing import get_train_transforms, get_val_test_transforms


def _validate_class_mapping(class_to_idx: Dict[str, int]) -> None:
    """Ensure discovered classes exactly match config class order."""
    discovered = set(class_to_idx.keys())
    expected = set(CLASS_NAMES)
    if discovered != expected:
        missing = sorted(expected - discovered)
        extra = sorted(discovered - expected)
        raise ValueError(
            f"Class mismatch between dataset folder and config.\n"
            f"Missing classes: {missing}\n"
            f"Unexpected classes: {extra}"
        )


def build_datasets(
    splits_dir: str = "dataset_v2/splits",
    image_size: int = 224,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """Build train/val/test datasets from split folders."""
    root = Path(splits_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Split folders not found under {root}. "
            f"Expected: train/, val/, test/."
        )

    train_ds = datasets.ImageFolder(str(train_dir), transform=get_train_transforms(image_size))
    val_ds = datasets.ImageFolder(str(val_dir), transform=get_val_test_transforms(image_size))
    test_ds = datasets.ImageFolder(str(test_dir), transform=get_val_test_transforms(image_size))

    _validate_class_mapping(train_ds.class_to_idx)
    _validate_class_mapping(val_ds.class_to_idx)
    _validate_class_mapping(test_ds.class_to_idx)

    return train_ds, val_ds, test_ds


def build_dataloaders(
    splits_dir: str = "dataset_v2/splits",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    train_ds, val_ds, test_ds = build_datasets(splits_dir=splits_dir, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }