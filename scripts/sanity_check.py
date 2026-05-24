"""Sanity checks for refactored Plant Disease Identification pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config.disease_classes import (
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    PLANT_DISEASE_CLASSES,
)
try:
    import torch
    from src.models import get_resnet50, get_mobilenet_v2

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    get_resnet50 = None
    get_mobilenet_v2 = None
    TORCH_AVAILABLE = False

OLD_CLASSES = [
    "Coffee_Healthy",
    "Coffee_Rust",
    "Citrus_Canker",
    "Citrus_Greening",
    "Tomato_Curl",
    "Tomato_Blight",
    "Rice_Blight",
]

CORE_FILES_TO_SCAN = [
    "src/config/disease_classes.py",
    "src/models.py",
    "src/dataset.py",
    "scripts/crawl.py",
    "scripts/preprocess.py",
    "scripts/split_dataset.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "README.md",
    "SETUP_GUIDE.md",
    "notebooks/02_eda_preprocessing.ipynb",
]


def check_config():
    print("Checking disease class config...")
    assert NUM_CLASSES == 16, f"NUM_CLASSES must be 16, got {NUM_CLASSES}"
    assert len(CLASS_NAMES) == 16, f"CLASS_NAMES length must be 16, got {len(CLASS_NAMES)}"
    assert set(CLASS_TO_IDX.keys()) == set(CLASS_NAMES), "CLASS_TO_IDX keys mismatch"
    assert set(IDX_TO_CLASS.values()) == set(CLASS_NAMES), "IDX_TO_CLASS values mismatch"

    for i, cls in enumerate(CLASS_NAMES):
        assert CLASS_TO_IDX[cls] == i, f"class index mismatch for {cls}"
        assert IDX_TO_CLASS[i] == cls, f"reverse class index mismatch for {cls}"
        assert cls in PLANT_DISEASE_CLASSES, f"{cls} missing in PLANT_DISEASE_CLASSES"
        kws = PLANT_DISEASE_CLASSES[cls].get("keywords", [])
        assert isinstance(kws, list) and len(kws) > 0, f"{cls} must have non-empty keywords"

    print("Config check passed.")


def check_required_scripts():
    print("Checking required scripts...")
    required = [
        Path("scripts/data_statistics.py"),
        Path("scripts/extract_features.py"),
        Path("scripts/sanity_check.py"),
    ]
    for p in required:
        assert p.exists(), f"Missing required script: {p}"
    print("Required scripts check passed.")


def check_dataset_structure():
    print("Checking dataset_v2 folder structure...")
    base = Path("dataset_v2")
    required = [
        base / "raw",
        base / "clean",
        base / "splits" / "train",
        base / "splits" / "val",
        base / "splits" / "test",
    ]
    for p in required:
        if not p.exists():
            print(f"WARNING: Missing folder: {p}")

    for split in ["train", "val", "test"]:
        split_dir = base / "splits" / split
        if split_dir.exists():
            missing = [c for c in CLASS_NAMES if not (split_dir / c).exists()]
            if missing:
                print(f"WARNING: Missing class folders in {split}: {missing}")
            else:
                print(f"{split} contains all 16 class folders.")
    print("Dataset structure check done.")


def check_model_shapes():
    print("Checking model output shapes...")
    if not TORCH_AVAILABLE:
        print("WARNING: torch not installed, skip model shape check.")
        return

    x = torch.randn(2, 3, 224, 224)

    resnet = get_resnet50(num_classes=NUM_CLASSES, pretrained=False)
    y1 = resnet(x)
    assert tuple(y1.shape) == (2, 16), f"ResNet50 output shape must be [2,16], got {tuple(y1.shape)}"

    mobilenet = get_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
    y2 = mobilenet(x)
    assert tuple(y2.shape) == (2, 16), f"MobileNetV2 output shape must be [2,16], got {tuple(y2.shape)}"

    print("Model shape check passed.")


def check_split_hash_overlap():
    print("Checking md5 hash overlap across train/val/test metadata...")
    split_meta = Path("dataset_v2/metadata/split_metadata.parquet")
    if not split_meta.exists():
        print("WARNING: split_metadata.parquet not found, skip hash overlap check.")
        return

    import pandas as pd

    df = pd.read_parquet(split_meta)
    required_cols = {"split", "md5_hash"}
    if not required_cols.issubset(df.columns):
        print("WARNING: split metadata missing split/md5_hash columns, skip overlap check.")
        return

    train_hashes = set(df[df["split"] == "train"]["md5_hash"].dropna().astype(str))
    val_hashes = set(df[df["split"] == "val"]["md5_hash"].dropna().astype(str))
    test_hashes = set(df[df["split"] == "test"]["md5_hash"].dropna().astype(str))

    tv = train_hashes & val_hashes
    tt = train_hashes & test_hashes
    vt = val_hashes & test_hashes

    assert len(tv) == 0 and len(tt) == 0 and len(vt) == 0, (
        f"Found md5 overlap: train-val={len(tv)}, train-test={len(tt)}, val-test={len(vt)}"
    )
    print("No md5 overlap across splits.")


def check_readme_content():
    print("Checking README content...")
    readme = Path("README.md")
    assert readme.exists(), "README.md not found"

    text = readme.read_text(encoding="utf-8", errors="ignore")
    assert "dataset_v2" in text, "README must mention dataset_v2"

    for cls in CLASS_NAMES:
        assert cls in text, f"README missing class: {cls}"

    print("README content check passed.")


def check_no_old_classes_in_core():
    print("Checking old class names are removed from core/docs files...")
    violations = []

    for file_path in CORE_FILES_TO_SCAN:
        p = Path(file_path)
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        for old_cls in OLD_CLASSES:
            if old_cls in content:
                violations.append((file_path, old_cls))

    assert not violations, f"Found old classes in files: {violations}"
    print("Old class cleanup check passed.")


def main():
    check_config()
    check_required_scripts()
    check_dataset_structure()
    check_model_shapes()
    check_split_hash_overlap()
    check_readme_content()
    check_no_old_classes_in_core()
    print("\nAll sanity checks completed.")


if __name__ == "__main__":
    main()