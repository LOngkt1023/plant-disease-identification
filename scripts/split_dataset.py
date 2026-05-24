"""Stratified train/val/test split for plant disease dataset."""

import argparse
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

sys.path.append(str(Path(__file__).parent.parent))

from src.config.disease_classes import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test split of clean dataset.")
    parser.add_argument("--clean-dir", type=str, default="dataset_v2/clean", help="Path to clean images.")
    parser.add_argument("--output-dir", type=str, default="dataset_v2/splits", help="Path to output splits.")
    parser.add_argument("--metadata-dir", type=str, default="dataset_v2/metadata", help="Path to metadata directory.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    clean_dir = Path(args.clean_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STRATIFIED DATASET SPLIT")
    print(f"Train: {args.train_ratio*100:.0f}% | Val: {args.val_ratio*100:.0f}% | Test: {args.test_ratio*100:.0f}%")
    print("=" * 60)

    # Try to load existing clean metadata for md5 hash deduplication
    clean_meta_path = metadata_dir / "clean_metadata.parquet"
    hash_map = {}  # md5_hash -> image_path
    if clean_meta_path.exists():
        try:
            meta_df = pd.read_parquet(clean_meta_path)
            if "md5_hash" in meta_df.columns:
                for _, row in meta_df.iterrows():
                    h = row.get("md5_hash", "")
                    p = row.get("clean_path", "")
                    if h and h not in hash_map:
                        hash_map[h] = p
        except Exception as e:
            print(f"Warning: Could not load clean metadata for hash dedup: {e}")

    # Gather all images with their class labels
    records = []
    for class_name in CLASS_NAMES:
        class_clean_dir = clean_dir / class_name
        if not class_clean_dir.exists():
            print(f"WARNING: clean/{class_name} not found. Skipping.")
            continue
        img_files = sorted(
            list(class_clean_dir.glob("*.jpg")) +
            list(class_clean_dir.glob("*.jpeg")) +
            list(class_clean_dir.glob("*.png"))
        )
        for img_path in img_files:
            records.append({"image_path": str(img_path), "class_name": class_name})

    if not records:
        print("ERROR: No clean images found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"\nTotal clean images found: {len(df)}")
    print(df["class_name"].value_counts())

    # Compute md5 if available and filter out cross-class duplicates
    from src.utils import compute_md5
    print("\nComputing MD5 hashes for deduplication...")
    seen_hashes = set()
    deduped = []
    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        h = compute_md5(img_path)
        if h is None or h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append({"image_path": row["image_path"], "class_name": row["class_name"], "md5_hash": h})

    df = pd.DataFrame(deduped)
    print(f"After deduplication: {len(df)} unique images.")

    # Stratified split: First split off test, then split remaining into train/val
    # Relative val ratio from non-test portion
    val_relative = args.val_ratio / (args.train_ratio + args.val_ratio)

    split_records = []

    # Split per class to ensure stratification
    train_list, val_list, test_list = [], [], []
    for class_name in CLASS_NAMES:
        class_df = df[df["class_name"] == class_name]
        if len(class_df) < 3:
            print(f"WARNING: {class_name} has <3 images, cannot split. Assigning all to train.")
            for _, row in class_df.iterrows():
                train_list.append({**row.to_dict(), "split": "train"})
            continue

        # First split: train+val / test
        try:
            trainval_df, test_df = train_test_split(
                class_df, test_size=args.test_ratio, random_state=args.seed, shuffle=True
            )
        except ValueError:
            # If test split would be 0 items, put 1 in test
            test_df = class_df.iloc[:1]
            trainval_df = class_df.iloc[1:]

        # Second split: train / val
        if len(trainval_df) < 2:
            train_df = trainval_df
            val_df = pd.DataFrame(columns=class_df.columns)
        else:
            try:
                train_df, val_df = train_test_split(
                    trainval_df, test_size=val_relative, random_state=args.seed, shuffle=True
                )
            except ValueError:
                val_df = trainval_df.iloc[:1]
                train_df = trainval_df.iloc[1:]

        for _, row in train_df.iterrows():
            train_list.append({**row.to_dict(), "split": "train"})
        for _, row in val_df.iterrows():
            val_list.append({**row.to_dict(), "split": "val"})
        for _, row in test_df.iterrows():
            test_list.append({**row.to_dict(), "split": "test"})

    split_records = train_list + val_list + test_list
    split_df = pd.DataFrame(split_records)

    # Validate: no md5 collision between splits
    train_hashes = set(split_df[split_df["split"] == "train"]["md5_hash"])
    val_hashes = set(split_df[split_df["split"] == "val"]["md5_hash"])
    test_hashes = set(split_df[split_df["split"] == "test"]["md5_hash"])

    tv_overlap = train_hashes & val_hashes
    tt_overlap = train_hashes & test_hashes
    vt_overlap = val_hashes & test_hashes

    if tv_overlap or tt_overlap or vt_overlap:
        print(f"WARNING: Hash overlaps found! Train∩Val={len(tv_overlap)}, Train∩Test={len(tt_overlap)}, Val∩Test={len(vt_overlap)}")
    else:
        print("Hash deduplication across splits: OK (no overlaps)")

    # Copy images to split directories and build paths
    print("\nCopying images to split directories...")
    final_records = []
    for _, row in split_df.iterrows():
        img_path = Path(row["image_path"])
        class_name = row["class_name"]
        split = row["split"]
        filename = img_path.name

        dest_dir = output_dir / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)

        final_records.append({
            **row.to_dict(),
            "split_path": str(dest_path)
        })

    final_df = pd.DataFrame(final_records)

    # Save split metadata
    metadata_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(metadata_dir / "split_metadata.parquet", index=False)
    final_df.to_csv(metadata_dir / "split_metadata.csv", index=False, encoding="utf-8")
    print(f"Saved split metadata: {metadata_dir / 'split_metadata.parquet'}")

    # Print statistics
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        split_data = final_df[final_df["split"] == split]
        print(f"\n[{split.upper()}] Total: {len(split_data)}")
        counts = split_data["class_name"].value_counts()
        for class_name in CLASS_NAMES:
            c = counts.get(class_name, 0)
            print(f"  {class_name:35s}: {c:5d}")

    total = len(final_df)
    n_train = len(train_list)
    n_val = len(val_list)
    n_test = len(test_list)
    print(f"\nTotal: {total} | Train: {n_train} ({n_train/total*100:.1f}%) | Val: {n_val} ({n_val/total*100:.1f}%) | Test: {n_test} ({n_test/total*100:.1f}%)")
    print("\nSplit complete!")


if __name__ == "__main__":
    main()