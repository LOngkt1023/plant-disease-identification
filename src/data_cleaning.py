"""Phân hệ làm sạch và xác thực dữ liệu.

Xử lý:
  - Xác thực URL và loại bỏ trùng lặp
  - Phát hiện tệp lỗi hỏng
  - Phát hiện ảnh trùng lặp (bằng mã băm hash)
  - Làm sạch siêu dữ liệu (metadata)
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm


def compute_image_hash(img_path: Path, method: str = "md5") -> str:
    """Tính toán mã băm của tệp ảnh để phát hiện ảnh trùng lặp."""
    hasher = hashlib.md5() if method == "md5" else hashlib.sha256()
    try:
        with open(img_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Lỗi khi tính mã băm cho {img_path}: {e}")
        return None


def validate_image_file(img_path: Path) -> Tuple[bool, str]:
    """Xác thực tính toàn vẹn của tệp ảnh.
    
    Trả về: (is_valid, lý do)
    """
    if not img_path.exists():
        return False, "Không tìm thấy tệp"

    if img_path.stat().st_size < 1024:
        return False, "Tệp quá nhỏ (<1KB)"

    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, "OK"
    except Exception as e:
        return False, f"Ảnh lỗi: {str(e)[:50]}"


def scan_and_clean_dataset(
    data_root: str = "data\\raw",
    output_metadata: str = "data\\raw\\_metadata.csv",
) -> pd.DataFrame:
    """Quét tất cả các ảnh, xác thực và phát hiện ảnh trùng lặp.
    
    Trả về DataFrame chứa siêu dữ liệu ảnh và kết quả xác thực.
    """
    data_root = Path(data_root)
    records = []

    # Quét qua tất cả các lớp phân loại
    for class_dir in data_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\nĐang quét {class_name}...")

        for img_file in tqdm(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")), desc=class_name):
            is_valid, reason = validate_image_file(img_file)
            img_hash = compute_image_hash(img_file) if is_valid else None

            record = {
                "class": class_name,
                "filename": img_file.name,
                "path": str(img_file),
                "size_bytes": img_file.stat().st_size,
                "is_valid": is_valid,
                "validation_reason": reason,
                "hash_md5": img_hash,
                "width": None,
                "height": None,
            }

            # Lấy kích thước ảnh
            if is_valid:
                try:
                    with Image.open(img_file) as img:
                        record["width"], record["height"] = img.size
                except:
                    pass

            records.append(record)

    df = pd.DataFrame(records)

    # Phát hiện ảnh trùng lặp
    print("\nĐang phát hiện ảnh trùng lặp...")
    df["is_duplicate"] = df.groupby("class")["hash_md5"].transform(
        lambda x: x.duplicated(keep="first")
    )

    # Loại bỏ ảnh trùng lặp và ảnh lỗi hỏng
    print("\nĐang dọn dẹp...")
    removed_count = 0

    for idx, row in df[df["is_duplicate"] | ~df["is_valid"]].iterrows():
        try:
            Path(row["path"]).unlink()
            removed_count += 1
            print(f"  Đã xóa: {row['filename']} ({row['validation_reason']})")
        except Exception as e:
            print(f"  Lỗi khi xóa {row['filename']}: {e}")

    # Lưu siêu dữ liệu đã làm sạch
    df_clean = df[~(df["is_duplicate"] | ~df["is_valid"])].copy()
    df_clean.to_csv(output_metadata, index=False)

    print(f"\n{'='*60}")
    print(f"Hoàn thành làm sạch:")
    print(f"  Tổng số: {len(df)}")
    print(f"  Hợp lệ: {len(df[df['is_valid']])}")
    print(f"  Đã xóa (lỗi/trùng lặp): {removed_count}")
    print(f"  Đã lưu tại: {output_metadata}")
    print(f"{'='*60}")

    return df_clean


def get_class_distribution(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Lấy số lượng ảnh của từng lớp phân loại."""
    return metadata_df["class"].value_counts().to_frame(name="count")


def get_image_size_stats(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Lấy số liệu thống kê về chiều rộng/chiều cao cho từng lớp phân loại."""
    return (
        metadata_df.groupby("class")[["width", "height"]]
        .agg(["min", "max", "mean", "std"])
        .round(0)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Làm sạch và xác thực tập dữ liệu")
    parser.add_argument("--data-root", default="data\\raw")
    parser.add_argument("--output", default="data\\raw\\_metadata.csv")
    args = parser.parse_args()

    df = scan_and_clean_dataset(args.data_root, args.output)
    print("\nPhân bổ lớp dữ liệu:")
    print(get_class_distribution(df))
    print("\nThống kê kích thước ảnh:")
    print(get_image_size_stats(df))

