"""Phân hệ tiền xử lý ảnh.

Xử lý:
  - Thay đổi kích thước ảnh về kích thước chuẩn (224x224)
  - Lọc Gaussian (giảm nhiễu)
  - Chuẩn hóa dữ liệu
  - Thiết lập tăng cường dữ liệu (Data augmentation)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def resize_image(img_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Thay đổi kích thước ảnh về kích thước mục tiêu trong khi vẫn giữ nguyên tỷ lệ khung hình (có thêm padding)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Tính toán tỷ lệ khung hình
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    if aspect_ratio > target_aspect:
        # Thay đổi chiều rộng theo mục tiêu, thêm padding cho chiều cao
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Thay đổi chiều cao theo mục tiêu, thêm padding cho chiều rộng
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    # Thay đổi kích thước ảnh
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tạo khung vẽ canvas với phần đệm padding
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128  # Padding màu xám
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized

    return canvas


def apply_gaussian_filter(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Áp dụng bộ lọc Gaussian để giảm nhiễu trong khi vẫn giữ lại các cạnh chi tiết."""
    # Áp dụng làm mờ Gaussian (Gaussian blur)
    img_blurred = cv2.GaussianBlur(img, (5, 5), sigma)

    # Trộn với ảnh gốc để giữ lại các cạnh chi tiết
    img_filtered = cv2.addWeighted(img, 0.7, img_blurred, 0.3, 0)

    return img_filtered.astype(np.uint8)


def normalize_image(img: np.ndarray, method: str = "imagenet") -> np.ndarray:
    """Chuẩn hóa các giá trị pixel của ảnh.
    
    Các phương pháp:
      - "imagenet": Sử dụng giá trị trung bình/độ lệch chuẩn của ImageNet
      - "minmax": Co giãn về khoảng [0, 1]
      - "zscore": Trung bình bằng 0, phương sai bằng 1
    """
    img = img.astype(np.float32)

    if method == "imagenet":
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        img = (img - mean) / std
    elif method == "minmax":
        img = img / 255.0
    elif method == "zscore":
        img = (img - img.mean()) / (img.std() + 1e-8)

    return img


def preprocess_image(
    img_path: Path,
    output_path: Path,
    target_size: Tuple[int, int] = (224, 224),
    apply_filter: bool = True,
    normalize: bool = True,
 ) -> bool:
    """Tiền xử lý một ảnh đơn lẻ và lưu vào thư mục đầu ra."""
    try:
        # Thay đổi kích thước
        img = resize_image(img_path, target_size)
        if img is None:
            return False

        # Áp dụng bộ lọc Gaussian
        if apply_filter:
            img = apply_gaussian_filter(img, sigma=1.0)

        # Chuẩn hóa
        if normalize:
            img = normalize_image(img, method="minmax")
            img = (img * 255).astype(np.uint8)

        # Lưu ảnh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

        return True

    except Exception as e:
        print(f"Lỗi khi tiền xử lý ảnh {img_path}: {e}")
        return False


def preprocess_dataset(
    data_root: str = "data\\raw",
    output_root: str = "data\\processed",
    target_size: Tuple[int, int] = (224, 224),
    apply_filter: bool = True,
    normalize: bool = True,
) -> Dict[str, int]:
    """Tiền xử lý toàn bộ tập dữ liệu.
    
    Trả về số liệu thống kê về các ảnh đã được xử lý.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)

    stats = {}

    # Xử lý từng lớp phân loại
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = output_root / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        failed = 0

        print(f"\nĐang xử lý {class_name}...")
        for img_file in tqdm(
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png")),
            desc=class_name,
        ):
            output_file = output_class_dir / img_file.name

            success = preprocess_image(
                img_file,
                output_file,
                target_size=target_size,
                apply_filter=apply_filter,
                normalize=normalize,
            )

            if success:
                processed += 1
            else:
                failed += 1

        stats[class_name] = {
            "processed": processed,
            "failed": failed,
            "total": processed + failed,
        }

    print(f"\n{'='*60}")
    print("Hoàn thành tiền xử lý:")
    for class_name, class_stats in stats.items():
        print(
            f"  {class_name}: {class_stats['processed']}/{class_stats['total']} "
            f"({class_stats['failed']} ảnh lỗi)"
        )
    print(f"{'='*60}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tiền xử lý ảnh trong tập dữ liệu")
    parser.add_argument("--input", default="data\\raw")
    parser.add_argument("--output", default="data\\processed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        data_root=args.input,
        output_root=args.output,
        target_size=(args.size, args.size),
        apply_filter=not args.no_filter,
        normalize=not args.no_normalize,
    )

