"""Phân hệ lưu trữ Parquet để quản lý siêu dữ liệu (metadata) hiệu quả.

Sử dụng Apache Parquet (định dạng dạng cột - columnar format) để tối ưu hóa nén và hiệu suất truy vấn.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def save_metadata_to_parquet(
    metadata_df: pd.DataFrame,
    output_path: str = "data\\raw\\_metadata.parquet",
) -> None:
    """Lưu DataFrame siêu dữ liệu sang định dạng Apache Parquet.
    
    Các ưu điểm so với CSV:
      - Tỷ lệ nén tốt hơn (nhỏ hơn 50-80%)
      - Đọc/Ghi nhanh hơn
      - Giữ nguyên các kiểu dữ liệu của cột
      - Hỗ trợ các cấu trúc lồng nhau (nested structures)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Chuyển đổi sang bảng PyArrow
    table = pa.Table.from_pandas(metadata_df)

    # Ghi vào Parquet với thuật toán nén
    pq.write_table(
        table,
        str(output_path),
        compression="snappy",  # hoặc "gzip" để nén tốt hơn nữa
        use_dictionary=True,  # Bật mã hóa từ điển (dictionary encoding)
    )

    print(f"Siêu dữ liệu đã được lưu tại: {output_path}")
    print(f"Kích thước: {output_path.stat().st_size / (1024*1024):.2f} MB")


def load_metadata_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Tải siêu dữ liệu từ tệp Parquet."""
    parquet_path = Path(parquet_path)
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    return df


def export_metadata_to_csv(
    parquet_path: str,
    csv_output: str = "data\\raw\\_metadata.csv",
) -> None:
    """Xuất siêu dữ liệu Parquet sang CSV (để tương thích ngược)."""
    df = load_metadata_from_parquet(parquet_path)
    df.to_csv(csv_output, index=False)
    print(f"Đã xuất sang CSV: {csv_output}")


def create_dataset_manifest(
    metadata_parquet: str,
    output_manifest: str = "data\\.manifest",
) -> Dict:
    """Tạo tệp manifest của tập dữ liệu chứa các thông tin thống kê tóm tắt."""
    df = load_metadata_from_parquet(metadata_parquet)

    manifest = {
        "total_images": len(df),
        "classes": sorted(df["class"].unique().tolist()),
        "class_counts": df["class"].value_counts().to_dict(),
        "avg_image_size": {
            "width": float(df["width"].mean()),
            "height": float(df["height"].mean()),
        },
        "size_range": {
            "min_width": int(df["width"].min()),
            "max_width": int(df["width"].max()),
            "min_height": int(df["height"].min()),
            "max_height": int(df["height"].max()),
        },
        "metadata_path": metadata_parquet,
    }

    # Lưu manifest dưới dạng JSON
    import json

    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest đã được lưu tại: {output_manifest}")
    return manifest


def convert_csv_to_parquet(csv_path: str, output_path: str = None) -> None:
    """Chuyển đổi tệp siêu dữ liệu CSV hiện có sang định dạng Parquet."""
    if output_path is None:
        output_path = csv_path.replace(".csv", ".parquet")

    df = pd.read_csv(csv_path)
    save_metadata_to_parquet(df, output_path)
    print(f"Đã chuyển đổi {csv_path} -> {output_path}")


def get_dataset_stats(metadata_parquet: str) -> Dict:
    """Lấy số liệu thống kê toàn diện của tập dữ liệu từ siêu dữ liệu Parquet."""
    df = load_metadata_from_parquet(metadata_parquet)

    stats = {
        "total_images": len(df),
        "valid_images": len(df[df["is_valid"] == True]) if "is_valid" in df.columns else len(df),
        "total_size_mb": df["size_bytes"].sum() / (1024 * 1024),
        "classes": {
            class_name: {
                "count": int(count),
                "avg_width": float(df[df["class"] == class_name]["width"].mean()),
                "avg_height": float(df[df["class"] == class_name]["height"].mean()),
                "avg_size_kb": float(df[df["class"] == class_name]["size_bytes"].mean() / 1024),
            }
            for class_name, count in df["class"].value_counts().items()
        },
    }

    return stats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Quản lý lưu trữ siêu dữ liệu Parquet")
    parser.add_argument("--convert", help="Chuyển đổi CSV sang Parquet")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stats", help="Hiển thị thống kê từ Parquet")
    parser.add_argument("--manifest", help="Tạo manifest từ Parquet")
    args = parser.parse_args()

    if args.convert:
        convert_csv_to_parquet(args.convert, args.output)
    elif args.stats:
        stats = get_dataset_stats(args.stats)
        print(json.dumps(stats, indent=2))
    elif args.manifest:
        manifest = create_dataset_manifest(args.manifest)
        print(json.dumps(manifest, indent=2))

