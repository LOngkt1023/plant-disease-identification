"""
Tiện ích chuyển đổi metadata từ JSONL sang định dạng Parquet.

Parquet là định dạng cột (columnar format) tối ưu cho lưu trữ và xử lý dữ liệu lớn:
  - Ép nén dữ liệu tốt hơn (tiết kiệm 70-80% dung lượng so với JSONL)
  - Hỗ trợ "Predicate Pushdown" (chỉ tải cột cần thiết, không cần đọc toàn bộ file)
  - Tương thích tốt với Pandas, Polars, DuckDB, và các ML frameworks

Cách sử dụng:
  python src/convert_metadata.py \
    --input data/raw/Rice_Healthy/metadata.jsonl \
    --output data/raw/Rice_Healthy/metadata.parquet
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import argparse

# ── Cấu hình Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def read_jsonl(jsonl_path: Path) -> List[Dict]:
    """
    Đọc file JSONL và trả về danh sách các dict.
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as exc:
                log.warning("Dòng %d bị lỗi JSON, bỏ qua: %s", i + 1, exc)
                continue
    return records


def jsonl_to_parquet(
    input_path: str,
    output_path: Optional[str] = None,
    compression: str = "snappy",
) -> None:
    """
    Chuyển đổi JSONL → Parquet.
    
    Args:
        input_path: Đường dẫn file JSONL
        output_path: Đường dẫn file Parquet (mặc định: thay .jsonl → .parquet)
        compression: Loại nén ("snappy", "gzip", "brotli", hoặc None)
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("Cần cài đặt pandas: pip install pandas")
        return
    
    try:
        import pyarrow.parquet  # type: ignore
    except ImportError:
        log.error("Cần cài đặt pyarrow: pip install pyarrow")
        return
    
    input_path = Path(input_path)
    if not input_path.exists():
        log.error("File JSONL không tồn tại: %s", input_path)
        return
    
    if output_path is None:
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)
    
    log.info("Đọc JSONL từ: %s", input_path)
    records = read_jsonl(input_path)
    
    if not records:
        log.warning("Không có record nào trong JSONL")
        return
    
    log.info("Đã đọc %d record, đang chuyển thành DataFrame...", len(records))
    df = pd.DataFrame(records)
    
    log.info(
        "Kích thước JSONL: %.2f MB → Parquet (%s): ~%.2f MB (ước tính)",
        input_path.stat().st_size / 1024 / 1024,
        compression or "không nén",
        df.memory_usage(deep=True).sum() / 1024 / 1024 * 0.2,  # Ước tính
    )
    
    log.info("Ghi Parquet: %s", output_path)
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression=compression,
        index=False,
    )
    
    log.info(
        "✓ Chuyển đổi thành công: %d dòng, %d cột → %s (%.2f MB)",
        len(df),
        len(df.columns),
        output_path,
        output_path.stat().st_size / 1024 / 1024,
    )


def jsonl_to_parquet_batch(
    input_dir: str,
    pattern: str = "*/metadata.jsonl",
    compression: str = "snappy",
) -> None:
    """
    Chuyển đổi hàng loạt tất cả file metadata.jsonl trong thư mục.
    
    Args:
        input_dir: Thư mục gốc chứa các thư mục lớp
        pattern: Glob pattern để tìm file JSONL (mặc định: */metadata.jsonl)
        compression: Loại nén
    """
    input_dir = Path(input_dir)
    jsonl_files = sorted(input_dir.glob(pattern))
    
    if not jsonl_files:
        log.warning("Không tìm thấy file JSONL nào khớp với pattern: %s/%s", input_dir, pattern)
        return
    
    log.info("Tìm thấy %d file JSONL cần chuyển đổi", len(jsonl_files))
    
    for jsonl_file in jsonl_files:
        output_file = jsonl_file.with_name("metadata.parquet")
        if output_file.exists():
            log.info("⊘ %s đã tồn tại, bỏ qua", output_file.name)
            continue
        
        log.info("[%s] Chuyển đổi...", jsonl_file.parent.name)
        jsonl_to_parquet(str(jsonl_file), str(output_file), compression)


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi JSONL → Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chuyển đổi một file
  python src/convert_metadata.py \\
    --input data/raw/Rice_Healthy/metadata.jsonl \\
    --output data/raw/Rice_Healthy/metadata.parquet

  # Chuyển đổi tất cả file metadata.jsonl trong thư mục
  python src/convert_metadata.py \\
    --batch data/raw/ \\
    --compression snappy
        """,
    )
    
    parser.add_argument(
        "--input",
        help="Đường dẫn file JSONL input",
    )
    parser.add_argument(
        "--output",
        help="Đường dẫn file Parquet output (tùy chọn, mặc định: thay .jsonl → .parquet)",
    )
    parser.add_argument(
        "--batch",
        help="Thư mục để chuyển đổi hàng loạt tất cả metadata.jsonl",
    )
    parser.add_argument(
        "--compression",
        choices=["snappy", "gzip", "brotli", "none"],
        default="snappy",
        help="Loại nén (mặc định: snappy)",
    )
    
    args = parser.parse_args()
    
    compression = None if args.compression == "none" else args.compression
    
    if args.batch:
        jsonl_to_parquet_batch(args.batch, compression=compression)
    elif args.input:
        jsonl_to_parquet(args.input, args.output, compression=compression)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
