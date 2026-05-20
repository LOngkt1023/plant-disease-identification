"""Các tiện ích hỗ trợ xuất nhập tập dữ liệu và vẽ biểu đồ."""

from pathlib import Path
from typing import List


def list_image_files(folder: str) -> List[str]:
    p = Path(folder)
    return [str(x) for x in p.glob('**/*') if x.is_file()]


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

