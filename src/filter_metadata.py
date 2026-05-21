"""Lọc metadata crawl bằng bộ lọc từ khóa dùng chung.

Script này đọc file JSONL metadata do crawler sinh ra và ghi ra file mới
chỉ gồm những ảnh được keyword_filter chấp nhận.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

try:
    from .keyword_filter import check_image_relevance
except (ImportError, ValueError):
    from keyword_filter import check_image_relevance  # type: ignore


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def filter_metadata(input_path: Path, output_path: Path, class_name: str) -> Dict[str, int]:
    kept = 0
    rejected = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_handle:
        for entry in _iter_jsonl(input_path):
            url = entry.get("url", "")
            query = entry.get("query", "")
            title = entry.get("title", "")
            description = entry.get("description", "")
            is_relevant, _reason = check_image_relevance(
                url=url,
                query=query,
                class_name=class_name,
                title=title,
                description=description,
            )
            if is_relevant:
                out_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected += 1

    return {"kept": kept, "rejected": rejected, "total": kept + rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc metadata JSONL bằng keyword_filter.")
    parser.add_argument("input", help="Đường dẫn metadata.jsonl đầu vào")
    parser.add_argument("output", help="Đường dẫn JSONL đầu ra")
    parser.add_argument("--class-name", required=True, help="Tên lớp, ví dụ Rice_Healthy")
    args = parser.parse_args()

    summary = filter_metadata(Path(args.input), Path(args.output), args.class_name)
    print(f"Kept: {summary['kept']}, rejected: {summary['rejected']}, total: {summary['total']}")