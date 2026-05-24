"""Search query profiles built from central disease class configuration.

Không hard-code class rải rác: toàn bộ class/keyword lấy từ src.config.disease_classes.
"""
from __future__ import annotations

import random
from typing import List, Optional

try:
    from .config.disease_classes import PLANT_DISEASE_CLASSES
except (ImportError, ValueError):
    from config.disease_classes import PLANT_DISEASE_CLASSES


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def build_search_queries(
    class_name: str,
    fallback: Optional[List[str]] = None,
    limit: int = 50,
    shuffle: bool = True,
) -> List[str]:
    class_cfg = PLANT_DISEASE_CLASSES.get(class_name, {})
    keywords = list(class_cfg.get("keywords", []))
    base = fallback or [class_name.replace("_", " ")]
    combined = _dedupe(base + keywords)
    if shuffle:
        random.shuffle(combined)
    return combined[:limit]


def list_supported_classes() -> List[str]:
    return list(PLANT_DISEASE_CLASSES.keys())


def get_query_count(class_name: str) -> int:
    return len(PLANT_DISEASE_CLASSES.get(class_name, {}).get("keywords", []))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate search queries from central class config.")
    parser.add_argument("class_name", nargs="?", help="Tên class (bỏ qua để liệt kê tất cả)")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    if args.class_name:
        print(f"Queries for {args.class_name}:")
        for i, q in enumerate(
            build_search_queries(args.class_name, limit=args.limit, shuffle=not args.no_shuffle), 1
        ):
            print(f"{i:>2}. {q}")
    else:
        print("Supported classes:")
        for cls in list_supported_classes():
            print(f"- {cls} ({get_query_count(cls)} keywords)")