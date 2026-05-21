"""Phân hệ làm sạch và xác thực dữ liệu nâng cấp — Pipeline 2 tầng AI.

Pipeline:
  Tầng 0 — Xác thực cơ bản   : Kiểm tra file tồn tại, kích thước ≥ 1KB, PIL verify.
  Tầng 1 — Lọc chủ đề (CLIP) : Xác nhận ảnh là lá lúa (openai/clip-vit-base-patch32).
  Tầng 2 — Phân loại bệnh    : Gán nhãn bệnh (prithivMLmods/Rice-Leaf-Disease).
  Tầng 3 — Phát hiện trùng   : Loại bỏ ảnh trùng lặp qua MD5 hash.

Kết quả:
  - Ảnh rác → cách ly vào `data/rejected/<class>/`
  - Ảnh sạch → metadata được ghi lại vào `metadata.jsonl` của từng lớp
  - Báo cáo CSV đầy đủ → `src/rice_healthy_filtered.csv`
  - Báo cáo tổng hợp toàn dataset → `data/raw/_metadata.csv`
"""

import csv
from .utils import compute_md5
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn mặc định
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent

DATA_RAW_DIR       = _PROJECT_ROOT / "data" / "raw"
DATA_REJECTED_DIR  = _PROJECT_ROOT / "data" / "rejected"
CSV_FILTERED       = _SRC_DIR / "rice_healthy_filtered.csv"      # khớp filter_rice_images.py
CSV_METADATA       = DATA_RAW_DIR / "_metadata.csv"

# Nhãn bệnh lúa hỗ trợ (phải khớp với model Rice-Leaf-Disease)
RICE_DISEASE_LABELS: List[str] = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot",
    "Healthy",
    "Tungro",
]

# ---------------------------------------------------------------------------
# Tầng 1 — CLIP filter (kiểm tra ảnh có phải lá lúa không)
# ---------------------------------------------------------------------------

def _load_clip_model():
    """Tải model CLIP (lazy, chỉ tải một lần)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor
    except Exception as e:
        print(f"⚠️  Không thể tải CLIP model: {e}")
        return None, None


_CLIP_MODEL = None
_CLIP_PROCESSOR = None

def _get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        print("⏳ Đang tải CLIP model (lần đầu chạy có thể mất vài phút)...")
        _CLIP_MODEL, _CLIP_PROCESSOR = _load_clip_model()
    return _CLIP_MODEL, _CLIP_PROCESSOR


def is_rice_leaf_clip(
    image_path: Path,
    threshold: float = 0.4,
) -> Tuple[bool, float]:
    """Tầng 1: Dùng CLIP để kiểm tra xem ảnh có phải lá lúa không.

    Args:
        image_path: Đường dẫn tệp ảnh.
        threshold:  Ngưỡng xác suất tối thiểu để chấp nhận là lá lúa.

    Returns:
        (is_rice, clip_rice_prob)
    """
    model, processor = _get_clip()
    if model is None:
        # Fallback: chấp nhận tất cả nếu không tải được model
        return True, 1.0

    try:
        import torch
        image = Image.open(image_path).convert("RGB")
        texts = [
            "a photo of a rice leaf",
            "a photo of a green leaf",
            "a photo of a tree leaf",
            "a photo of a grass blade",
        ]
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        rice_prob = float(probs[0][0].item())
        return rice_prob > threshold, rice_prob
    except Exception as e:
        print(f"  ⚠️  CLIP lỗi ({image_path.name}): {e}")
        return False, 0.0


# ---------------------------------------------------------------------------
# Tầng 2 — Rice-Leaf-Disease classifier
# ---------------------------------------------------------------------------

def _load_rice_model():
    """Tải model phân loại bệnh lúa (lazy)."""
    try:
        from transformers import AutoImageProcessor, SiglipForImageClassification
        rice_model_name = "prithivMLmods/Rice-Leaf-Disease"
        processor = AutoImageProcessor.from_pretrained(rice_model_name)
        model = SiglipForImageClassification.from_pretrained(rice_model_name)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"⚠️  Không thể tải Rice-Leaf-Disease model: {e}")
        return None, None


_RICE_MODEL = None
_RICE_PROCESSOR = None

def _get_rice_model():
    global _RICE_MODEL, _RICE_PROCESSOR
    if _RICE_MODEL is None:
        print("⏳ Đang tải Rice-Leaf-Disease model...")
        _RICE_MODEL, _RICE_PROCESSOR = _load_rice_model()
    return _RICE_MODEL, _RICE_PROCESSOR


def predict_rice_disease(image_path: Path) -> Tuple[str, float]:
    """Tầng 2: Phân loại bệnh lúa và trả về (nhãn, độ tin cậy).

    Returns:
        (label, confidence) — label là một trong RICE_DISEASE_LABELS.
    """
    model, processor = _get_rice_model()
    if model is None:
        return "Unknown", 0.0

    try:
        import torch
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
        max_conf, idx = torch.max(probs, dim=0)
        label = RICE_DISEASE_LABELS[int(idx.item())]
        return label, float(max_conf.item())
    except Exception as e:
        print(f"  ⚠️  Rice model lỗi ({image_path.name}): {e}")
        return "Unknown", 0.0


# ---------------------------------------------------------------------------
# Tầng 0 — Xác thực tệp cơ bản
# ---------------------------------------------------------------------------

def validate_image_file(img_path: Path) -> Tuple[bool, str]:
    """Kiểm tra tính toàn vẹn của tệp ảnh (tồn tại, kích thước, PIL verify).

    Returns:
        (is_valid, reason) — reason là "OK" hoặc mô tả lỗi.
    """
    if not img_path.exists():
        return False, "file_not_found"
    if img_path.stat().st_size < 1024:
        return False, "file_too_small (<1KB)"
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, "OK"
    except Exception as e:
        return False, f"corrupted: {str(e)[:60]}"


# ---------------------------------------------------------------------------
# Hash MD5 — phát hiện trùng lặp
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata_map(class_dir: Path) -> Dict[str, Dict]:
    """Tải `metadata.jsonl` thành dict {filename → entry}."""
    meta_map: Dict[str, Dict] = {}
    meta_path = class_dir / "metadata.jsonl"
    if not meta_path.exists():
        return meta_map
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    fn = entry.get("filename")
                    if fn:
                        meta_map[fn] = entry
                except Exception:
                    pass
    except Exception as e:
        print(f"  ⚠️  Không tải được metadata.jsonl ({class_dir.name}): {e}")
    return meta_map


# ---------------------------------------------------------------------------
# Core: quét & làm sạch toàn bộ dataset
# ---------------------------------------------------------------------------

def scan_and_clean_dataset(
    data_root: Optional[str] = None,
    output_csv: Optional[str] = None,
    rejected_root: Optional[str] = None,
    clip_threshold: float = 0.4,
    write_filtered_csv: bool = True,
) -> List[Dict]:
    """Quét toàn bộ dataset, áp dụng pipeline 4 tầng và xuất báo cáo CSV.

    Args:
        data_root:          Thư mục gốc chứa các lớp ảnh (mặc định: data/raw/).
        output_csv:         Đường dẫn file CSV tổng hợp toàn dataset.
        rejected_root:      Thư mục cách ly ảnh lỗi (mặc định: data/rejected/).
        clip_threshold:     Ngưỡng xác suất CLIP để chấp nhận là lá lúa (default 0.4).
        write_filtered_csv: Có ghi file `rice_healthy_filtered.csv` hay không.

    Returns:
        Danh sách các bản ghi của ảnh hợp lệ được giữ lại.
    """
    root = Path(data_root) if data_root else DATA_RAW_DIR
    rejected = Path(rejected_root) if rejected_root else DATA_REJECTED_DIR
    csv_out = Path(output_csv) if output_csv else CSV_METADATA

    # Tự động điều chỉnh nếu chạy từ bên trong thư mục src/
    if not root.exists():
        fallback = Path("..") / root
        if fallback.exists():
            print(f"⚠️  Phát hiện chạy từ thư mục con. Dùng đường dẫn: {fallback}")
            root = fallback
            rejected = Path("..") / rejected
            csv_out = Path("..") / csv_out
        else:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu: '{root}'"
            )

    rejected.mkdir(parents=True, exist_ok=True)

    # Thống kê tổng hợp
    stats = {
        "scanned":  0,
        "kept":     0,
        "rejected": 0,
        "reasons": {
            "invalid_file":    0,
            "not_rice_leaf":   0,
            "duplicate_hash":  0,
            "unknown_disease": 0,
        },
    }

    # Ghi rice_healthy_filtered.csv theo chuẩn filter_rice_images.py
    filtered_rows: List[List] = []      # [filename, final_label, disease_conf, clip_rice_prob]

    # Bản ghi toàn dataset cho CSV tổng hợp
    all_records: List[Dict] = []

    print(f"\n{'='*72}")
    print(f"  BẮT ĐẦU LÀM SẠCH DATASET — {root}")
    print(f"{'='*72}")

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir()
                        and not d.name.startswith(("_", "."))
                        and d.name.lower() != "rejected")

    if not class_dirs:
        print(f"⚠️  Không tìm thấy lớp nào trong {root}")
        return []

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n📂 Lớp: {class_name}")

        meta_map = load_metadata_map(class_dir)
        img_files = sorted(
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )

        if not img_files:
            print(f"   (Không có ảnh)")
            continue

        class_rejected_dir = rejected / class_name
        seen_hashes: Set[str] = set()
        kept_entries:     List[Dict] = []
        rejected_entries: List[Dict] = []

        for img_path in tqdm(img_files, desc=f"  Lọc {class_name}"):
            stats["scanned"] += 1
            filename = img_path.name

            try:
                file_size = img_path.stat().st_size
            except Exception:
                file_size = 0

            # Khởi tạo bản ghi metadata
            base_meta = meta_map.get(filename, {
                "filename": filename,
                "url": "",
                "query": "",
                "title": "",
                "description": "",
                "source": "local",
            })

            def _reject(reason: str, reason_key: str, extra: Optional[Dict] = None):
                """Cách ly ảnh và ghi log."""
                stats["rejected"] += 1
                stats["reasons"][reason_key] += 1
                class_rejected_dir.mkdir(parents=True, exist_ok=True)
                dest = class_rejected_dir / filename
                try:
                    shutil.move(str(img_path), str(dest))
                except Exception as mv_err:
                    print(f"    ⚠️  Di chuyển thất bại ({filename}): {mv_err}")
                entry = {**base_meta, "reject_reason": reason, "size_bytes": file_size}
                if extra:
                    entry.update(extra)
                rejected_entries.append(entry)

            # ── Tầng 0: Xác thực tệp ──────────────────────────────────────
            is_valid, err_msg = validate_image_file(img_path)
            if not is_valid:
                _reject(f"invalid_file: {err_msg}", "invalid_file")
                filtered_rows.append([filename, f"REJECTED ({err_msg})", 0.0, "N/A"])
                continue

            # Đọc kích thước ảnh
            try:
                with Image.open(img_path) as _img:
                    img_w, img_h = _img.size
            except Exception:
                img_w, img_h = 0, 0

            # ── Tầng 3: MD5 hash — trùng lặp ─────────────────────────────
            img_hash = compute_md5(img_path)
            if img_hash and img_hash in seen_hashes:
                _reject("duplicate_hash", "duplicate_hash",
                        {"image_hash": img_hash, "width": img_w, "height": img_h})
                filtered_rows.append([filename, "REJECTED (duplicate)", 0.0, "N/A"])
                continue
            if img_hash:
                seen_hashes.add(img_hash)

            # ── Tầng 1: CLIP — có phải lá lúa? ────────────────────────────
            is_rice, clip_prob = is_rice_leaf_clip(img_path, threshold=clip_threshold)
            if not is_rice:
                _reject("not_rice_leaf", "not_rice_leaf",
                        {"clip_rice_prob": clip_prob, "width": img_w, "height": img_h})
                filtered_rows.append([filename, "REJECTED (not rice leaf)", clip_prob, "N/A"])
                continue

            # ── Tầng 2: Phân loại bệnh ────────────────────────────────────
            disease_label, disease_conf = predict_rice_disease(img_path)
            if disease_label == "Unknown":
                _reject("unknown_disease", "unknown_disease",
                        {"clip_rice_prob": clip_prob, "width": img_w, "height": img_h})
                filtered_rows.append([filename, "REJECTED (unknown disease)", disease_conf, clip_prob])
                continue

            # ── Ảnh hợp lệ ────────────────────────────────────────────────
            stats["kept"] += 1
            url   = base_meta.get("url", "")
            query = base_meta.get("query", "")
            title = base_meta.get("title", "")

            kept_entry = {
                **base_meta,
                "image_hash":        img_hash,
                "width":             img_w,
                "height":            img_h,
                "size_bytes":        file_size,
                "disease_label":     disease_label,
                "disease_confidence": disease_conf,
                "clip_rice_prob":    clip_prob,
            }
            kept_entries.append(kept_entry)

            all_records.append({
                "class":              class_name,
                "filename":           filename,
                "path":               str(img_path),
                "url":                url,
                "query":              query,
                "title":              title,
                "size_bytes":         file_size,
                "width":              img_w,
                "height":             img_h,
                "hash_md5":           img_hash,
                "disease_label":      disease_label,
                "disease_confidence": round(disease_conf, 6),
                "clip_rice_prob":     round(clip_prob, 6),
            })

            filtered_rows.append([filename, disease_label, disease_conf, clip_prob])

        # ── Cập nhật metadata.jsonl của lớp ───────────────────────────────
        meta_path = class_dir / "metadata.jsonl"
        if kept_entries:
            with open(meta_path, "w", encoding="utf-8") as f:
                for entry in kept_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            if meta_path.exists():
                meta_path.unlink()

        # Ghi log ảnh bị loại
        if rejected_entries:
            class_rejected_dir.mkdir(parents=True, exist_ok=True)
            rej_log = class_rejected_dir / "rejected_metadata.jsonl"
            with open(rej_log, "w", encoding="utf-8") as f:
                for entry in rejected_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"   → Đã cách ly {len(rejected_entries)} ảnh vào {class_rejected_dir.relative_to(_PROJECT_ROOT)}")

    # ── Ghi rice_healthy_filtered.csv (khớp filter_rice_images.py) ────────
    if write_filtered_csv and filtered_rows:
        CSV_FILTERED.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_FILTERED, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "final_label", "disease_confidence", "clip_rice_prob"])
            writer.writerows(filtered_rows)

    # ── Ghi _metadata.csv tổng hợp ───────────────────────────────────────
    if all_records:
        keys = [
            "class", "filename", "path", "url", "query", "title",
            "size_bytes", "width", "height", "hash_md5",
            "disease_label", "disease_confidence", "clip_rice_prob",
        ]
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_records:
                writer.writerow({k: r.get(k, "") for k in keys})

    # ── Báo cáo tổng kết ─────────────────────────────────────────────────
    scanned  = stats["scanned"]
    kept     = stats["kept"]
    rejected_total = stats["rejected"]
    pct_kept = (kept / scanned * 100) if scanned else 0

    accepted_csv = sum(1 for row in filtered_rows if not str(row[1]).startswith("REJECTED"))
    rejected_csv = len(filtered_rows) - accepted_csv

    print(f"\n{'='*72}")
    print(f"  BÁO CÁO LÀM SẠCH DATASET")
    print(f"{'-'*72}")
    print(f"  Tổng ảnh đã quét     : {scanned:>6}")
    print(f"  Ảnh HỢP LỆ (giữ lại): {kept:>6}  ({pct_kept:.1f}%)")
    print(f"  Ảnh bị LOẠI (cách ly): {rejected_total:>6}  ({100 - pct_kept:.1f}%)")
    print(f"\n  Chi tiết lý do loại bỏ:")
    print(f"    • File lỗi/hỏng       : {stats['reasons']['invalid_file']:>5}")
    print(f"    • Không phải lá lúa   : {stats['reasons']['not_rice_leaf']:>5}")
    print(f"    • Trùng lặp MD5       : {stats['reasons']['duplicate_hash']:>5}")
    print(f"    • Lỗi phân loại bệnh  : {stats['reasons']['unknown_disease']:>5}")
    print(f"\n  📊 Kết quả: {accepted_csv} ảnh được chấp nhận (là lá lúa), {rejected_csv} ảnh bị loại.")
    print(f"  ✅ Chi tiết lưu tại {CSV_FILTERED}")
    if all_records:
        print(f"  📋 Metadata tổng hợp  : {csv_out}")
    print(f"  🗑️  Ảnh rác cách ly tại : {rejected}")
    print(f"{'='*72}\n")

    return all_records


# ---------------------------------------------------------------------------
# Thống kê phụ trợ
# ---------------------------------------------------------------------------

def get_class_distribution(records: List[Dict]) -> Dict[str, int]:
    """Đếm số ảnh theo từng lớp phân loại."""
    counts: Dict[str, int] = {}
    for r in records:
        cls = r.get("class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def get_disease_distribution(records: List[Dict]) -> Dict[str, int]:
    """Đếm số ảnh theo từng nhãn bệnh."""
    counts: Dict[str, int] = {}
    for r in records:
        label = r.get("disease_label", "Unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def get_image_size_stats(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Thống kê kích thước ảnh (min/max/mean) theo từng lớp."""
    raw: Dict[str, Dict[str, List[int]]] = {}
    for r in records:
        cls = r.get("class", "")
        w, h = r.get("width", 0), r.get("height", 0)
        if not cls or not w or not h:
            continue
        if cls not in raw:
            raw[cls] = {"w": [], "h": []}
        raw[cls]["w"].append(w)
        raw[cls]["h"].append(h)

    report: Dict[str, Dict[str, float]] = {}
    for cls, data in raw.items():
        ws, hs = data["w"], data["h"]
        report[cls] = {
            "w_min":  min(ws),
            "w_max":  max(ws),
            "w_mean": sum(ws) / len(ws),
            "h_min":  min(hs),
            "h_max":  max(hs),
            "h_mean": sum(hs) / len(hs),
        }
    return report


# ---------------------------------------------------------------------------
# Apply filter từ CSV → Move + Update metadata + Report
# ---------------------------------------------------------------------------

def apply_filter_from_csv(
    csv_path: str,
    source_dir: str,
    rejected_dir: str,
    dry_run: bool = False,
) -> Dict:
    """
    Áp dụng kết quả filter từ CSV:
    - Move ảnh REJECTED → data/rejected/<class>/
    - Cập nhật metadata.jsonl (xóa entry ảnh bị loại)
    - In báo cáo thống kê

    CSV có thể có nhiều schema khác nhau → auto-detect cột:
        filename / file_name / image / path
        status / decision / label
        reason / reject_reason
    """

    csv_path = Path(csv_path)
    source_dir = Path(source_dir)
    rejected_dir = Path(rejected_dir)

    rejected_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV không tồn tại: {csv_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir không tồn tại: {source_dir}")

    print(f"\n📄 Đang đọc CSV: {csv_path}")
    print(f"📁 Source: {source_dir}")
    print(f"🗑️ Rejected: {rejected_dir}")
    print(f"{'🧪 DRY RUN (không di chuyển file)' if dry_run else '🚀 APPLY CHANGES'}")

    # -----------------------------------------------------------------------
    # Detect columns
    # -----------------------------------------------------------------------

    def pick_col(row_keys, candidates):
        for c in candidates:
            if c in row_keys:
                return c
        return None

    # -----------------------------------------------------------------------
    # Load CSV
    # -----------------------------------------------------------------------

    rejected_files = set()
    accepted_files = set()
    reject_reasons: Dict[str, int] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV không có header")

        keys = [k.strip() for k in reader.fieldnames]

        col_file = pick_col(keys, ["filename", "file_name", "image", "path"])
        col_status = pick_col(keys, ["status", "decision", "label"])
        col_reason = pick_col(keys, ["reason", "reject_reason"])

        if not col_file or not col_status:
            raise ValueError(
                f"Không tìm thấy cột cần thiết trong CSV. Found={keys}"
            )

        print(f"🔎 Column mapping:")
        print(f"   file   → {col_file}")
        print(f"   status → {col_status}")
        print(f"   reason → {col_reason}")

        for row in reader:
            fname = (row.get(col_file) or "").strip()
            status = (row.get(col_status) or "").strip().lower()
            reason = (row.get(col_reason) or "unknown").strip()

            if not fname:
                continue

            # Normalize path → chỉ lấy filename
            fname = Path(fname).name

            if status in ("rejected", "reject", "0", "false"):
                rejected_files.add(fname)
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            else:
                accepted_files.add(fname)

    # -----------------------------------------------------------------------
    # Move rejected images
    # -----------------------------------------------------------------------

    moved = 0
    missing = 0

    for fname in rejected_files:
        src = source_dir / fname
        dst = rejected_dir / fname

        if not src.exists():
            missing += 1
            continue

        if dry_run:
            print(f"[DRY] move {src} → {dst}")
        else:
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                print(f"⚠️ Move lỗi {src}: {e}")

    # -----------------------------------------------------------------------
    # Update metadata.jsonl
    # -----------------------------------------------------------------------

    meta_path = source_dir / "metadata.jsonl"
    kept_entries = []
    removed_entries = 0

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    fname = entry.get("filename")
                    if fname in rejected_files:
                        removed_entries += 1
                        continue
                    kept_entries.append(entry)
                except:
                    continue

        if not dry_run:
            with open(meta_path, "w", encoding="utf-8") as f:
                for entry in kept_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    total = len(rejected_files) + len(accepted_files)
    accept_rate = len(accepted_files) / total if total else 0

    print("\n📊 REPORT")
    print("─" * 50)
    print(f"Tổng ảnh        : {total}")
    print(f"Accepted        : {len(accepted_files)}")
    print(f"Rejected        : {len(rejected_files)}")
    print(f"Accept rate     : {accept_rate:.2%}")
    print(f"Moved           : {moved}")
    print(f"Missing file    : {missing}")
    print(f"Metadata removed: {removed_entries}")

    if reject_reasons:
        print("\nTop reject reasons:")
        top = sorted(reject_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
        for reason, count in top:
            print(f"  - {reason}: {count}")

    return {
        "total": total,
        "accepted": len(accepted_files),
        "rejected": len(rejected_files),
        "accept_rate": accept_rate,
        "moved": moved,
        "missing": missing,
        "metadata_removed": removed_entries,
        "reasons": reject_reasons,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data cleaning & filter tooling"
    )

    subparsers = parser.add_subparsers(dest="command")

    # -----------------------------------------------------------------------
    # Command: apply-filter
    # -----------------------------------------------------------------------

    parser_filter = subparsers.add_parser(
        "apply-filter",
        help="Áp dụng CSV filter → move rejected + update metadata"
    )

    parser_filter.add_argument(
        "--csv", required=True,
        help="Path tới file CSV filter (vd: rice_healthy_filtered.csv)"
    )

    parser_filter.add_argument(
        "--source", required=True,
        help="Thư mục nguồn (vd: data/raw/Rice_Healthy)"
    )

    parser_filter.add_argument(
        "--rejected", required=True,
        help="Thư mục rejected (vd: data/rejected/Rice_Healthy)"
    )

    parser_filter.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ preview, không di chuyển file"
    )

    # -----------------------------------------------------------------------
    # Parse args
    # -----------------------------------------------------------------------

    args = parser.parse_args()

    if args.command == "apply-filter":
        apply_filter_from_csv(
            csv_path=args.csv,
            source_dir=args.source,
            rejected_dir=args.rejected,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
