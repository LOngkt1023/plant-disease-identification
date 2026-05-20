"""Trình thu thập dữ liệu web ẩn danh nâng cao — Công cụ xây dựng bộ dữ liệu ảnh bệnh cây trồng.

Các cải tiến trong phiên bản này:
  - ProxyPool an toàn đa luồng (threading.Lock bảo vệ get/remove/_refresh)
  - Khả năng tiếp tục từ điểm dừng (đọc file ảnh + metadata JSONL đã có)
  - Ghi metadata liên tục (ghi JSONL từng ảnh, an toàn khi sập chương trình)
  - Đặt tên file nguyên tử (chỉ tăng bộ đếm khi tải thành công)
  - requests.Session cho connection pooling + timeout đọc/kết nối
  - Cuộn động trong nodriver (cùng logic với selenium)
  - Kiểm tra sức khỏe proxy qua httpbin.org/ip (xác minh nội dung, không chỉ kiểm tra status)
  - Lọc kích thước ảnh (tối thiểu 100×100 px, không chỉ kiểm tra kích thước byte thô)
  - Cảnh báo đạo đức khi khởi động với prompt xác nhận
"""

import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, unquote
import urllib.error
import urllib.request

# ── Cấu hình Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Đảm bảo terminal Windows hỗ trợ UTF-8 để hiển thị tiếng Việt đúng
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
)
from tqdm import tqdm
import hashlib

try:
    import curl_cffi.requests as cffi_requests  # type: ignore
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False
    log.debug("curl_cffi chưa cài đặt — sẽ dùng requests thông thường (có thể bị chặn TLS).")

try:
    import undetected_chromedriver as uc
except ImportError:
    log.warning("Chưa cài đặt undetected_chromedriver — đang chuyển sang Selenium dự phòng.")
    from selenium import webdriver as uc  # type: ignore

from selenium.webdriver.common.by import By

try:
    from .utils import ensure_dir
except (ImportError, ValueError):
    from utils import ensure_dir  # type: ignore


# ── Hằng số ───────────────────────────────────────────────────────────────────

# Chỉ thử lại với các lỗi mạng/IO tạm thời, không bao giờ thử lại khi gặp lỗi logic code
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
    urllib.error.URLError,
    urllib.error.HTTPError,
)

# Kích thước ảnh tối thiểu chấp nhận (lọc ảnh quá nhỏ / thumbnail chất lượng thấp)
MIN_IMAGE_WIDTH  = 100  # pixel
MIN_IMAGE_HEIGHT = 100  # pixel

# (Flickr support removed) - Flickr API key and fetch function removed to simplify
# the scraping stack and avoid rate-limited public API usage.

DATASET_CLASSES: Dict[str, str] = {
    "Rice_Healthy":    "La lua khoe manh",
    "Rice_Blast":      "La lua benh dao on",
    "Rice_Blight":     "La lua benh bac la",
    "Coffee_Healthy":  "La ca phe khoe manh",
    "Coffee_Rust":     "La ca phe benh ri sat",
    "Tomato_Healthy":  "La ca chua khoe manh",
    "Tomato_Blight":   "La ca chua benh suong mai",
    "Tomato_Curl":     "La ca chua benh xoan la",
    "Citrus_Canker":   "La cam benh loet",
    "Citrus_Greening": "La cam benh vang la",
}

SEARCH_QUERIES: Dict[str, List[str]] = {
    # Nhàn nẹ lại: mỗi lớp có từ khóa nhiều ngôn ngữ (Tiếng Anh, Tiếng Việt, Tiếng Trung, Tiếng Thái)
    # Ưu tiên từ khóa "close-up", "single leaf", "macro" để giảm ảnh cảnh đồng, ảnh nhiều lá
    # CẢI TIẾN: Thêm từ khóa ngách để tránh lá ngô, lá sả, ảnh infographic
    "Rice_Healthy": [
        # Tiếng Anh: cận cảnh, single leaf, macro
        "healthy rice leaf close-up single",
        "rice plant healthy green leaf macro",
        "Oryza sativa healthy leaf",
        "IR64 rice healthy leaf",
        "jasmine rice green leaf close-up",
        "paddy rice single leaf macro",
        # Tiếng Việt: lúa IR64, lúa thơm, lá cây đơn
        "lú a khỏe mạnh lá cây",
        "lá lúa IR64 xanh lá cây",
        "lúa thơm lá lẻ khỏe mạnh",
        "cận cảnh lá lúa xanh",
        "lúa Oryza sativa lá đơn",
        # Tiếng Trung: Oryza sativa, lúa xanh
        "稻叶 健康 病虫害",
        "水稻 绿色 叶片 特写",
        "Oryza sativa 单叶 健康",
        "稻叶细节 绿色 特写",
        # Tiếng Thái: ข้าว = lúa
        "ข้าวสารใบเขียว",
        "ใบข้าว สุขภาพดี",
        "ข้าวหอม ใบเดี่ยว สีเขียว",
    ],
    "Rice_Blast": [
        # Tiếng Anh: cận cảnh, single leaf, disease lesion
        "rice blast disease leaf close-up",
        "Magnaporthe oryzae rice leaf lesion",
        "rice blast fungal infection single leaf",
        "rice blast brown spot macro",
        "pyricularia oryzae leaf spot",
        # Tiếng Việt: bệnh đạo ôn, lá lúa
        "lú a bệnh đạo ôn lá",
        "bệnh đạo ôn lúa lá cây",
        "lá lúa bệnh nâu",
        "cận cảnh lá lúa bệnh đạo ôn",
        # Tiếng Trung: 稻瘟病 (Magnaporthe)
        "稻瘟病 叶片 特写",
        "水稻纹枯病 病斑",
        "Magnaporthe 水稻 叶片",
        "稻叶 棕色 病斑 特写",
        # Tiếng Thái: โรคไข้ทองข้าว
        "โรคไข้ทองข้าว ใบ",
        "ข้าวบิด พยาธิ",
        "ข้าว ใบ โรคราในข้าว",
    ],
    "Rice_Blight": [
        # Tiếng Anh: bacterial blight, leaf spot
        "rice bacterial blight leaf close-up",
        "Xanthomonas oryzae rice leaf",
        "rice blight yellowing single leaf",
        "rice bacterial streak lesion",
        "bacterial leaf streak rice macro",
        # Tiếng Việt: bệnh bạc lá, lá lúa
        "lú a bằc lá bệnh",
        "lá lúa bệnh bạc lá",
        "bệnh bạc lá lúa cây",
        "cận cảnh lá lúa bệnh vàng",
        # Tiếng Trung: 稻叶枯病 (bacterial blight)
        "稻叶枯病 叶片",
        "水稻细菌性条纹病",
        "白叶枯病 稻 特写",
        "稻 细菌性 条纹 叶片",
        # Tiếng Thái: โรคแผลสีขาว
        "ใบข้าว บาดแผลสีขาว",
        "ข้าวใบแห้ง",
        "ข้าว ใบ โรค",
    ],
    "Coffee_Healthy": [
        "healthy coffee leaf single close-up",
        "Coffea arabica healthy green leaf macro",
        "coffee plant leaf green isolated",
        "lá cà phê khỏe mạnh",
        "咖啡叶 健康 特写",
        "阿拉比卡咖啡 绿叶",
        "ใบกาแฟ สีเขียว",
        "สุขภาพกาแฟ ใบสวย",
    ],
    "Coffee_Rust": [
        "coffee leaf rust disease close-up",
        "Hemileia vastatrix coffee leaf orange spots",
        "coffee rust fungus single leaf",
        "lá cà phê gỉ sắt bệnh",
        "咖啡叶锈病 橙色斑点",
        "咖啡铁皮病 特写",
        "โรคสนิมใบกาแฟ",
        "ใบกาแฟ โรคแดง",
    ],
    "Tomato_Healthy": [
        "healthy tomato leaf single close-up",
        "Solanum lycopersicum healthy leaf macro",
        "tomato plant green leaf isolated",
        "lá cà chua khỏe mạnh",
        "番茄叶 健康 绿色",
        "黄瓜状番茄 无病害",
        "ใบมะเขือเทศ สดใจ",
        "มะเขือเทศ ใบสุขภาพดี",
    ],
    "Tomato_Blight": [
        "tomato late blight leaf close-up",
        "Phytophthora infestans tomato leaf",
        "tomato blight brown lesion single leaf",
        "lá cà chua sương mai bệnh",
        "番茄晚疫病 褐色病斑",
        "番茄疫病 叶片腐烂",
        "โรคหนาวเย็นมะเขือเทศ",
        "ใบมะเขือเทศ สูบเหี่ย",
    ],
    "Tomato_Curl": [
        "tomato leaf curl virus close-up",
        "TYLCV tomato curled leaf",
        "tomato leaf curl disease single plant",
        "lá cà chua xoăn lá bệnh",
        "番茄卷叶病毒 卷曲",
        "番茄黄化卷叶病毒 病状",
        "ไวรัสม้วนใบมะเขือเทศ",
        "ใบมะเขือเทศ โค้งม้วน",
    ],
    "Citrus_Canker": [
        "citrus canker disease leaf close-up",
        "Xanthomonas citri citrus leaf lesion",
        "orange leaf canker spots macro",
        "lá cam loét bệnh cần",
        "柑橘溃疡病 叶片病斑",
        "橙叶 黄色斑点 病害",
        "โรคแผลสะดือส้ม",
        "ใบส้ม ฝ้ากรรม",
    ],
    "Citrus_Greening": [
        "citrus greening HLB disease leaf",
        "Huanglongbing citrus yellow mottled leaf",
        "citrus greening asymmetric yellowing leaf",
        "lá cam vàng HLB bệnh",
        "柑橘黄龙病 黄化叶",
        "柠檬绿叶病 斑驳",
        "โรคเหลืองมะกดลิ่ว",
        "ใบส้มฟาง สีเหลือง",
    ],
}

# ── Từ khóa LOẠI TRỪ: lọc ảnh không liên quan (infographic, diagram, other plants) ──
# Nếu URL hoặc từ khóa tìm kiếm chứa bất kỳ từ khóa nào dưới đây, ảnh sẽ bị bỏ qua
# CẢNH BÁO: Các từ khóa được thiết kế để dùng word boundary regex (\b...\b)
#           nên tránh các từ quá ngắn (cỏ, chè) → dùng từ khóa cụ thể hơn (corn, barley, wheat)
#           Ưu tiên tiếng Anh vì hầu hết URL ảnh trên thế giới dùng tiếng Anh không dấu
EXCLUDE_KEYWORDS: Dict[str, List[str]] = {
    # Chung cho tất cả: loại bỏ infographic, diagram, cartoon, vector
    "common": [
        "infographic",
        "diagram", 
        "cartoon",
        "chart",
        "graph",
        "vector",
        "illustration",
        "icon",
        "emoji",
        "logo",
        "animation",
        "drawing",
        "sketch",
        "painting",
        "artwork",
        "flower",  # hoa không phải lá bệnh cây trồng
        "blossom",
        "petal",
    ],
    # LÚA (Rice): loại bỏ ảnh từ ngô, lúa mạch, cà phê, cà chua, cam/quýt, v.v.
    "Rice_Healthy": [
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "tomato", "solanum", "lycopersicum",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "sorghum", "sugarcane", "millet",
        "sedge", "carex",
        "cattail", "typha",
        "reed", "phragmites",
        "bamboo", "grass_like", "poaceae",
    ],
    "Rice_Blast": [
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "tomato", "solanum", "lycopersicum",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "sorghum", "sugarcane", "millet",
        "sedge", "carex",
        "cattail", "typha",
        "reed", "phragmites",
        "bamboo", "grass_like", "poaceae",
    ],
    "Rice_Blight": [
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "tomato", "solanum", "lycopersicum",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "sorghum", "sugarcane", "millet",
        "sedge", "carex",
        "cattail", "typha",
        "reed", "phragmites",
        "bamboo", "grass_like", "poaceae",
    ],
    # CÀ PHÊ (Coffee): loại bỏ ảnh từ lúa, ngô, cà chua, cam/quýt, v.v.
    "Coffee_Healthy": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "tomato", "solanum", "lycopersicum",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "mint", "mentha",
        "basil", "ocimum",
        "tea", "camellia", "thea",
    ],
    "Coffee_Rust": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "tomato", "solanum", "lycopersicum",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "mint", "mentha",
        "basil", "ocimum",
        "tea", "camellia", "thea",
    ],
    # CÀ CHUA (Tomato): loại bỏ ảnh từ lúa, ngô, cà phê, cam/quýt, v.v.
    "Tomato_Healthy": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper", "paprika",
        "chili", "pimiento",
    ],
    "Tomato_Blight": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper", "paprika",
        "chili", "pimiento",
    ],
    "Tomato_Curl": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cacao", "cocoa", "theobroma",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper", "paprika",
        "chili", "pimiento",
    ],
    # CAM/QUÝT (Citrus): loại bỏ ảnh từ lúa, ngô, cà phê, cà chua, v.v.
    "Citrus_Canker": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "tomato", "solanum", "lycopersicum",
        "cacao", "cocoa", "theobroma",
        "apple", "malus", "pome",
        "pear", "pyrus",
        "plum", "prunus", "stone_fruit",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
    ],
    "Citrus_Greening": [
        "rice", "oryza", "paddy",
        "corn", "maize", "zea_mays",
        "wheat", "triticum", "barley", "hordeum",
        "coffee", "coffea", "arabica", "robusta",
        "tomato", "solanum", "lycopersicum",
        "cacao", "cocoa", "theobroma",
        "apple", "malus", "pome",
        "pear", "pyrus",
        "plum", "prunus", "stone_fruit",
        "potato", "solanum_tuberosum",
        "pepper", "capsicum", "bell_pepper",
        "eggplant", "aubergine", "brinjal",
        "cucumber", "cucumis", "melon",
        "bean", "legume", "fabaceae",
    ],
}

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Danh sách CSS selector để tìm nút "Xem thêm ảnh" trên Bing
SEE_MORE_SELECTORS = [
    "input[type='button'][value='See more images']",
    "input[value='See more images']",
    "#seemorekey",
    ".b_seemore",
    ".mop",
]


# ── Hàm tiện ích nhỏ ──────────────────────────────────────────────────────────

def jitter(min_s: float = 1.0, max_s: float = 3.0) -> float:
    """Trả về độ trễ ngẫu nhiên (giây) để mô phỏng hành vi con người."""
    return random.uniform(min_s, max_s)


def random_ua() -> str:
    """Chọn ngẫu nhiên một User-Agent từ danh sách có sẵn."""
    return random.choice(USER_AGENTS)


def _should_exclude_url(url: str, query: str, class_name: str) -> bool:
    """
    Kiểm tra xem URL/query có chứa từ khóa loại trừ không.
    
    Cải tiến:
      1. Decode URL percent-encoded (ví dụ: %20 → space, %E2 → ký tự Unicode)
      2. Dùng regex word boundaries (\b) để tránh keyword overlap
         (ví dụ: "cỏ" không sẽ match "coffee")
      3. Case-insensitive comparison
    
    Trả về True nếu URL/query nên bị loại bỏ.
    """
    try:
        # Decode URL percent-encoded (ví dụ: %20 thành space, %E2 thành ký tự Unicode)
        decoded_url = unquote(url)
    except Exception:
        decoded_url = url
    
    combined_text = f"{decoded_url} {query}".lower()
    
    # Kiểm tra từ khóa loại trừ chung
    common_exclude = EXCLUDE_KEYWORDS.get("common", [])
    for keyword in common_exclude:
        keyword_lower = keyword.lower()
        # Dùng word boundary regex để tránh false positives
        # Ví dụ: "cỏ" không match "coffee", nhưng match "cỏ dại"
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, combined_text):
            return True
    
    # Kiểm tra từ khóa loại trừ riêng cho lớp cây này
    class_exclude = EXCLUDE_KEYWORDS.get(class_name, [])
    for keyword in class_exclude:
        keyword_lower = keyword.lower()
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, combined_text):
            return True
    
    return False


def _cleanup_temp_files(class_dir: Path) -> None:
    """Xóa các file tạm `_tmp_*.jpg` còn sót lại từ lần chạy bị gián đoạn trước."""
    for tmp in class_dir.glob("_tmp_*.jpg"):
        try:
            tmp.unlink()
            log.debug("Đã xóa file tạm còn sót: %s", tmp.name)
        except Exception as exc:
            log.warning("Không thể xóa file tạm %s: %s", tmp.name, exc)


def _compute_image_hash(img_path: Path) -> Optional[str]:
    """
    Tính MD5 hash của ảnh để phát hiện ảnh trùng lặp.
    
    Trả về hash string (dạng hex), hoặc None nếu lỗi.
    """
    try:
        with open(img_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as exc:
        log.debug("Không thể tính hash cho %s: %s", img_path.name, exc)
        return None


def _load_image_hashes(class_dir: Path) -> Set[str]:
    """
    Tải tập hợp các hash ảnh đã có sẵn.
    Dùng để phát hiện và loại bỏ ảnh trùng lặp từ các URL khác nhau.
    """
    hashes: Set[str] = set()
    meta_path = class_dir / "metadata.jsonl"
    
    if not meta_path.exists():
        return hashes
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Nếu metadata chứa hash, tải vào tập hợp
                    if "image_hash" in entry:
                        hashes.add(entry["image_hash"])
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        log.warning("Lỗi tải image hashes: %s", exc)
    
    return hashes


# ── Cơ chế Resume: đọc trạng thái đã tải ─────────────────────────────────────

def _load_existing_state(class_dir: Path) -> Tuple[int, int, Set[str]]:
    """
    Quét thư mục lớp để khôi phục trạng thái từ lần chạy trước.

    Trả về:
        next_idx        -- Chỉ số file tiếp theo cần dùng khi đặt tên ảnh mới.
        downloaded      -- Số ảnh thực tế đã có trong thư mục (đếm file img_*.jpg).
        downloaded_urls -- Tập hợp URL đã tải (đọc từ metadata.jsonl nếu có) để tránh tải lại.

    Lưu ý quan trọng:
        `downloaded` được tính từ số file ảnh thực tế trong thư mục,
        KHÔNG phải từ số dòng trong metadata.jsonl.
        Điều này đảm bảo tính chính xác ngay cả khi metadata.jsonl bị mất hoặc bị hỏng.
    """
    # Tìm tất cả file ảnh hợp lệ đã tải về
    existing_imgs = sorted(class_dir.glob("img_*.jpg"))
    downloaded = len(existing_imgs)  # số ảnh thực tế — nguồn sự thật duy nhất

    # Tính chỉ số file tiếp theo từ chỉ số lớn nhất đang có
    # Dùng try/except cho từng file để không bị vỡ nếu có tên file bất thường
    if existing_imgs:
        indices = []
        for p in existing_imgs:
            try:
                idx = int(p.stem.split("_")[1])
                indices.append(idx)
            except (IndexError, ValueError):
                continue  # bỏ qua file có tên không khớp pattern
        next_idx = max(indices) + 1 if indices else downloaded
    else:
        next_idx = 0

    # Đọc metadata để lấy tập URL đã tải (nếu file tồn tại)
    downloaded_urls: Set[str] = set()
    meta_path = class_dir / "metadata.jsonl"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        url = entry.get("url")
                        if url:
                            downloaded_urls.add(url)
                    except json.JSONDecodeError:
                        pass  # Bỏ qua dòng bị hỏng, không dừng chương trình
    else:
        # metadata.jsonl không tồn tại: không thể lọc URL trùng → có thể tải lại
        log.warning(
            "Không tìm thấy metadata.jsonl trong '%s'. "
            "Sẽ không thể bỏ qua URL đã tải (có thể tải lại ảnh trùng).",
            class_dir,
        )

    if downloaded > 0:
        log.info(
            "  Resume: tìm thấy %d ảnh + %d URL trong metadata → tiếp tục từ img_%06d.jpg",
            downloaded, len(downloaded_urls), next_idx,
        )

    return next_idx, downloaded, downloaded_urls


# ── ProxyPool an toàn đa luồng ────────────────────────────────────────────────

class ProxyPool:
    """
    Pool proxy xoay vòng, an toàn đa luồng, kết hợp bộ lọc kiểm tra sức khỏe tự động.

    Tất cả các thao tác công khai (get / remove / _refresh) đều được bảo vệ
    bằng một threading.Lock để an toàn khi sử dụng với ThreadPoolExecutor.

    Cách sử dụng:
        proxy = proxy_pool.get()           # lấy ngẫu nhiên proxy đang hoạt động hoặc None
        proxy_pool.remove(dead_proxy)      # loại bỏ một proxy bị lỗi trong quá trình sử dụng
    """

    _HEALTH_URL     = "https://httpbin.org/ip"   # xác minh cả nội dung phản hồi, không chỉ status code
    _HEALTH_TIMEOUT = 5    # số giây chờ tối đa cho mỗi lần kiểm tra sức khỏe
    _MIN_HEALTHY    = 3    # dừng kiểm tra khi đã tìm thấy đủ số proxy hoạt động này
    _MAX_CANDIDATES = 40   # số lượng proxy thô tối đa cần cào từ trang danh sách

    def __init__(self) -> None:
        self.active_pool: List[str] = []
        self._lock = threading.Lock()  # bảo vệ active_pool khỏi race condition

    def _scrape_candidates(self) -> List[str]:
        """Cào danh sách proxy miễn phí từ free-proxy-list.net."""
        import re
        candidates: List[str] = []
        try:
            req = urllib.request.Request(
                "https://free-proxy-list.net/",
                headers={"User-Agent": USER_AGENTS[0]},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8")
            for ip, port in re.findall(
                r"<td>(\d{1,3}(?:\.\d{1,3}){3})</td><td>(\d+)</td>", html
            )[: self._MAX_CANDIDATES]:
                candidates.append(f"http://{ip}:{port}")
        except Exception as exc:
            log.warning("Cào danh sách proxy thất bại: %s", exc)
        return candidates

    def _is_alive(self, proxy: str) -> bool:
        """
        Kiểm tra sức khỏe proxy bằng cách kết nối tới httpbin.org/ip.
        Xác minh phản hồi JSON có chứa trường 'origin' (không chỉ kiểm tra status 200).
        """
        try:
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            opener = urllib.request.build_opener(handler)
            req = urllib.request.Request(
                self._HEALTH_URL, headers={"User-Agent": USER_AGENTS[0]}
            )
            with opener.open(req, timeout=self._HEALTH_TIMEOUT) as resp:
                if resp.status != 200:
                    return False
                # Xác minh nội dung: phản hồi phải là JSON có trường 'origin'
                body = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(body)
                return "origin" in data
        except Exception:
            return False

    def _refresh(self) -> None:
        """Cào ứng viên proxy mới và kiểm tra sức khỏe để cập nhật pool."""
        log.info("Pool proxy rỗng — đang cào danh sách proxy mới ...")
        candidates = self._scrape_candidates()
        if not candidates:
            log.warning("Không tìm thấy proxy nào. Sẽ sử dụng kết nối trực tiếp làm phương án dự phòng.")
            return

        log.info(
            "Đang kiểm tra sức khỏe %d proxy ứng viên (dừng khi tìm đủ %d proxy tốt) ...",
            len(candidates), self._MIN_HEALTHY,
        )
        healthy: List[str] = []
        for proxy in candidates:
            if self._is_alive(proxy):
                healthy.append(proxy)
                log.info("  [OK] %s", proxy)
                if len(healthy) >= self._MIN_HEALTHY:
                    break

        self.active_pool = healthy
        log.info("Pool hoạt động: %d proxy đã được xác minh.", len(self.active_pool))

    def get(self) -> Optional[str]:
        """Lấy ngẫu nhiên một proxy từ pool; tự động làm mới nếu pool rỗng."""
        with self._lock:
            if not self.active_pool:
                self._refresh()
            return random.choice(self.active_pool) if self.active_pool else None

    def remove(self, proxy: str) -> None:
        """Loại bỏ proxy chết khỏi pool (an toàn đa luồng, không ném lỗi nếu không tìm thấy)."""
        with self._lock:
            try:
                self.active_pool.remove(proxy)
                log.info(
                    "Đã loại bỏ proxy chết %s (còn lại %d proxy).",
                    proxy, len(self.active_pool),
                )
            except ValueError:
                pass  # Proxy đã bị xóa bởi luồng khác trước đó — bỏ qua


# Instance proxy pool toàn cục, dùng chung cho toàn bộ chương trình
proxy_pool = ProxyPool()


# ── Session requests an toàn đa luồng ────────────────────────────────────────

# Mỗi luồng worker có session riêng để tái sử dụng kết nối TCP (connection pooling)
_thread_local = threading.local()


def _get_session():
    """
    Trả về session HTTP cho luồng hiện tại.
    
    Ưu tiên: curl_cffi (TLS fingerprinting Chrome) → requests → None (urllib fallback)
    - curl_cffi: Giả mạo chữ ký TLS của Chrome, vượt qua Cloudflare/Anti-bot tốt hơn
    - requests: Thư viện HTTP chuẩn, nhanh nhưng dễ bị phát hiện TLS
    - urllib: Fallback cuối cùng, rất chậm
    """
    # Thử curl_cffi trước (TLS fingerprinting tốt nhất)
    if CURL_CFFI_AVAILABLE:
        if not hasattr(_thread_local, "cffi_session"):
            try:
                # Tạo session curl_cffi với Browser Fingerprint của Chrome 120
                session = cffi_requests.Session(
                    impersonate="chrome120",  # Giả mạo Chrome 120
                )
                _thread_local.cffi_session = session
                log.debug("Session curl_cffi (Chrome 120 impersonate) được tạo")
                return session
            except Exception as exc:
                log.debug("Không thể tạo curl_cffi session: %s, falling back to requests", exc)
        else:
            return _thread_local.cffi_session
    
    # Fallback: requests thường
    try:
        import requests  # type: ignore
    except ImportError:
        return None

    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": random_ua()})
        _thread_local.session = session
        log.debug("Session requests được tạo")
    return _thread_local.session


# ── Hàm hỗ trợ chạy async ────────────────────────────────────────────────────

def run_async(coro):
    """
    Chạy coroutine bất đồng bộ từ code đồng bộ.
    Hoạt động trong cả script thông thường VÀ Jupyter notebook
    (yêu cầu cài thêm nest_asyncio nếu chạy trong Jupyter).
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Đang ở trong event loop (ví dụ: Jupyter) — cần nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
        except ImportError:
            log.warning("Hãy cài đặt nest_asyncio để hỗ trợ chạy nodriver trong Jupyter: pip install nest_asyncio")
        return loop.run_until_complete(coro)
    except RuntimeError:
        # Không có event loop đang chạy — tạo mới và chạy
        return asyncio.run(coro)


# ── Thu thập URL — Selenium ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
def fetch_image_urls_with_selenium(
    query: str,
    max_results: int = 100,
    headless: bool = True,
    use_proxy: bool = False,
) -> List[Tuple[str, Dict]]:
    """
    Cào URL ảnh trên Bing Images thông qua undetected_chromedriver.

    Quy trình:
      1. Mở trình duyệt Chrome ẩn danh với proxy và User-Agent ngẫu nhiên.
      2. Cuộn trang nhiều lần để tải lazy-load ảnh, click nút 'Xem thêm' nếu có.
      3. Thu thập URL ảnh full-size từ panel xem trước bên phải.
      4. Đóng trình duyệt trong khối finally để đảm bảo không rò rỉ tài nguyên.
    """
    urls_with_meta: List[Tuple[str, Dict]] = []
    driver = None
    proxy: Optional[str] = proxy_pool.get() if use_proxy else None

    try:
        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")
            log.info("Trình duyệt đang sử dụng proxy: %s", proxy)

        options.add_argument(f"--user-agent={random_ua()}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")

        driver = uc.Chrome(options=options, version_main=None)
        driver.set_page_load_timeout(30)  # Timeout 30s để tránh treo
        driver.get(f"https://www.bing.com/images/search?q={quote_plus(query)}")
        time.sleep(jitter(2, 4))
        log.info("Đã tải trang: %s", driver.title)

        # Cuộn trang xuống cuối để kích hoạt lazy-load ảnh
        last_height = driver.execute_script("return document.body.scrollHeight")
        for scroll_idx in range(40):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(jitter(1.5, 2.5))

            # Nút "Xem thêm" là tùy chọn — bỏ qua mọi lỗi; đây là tính năng bổ sung
            try:
                for selector in SEE_MORE_SELECTORS:
                    for btn in driver.find_elements(By.CSS_SELECTOR, selector):
                        if btn.is_displayed() and btn.is_enabled():
                            btn.click()
                            log.debug("Đã click 'Xem thêm' tại lần cuộn #%d", scroll_idx + 1)
                            time.sleep(jitter(2, 3))
                            break
            except Exception:
                pass

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height and scroll_idx >= 10:
                # Trang không còn nội dung mới sau nhiều lần cuộn
                break
            last_height = new_height

        # Thu thập ảnh thu nhỏ, click vào từng ảnh để lấy URL full-size
        try:
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
            log.info("Tìm thấy %d ảnh thu nhỏ (thumbnails).", len(thumbnails))
        except Exception as exc:
            log.warning("Tìm phần tử ảnh thu nhỏ thất bại: %s", exc)
            thumbnails = []

        # ── Phương án 1: Trích xuất từ thuộc tính JSON trong a.iusc (NHANH, không cần click) ──
        # Mỗi phần tử a.iusc có thuộc tính `m` chứa JSON: {"murl":"<full-size-url>", ...}
        # Đây là cách ổn định nhất vì `iusc` là lớp cấu trúc, ít thay đổi hơn
        try:
            iusc_elements = driver.find_elements(By.CSS_SELECTOR, "a.iusc")
            log.info("Phương án 1 (iusc JSON): tìm thấy %d phần tử a.iusc.", len(iusc_elements))
            for el in iusc_elements[:max_results * 2]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    m_val = el.get_attribute("m")
                    if not m_val:
                        continue
                    data = json.loads(m_val)
                    murl = data.get("murl", "")
                    if murl and murl.startswith("http"):
                        urls_with_meta.append((murl, {
                            "source": "bing_selenium_iusc",
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "url": murl,
                        }))
                except Exception:
                    continue
        except Exception as exc:
            log.debug("Phương án 1 thất bại: %s", exc)

        # ── Phương án 2 (fallback): Click thumbnail + thử nhiều selector panel xem trước ──
        # Bing hay đổi class name; dùng danh sách selector để tránh phụ thuộc vào một class duy nhất
        _PREVIEW_SELECTORS = [
            "img.n1ddgdc",        # cũ (Bing cũ)
            "img.sib_r",          # biến thể khác
            ".imgpt img",         # container ảnh trong panel
            ".nofocus img",
            ".iusc img",
            "img[class*='dlme']", # class có thể thay đổi theo phiên bản Bing
        ]
        if len(urls_with_meta) < max_results:
            log.info(
                "Phương án 2 (click thumbnail): đang thử với %d thumbnail ...",
                len(thumbnails),
            )
            for thumb in thumbnails[:max_results * 2]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    thumb.click()
                    time.sleep(0.4)
                    found = False
                    for sel in _PREVIEW_SELECTORS:
                        try:
                            for actual in driver.find_elements(By.CSS_SELECTOR, sel):
                                src = actual.get_attribute("src") or actual.get_attribute("data-src")
                                # Loại bỏ thumbnail URL (thường chứa 'th?' hoặc rất ngắn)
                                if src and src.startswith("http") and "th?" not in src and len(src) > 80:
                                    urls_with_meta.append((src, {
                                        "source": "bing_selenium_click",
                                        "query": query,
                                        "timestamp": datetime.now().isoformat(),
                                        "url": src,
                                    }))
                                    found = True
                                    break
                            if found:
                                break
                        except Exception:
                            continue
                except Exception:
                    continue

        # ── Phương án 3 (fallback cuối cùng): Lấy src từ thumbnail (chất lượng thấp hơn) ──
        if len(urls_with_meta) < max_results // 2:
            log.warning(
                "Phương án 1 & 2 thu được %d URL. Dùng phương án 3: src của thumbnail (chất lượng thấp hơn).",
                len(urls_with_meta),
            )
            for thumb in thumbnails[:max_results]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    src = thumb.get_attribute("src") or thumb.get_attribute("data-src")
                    if src and src.startswith("http"):
                        urls_with_meta.append((src, {
                            "source": "bing_selenium_thumb",
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "url": src,
                        }))
                except Exception:
                    continue

        log.info("Tổng số URL đã thu thập được: %d", len(urls_with_meta))

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return urls_with_meta


# ── Thu thập URL — Nodriver (CDP) ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def fetch_image_urls_with_nodriver(
    query: str,
    max_results: int = 100,
    headless: bool = True,
    use_proxy: bool = False,
) -> List[Tuple[str, Dict]]:
    """
    Cào URL ảnh trên Bing Images thông qua nodriver (Chrome DevTools Protocol).

    Yêu cầu: pip install nodriver
    Ưu điểm so với Selenium: không cần ChromeDriver, khó bị phát hiện hơn.
    """
    try:
        import nodriver as nd
    except ImportError:
        log.error("Chưa cài đặt nodriver. Chạy lệnh: pip install nodriver")
        return []

    urls_with_meta: List[Tuple[str, Dict]] = []
    proxy: Optional[str] = proxy_pool.get() if use_proxy else None

    browser_args = [
        "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        f"--user-agent={random_ua()}",
    ]
    if headless:
        browser_args.append("--headless")
    if proxy:
        browser_args.append(f"--proxy-server={proxy}")
        log.info("Nodriver đang sử dụng proxy: %s", proxy)

    try:
        browser = await nd.start(browser_args=browser_args)
        page = await browser.get(f"https://www.bing.com/images/search?q={quote_plus(query)}")
        await page.wait(jitter(2.0, 4.0))

        # Cuộn trang để kích hoạt lazy-load ảnh (giống logic selenium)
        for _ in range(15):
            await page.scroll_down(300)
            await page.wait(jitter(0.8, 1.5))

        # Thử click nút "Xem thêm" nếu có
        try:
            for selector in SEE_MORE_SELECTORS:
                try:
                    btn = await page.find(selector)
                    if btn:
                        await btn.click()
                        await page.wait(jitter(1.5, 2.5))
                        log.debug("Nodriver: đã click nút 'Xem thêm' với selector '%s'", selector)
                        break
                except Exception:
                    pass
        except Exception:
            pass

        # ── Phương án 1: Trích xuất từ a.iusc JSON (nhanh, không cần click) ──
        try:
            iusc_elements = await page.select_all("a.iusc")
            log.info("Nodriver phương án 1 (iusc JSON): tìm thấy %d phần tử.", len(iusc_elements))
            for el in iusc_elements[:max_results * 2]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    m_val = el.attrs.get("m", "")
                    if not m_val:
                        continue
                    data = json.loads(m_val)
                    murl = data.get("murl", "")
                    if murl and murl.startswith("http"):
                        urls_with_meta.append((murl, {
                            "source": "bing_nodriver_iusc",
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "url": murl,
                            "proxy_used": proxy or "Direct",
                        }))
                except Exception:
                    continue
        except Exception as exc:
            log.debug("Nodriver phương án 1 thất bại: %s", exc)

        # ── Phương án 2 (fallback): src của img.mimg ──
        if len(urls_with_meta) < max_results // 2:
            log.warning("Nodriver: iusc thất bại, đang thử phương án img.mimg src ...")
            try:
                for el in (await page.select_all("img.mimg"))[:max_results]:
                    if len(urls_with_meta) >= max_results:
                        break
                    src = el.attrs.get("src") or el.attrs.get("data-src")
                    if src and src.startswith("http"):
                        urls_with_meta.append((src, {
                            "source": "bing_nodriver_thumb",
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "url": src,
                            "proxy_used": proxy or "Direct",
                        }))
            except Exception as exc:
                log.warning("Lỗi trích xuất phần tử ảnh: %s", exc)

        log.info("Nodriver tổng số URL thu được: %d", len(urls_with_meta))

        await browser.stop()

    except ImportError:
        return []
    except Exception as exc:
        log.warning("Lỗi Nodriver: %s — đang thử lại ...", exc)
        raise  # Ném lại để Tenacity tự động thử lại

    return urls_with_meta


# Note: Flickr support removed — use Bing and Pinterest sources only.


# ── Thu thập URL — Pinterest (Selenium) ──────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
def fetch_image_urls_with_pinterest(
    query: str,
    max_results: int = 100,
    headless: bool = True,
    use_proxy: bool = False,
) -> List[Tuple[str, Dict]]:
    """
    Cào URL ảnh từ Pinterest thông qua Selenium.
    
    Lưu ý: Pinterest sử dụng JavaScript động, nên cần thời gian chờ lâu hơn.
    Lọc ảnh theo kích thước để tránh lấy thumbnail.
    """
    urls_with_meta: List[Tuple[str, Dict]] = []
    driver = None
    proxy: Optional[str] = proxy_pool.get() if use_proxy else None
    
    try:
        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")
            log.info("Pinterest crawler đang sử dụng proxy: %s", proxy)
        
        options.add_argument(f"--user-agent={random_ua()}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        driver = uc.Chrome(options=options, version_main=None)
        driver.set_page_load_timeout(30)  # Timeout 30s để tránh treo
        driver.get(f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}")
        time.sleep(jitter(3, 5))
        
        log.info("Đã tải Pinterest: %s", driver.title)
        
        # Cuộn trang để tải thêm ảnh (Pinterest lazy-load mạnh)
        for scroll_idx in range(30):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(jitter(1.5, 2.5))
            
            if len(urls_with_meta) >= max_results:
                break
        
        # Trích xuất URL ảnh từ các phần tử img
        try:
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img[data-test-id='organic-pin']")
            log.info("Pinterest tìm thấy %d phần tử ảnh", len(img_elements))
            
            for img in img_elements[:max_results * 2]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    src = img.get_attribute("src")
                    # Loại bỏ URL thumbnail (chứa 'originals' hoặc '236x')
                    if src and "pinterest" in src and ("236x" not in src) and src.startswith("http"):
                        # Cố gắng lấy URL kích thước lớn
                        src = src.replace("/236x/", "/600x/")
                        
                        urls_with_meta.append((src, {
                            "source": "pinterest_selenium",
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "url": src,
                            "proxy_used": proxy or "Direct",
                        }))
                except Exception:
                    continue
        except Exception as exc:
            log.warning("Lỗi trích xuất Pinterest: %s", exc)
        
        log.info("Pinterest tổng số URL thu được: %d", len(urls_with_meta))
        
    except Exception as exc:
        log.warning("Lỗi Pinterest: %s", exc)
        if proxy:
            proxy_pool.remove(proxy)
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
    
    return urls_with_meta


# ── Tải ảnh xuống ─────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
def _download_image_core(
    url: str,
    output_path: Path,
    timeout_s: int = 20,
    use_proxy: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Logic cốt lõi để tải ảnh + xác thực kích thước + mã hóa lại thành JPEG.

    Quy trình:
      1. Tải dữ liệu thô từ URL (ưu tiên requests.Session, fallback về urllib).
      2. Từ chối dữ liệu nhỏ hơn 1 KB (ảnh lỗi / placeholder).
      3. Mở bằng PIL, kiểm tra kích thước tối thiểu 100×100 px.
      4. Lưu tạm vào file `_tmp_<name>`, sau đó đổi tên nguyên tử sang tên thật.
      5. Trả về metadata gồm kích thước thực tế trên đĩa (sau khi mã hóa JPEG).

    size_bytes = kích thước file JPEG thực tế trên đĩa (KHÔNG phải kích thước dữ liệu thô tải về).
    Tự động loại bỏ proxy khỏi pool nếu gặp lỗi mạng để tự làm sạch proxy chết.
    """
    proxy: Optional[str] = proxy_pool.get() if use_proxy else None
    ua = random_ua()

    try:
        # Ưu tiên dùng requests.Session để tận dụng connection pooling
        session = _get_session()
        if session is not None:
            proxies = {"http": proxy, "https": proxy} if proxy else None
            session.headers["User-Agent"] = ua
            resp = session.get(
                url,
                proxies=proxies,
                timeout=(10, timeout_s),  # (timeout kết nối, timeout đọc)
                stream=False,
            )
            resp.raise_for_status()
            raw = resp.content
        else:
            # Fallback về urllib nếu requests chưa cài
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            if proxy:
                handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
                with urllib.request.build_opener(handler).open(req, timeout=timeout_s) as r:
                    raw = r.read()
            else:
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    raw = r.read()

    except RETRYABLE_EXCEPTIONS:
        # Loại bỏ proxy lỗi khỏi pool trước khi để Tenacity thử lại
        if proxy:
            proxy_pool.remove(proxy)
        raise

    # Từ chối dữ liệu quá nhỏ (không phải ảnh hợp lệ)
    if len(raw) < 1024:
        return False, None

    try:
        img = Image.open(BytesIO(raw))
        # Tự động sửa hướng ảnh theo metadata EXIF (ảnh chụp điện thoại hay bị nghiêng)
        # exif_transpose xoay ảnh đúng hướng trước khi convert sang RGB
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception:
        return False, None  # File không phải ảnh hợp lệ

    w, h = img.size
    # Lọc ảnh có kích thước quá nhỏ (thumbnail, icon, ảnh lỗi)
    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
        return False, None

    # ── Validate chất lượng ảnh (lọc false positives) ──
    is_valid, confidence = _validate_image_quality(img, "")  # class_name không dùng trong validate
    if not is_valid:
        log.debug("Ảnh bị loại do chất lượng thấp (confidence=%0.2f): %s", confidence, url)
        return False, None

    # Lưu tạm vào file _tmp_, sau đó đổi tên nguyên tử để tránh file bị hỏng nửa chừng
    tmp_path = output_path.parent / f"_tmp_{output_path.name}"
    try:
        with open(tmp_path, "wb") as fh:
            img.save(fh, format="JPEG", quality=95)
        tmp_path.replace(output_path)  # đổi tên nguyên tử (atomic rename)
    except Exception:
        # Dọn dẹp file tạm nếu có lỗi khi lưu
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    # ── CẢI TIẾN: Tính MD5 hash của ảnh để phát hiện trùng lặp ──
    image_hash = _compute_image_hash(output_path)
    
    # ── CẢI TIẾN: Kiểm tra file corruption - thử mở lại ảnh vừa tải ──
    try:
        with Image.open(output_path) as verify_img:
            verify_img.verify()  # Xác minh JPEG không bị corrupted
    except Exception as exc:
        log.warning("Ảnh vừa lưu bị corrupted, xóa và thử lại: %s — %s", output_path.name, exc)
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None

    return True, {
        "filename": output_path.name,
        "width": w,
        "height": h,
        "size_bytes": output_path.stat().st_size,  # kích thước thực tế trên đĩa
        "url": url,
        "proxy_used": proxy or "Direct",
        "user_agent": ua,
        "download_time": datetime.now().isoformat(),
        "confidence": confidence,  # Điểm tin cậy (0.0-1.0), cao hơn = ảnh tốt hơn
        "image_hash": image_hash,  # MD5 hash để phát hiện trùng lặp
    }


def download_image(
    url: str,
    output_path: Path,
    timeout_s: int = 20,
    use_proxy: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    Hàm bọc ngoài: dọn dẹp file tải dở nếu gặp bất kỳ lỗi nào trước khi trả về kết quả.
    An toàn đa luồng — gọi an toàn từ các worker của ThreadPoolExecutor.
    """
    try:
        return _download_image_core(url, output_path, timeout_s, use_proxy)
    except Exception as exc:
        log.debug("Tải xuống thất bại (%s): %s", url, exc)
        # Xóa file ảnh lỗi nếu đã được tạo ra (tránh file hỏng)
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return False, None


# ── Hàm validate ảnh để lọc false positives ────────────────────────────────────

def _validate_image_quality(img: Image.Image, class_name: str = "") -> Tuple[bool, float]:
    """
    Kiểm tra chất lượng ảnh để lọc ảnh fake/không liên quan.
    
    Cải tiến: Loại bỏ numpy dependency, dùng PIL thuần.
    
    Các tiêu chí:
      1. Aspect ratio hợp lý (0.2 - 5.0)
      2. Không phải ảnh quá đơn giản (solid color)
      3. Kiểm tra entropy cơ bản
    
    Trả về: (is_valid, confidence_score)
    confidence_score: 0.0-1.0, cao hơn = ảnh tốt hơn
    """
    try:
        w, h = img.size
        aspect_ratio = w / h if h > 0 else 1.0
        
        # ── Tiêu chí 1: Aspect ratio hợp lý ──
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            return False, 0.1  # Quá dài hoặc quá vuông
        
        confidence = 0.5
        
        # ── Tiêu chí 2: Aspect ratio tốt ──
        if 0.3 < aspect_ratio < 2.5:
            confidence += 0.2
        
        # ── Tiêu chí 3: Entropy - kiểm tra ảnh không quá đơn giản ──
        # Resize nhỏ để tính toán nhanh (không dùng numpy, dùng PIL.ImageStat)
        img_small = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        from PIL import ImageStat
        try:
            stat = ImageStat.Stat(img_small)
            # Tính độ lệch chuẩn của mỗi kênh màu
            stdev_list = stat.stddev if hasattr(stat, 'stddev') else []
            avg_stdev = sum(stdev_list) / len(stdev_list) if stdev_list else 0.0
            
            # Ảnh quá đơn giản (solid color): avg_stdev < 10
            if avg_stdev < 10:
                confidence -= 0.3
            elif avg_stdev > 50:
                confidence += 0.1  # Ảnh phức tạp = tốt
        except Exception:
            pass  # Bỏ qua nếu lỗi, không ảnh hưởng validation
        
        # ── Tiêu chí 4: Kiểm tra màu xanh (green hue) ──
        # Dùng PIL convert sang HSV, kiểm tra pixel count thay vì numpy array
        try:
            img_hsv = img_small.convert("HSV")
            pixels = img_hsv.getdata()
            
            # Đếm pixel có hue xanh (Green range ~60-180 trong 0-255)
            green_count = sum(1 for pixel in pixels if 60 <= pixel[0] <= 180)
            green_ratio = green_count / len(pixels) if len(pixels) > 0 else 0.0
            
            if green_ratio > 0.15:
                confidence += 0.3
            elif green_ratio < 0.05:
                confidence -= 0.2  # Quá ít green
        except Exception:
            pass  # Bỏ qua nếu lỗi
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Quyết định: nếu confidence < 0.3 → loại
        is_valid = confidence >= 0.3
        
        return is_valid, confidence
        
    except Exception as exc:
        log.debug("Lỗi validate ảnh: %s", exc)
        return True, 0.5  # Mặc định pass nếu lỗi


# ── Hàm đếm URL available từ nền tảng ───────────────────────────────────────────

def count_available_urls(
    query: str,
    platform: str = "bing",
    headless: bool = True,
    use_proxy: bool = False,
    max_check: int = 500,
) -> int:
    """
    Kiểm tra số lượng ảnh available từ một nền tảng cho một query cụ thể.
    
    Lưu ý: 
    - Bing: lấy từ HTML, có thể không chính xác 100% nhưng gần thực tế
    - Flickr: dùng API, chính xác hơn
    - Pinterest: scroll và đếm, chậm hơn
    
    Trả về: Số lượng ảnh estimate (có thể lấy được), hoặc -1 nếu lỗi
    """
    try:
        if platform.lower() == "bing":
            driver = None
            try:
                options = uc.ChromeOptions()
                if headless:
                    options.add_argument("--headless")
                options.add_argument(f"--user-agent={random_ua()}")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
                driver = uc.Chrome(options=options, version_main=None)
                driver.set_page_load_timeout(30)  # Timeout 30s để tránh treo
                driver.get(f"https://www.bing.com/images/search?q={quote_plus(query)}")
                time.sleep(jitter(1.5, 2.5))
                
                # Scroll để tải thêm ảnh
                count = 0
                for _ in range(15):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(0.5)
                    
                    # Đếm số ảnh có thể trích xuất
                    try:
                        iusc_els = driver.find_elements(By.CSS_SELECTOR, "a.iusc")
                        count = len(iusc_els)
                        if count >= max_check:
                            break
                    except:
                        pass
                
                return min(count, max_check) if count > 0 else -1
                
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
        
        elif platform.lower() == "flickr":
            session = _get_session()
            try:
                params = {
                    "method": "flickr.photos.search",
                    "api_key": FLICKR_API_KEY,
                    "text": query,
                    "per_page": 1,
                    "format": "json",
                    "nojsoncallback": 1,
                }
                headers = {"User-Agent": random_ua()}
                
                if session:
                    resp = session.get(
                        "https://www.flickr.com/services/rest/",
                        params=params,
                        timeout=15,
                        headers=headers
                    )
                else:
                    # Fallback urllib
                    raise ImportError("requests not available, skip Flickr count")
                
                data = resp.json()
                if data.get("stat") == "ok":
                    total = int(data["photos"].get("total", 0))
                    return min(total, max_check)
                return -1
                
            except Exception as exc:
                log.warning("Không thể đếm Flickr URLs: %s", exc)
                return -1
        
        elif platform.lower() == "pinterest":
            # Pinterest khó đếm chính xác → return estimate
            return 100  # Estimate mặc định cho Pinterest
        
        else:
            return -1
            
    except Exception as exc:
        log.warning("Lỗi count URLs (%s): %s", platform, exc)
        return -1


# ── Thu thập ảnh cho từng lớp ────────────────────────────────────────────────

def crawl_class_stealth(
    class_name: str,
    output_root: str = "data/raw",
    max_images: int = 1000,
    headless: bool = True,
    scraper_type: str = "selenium",
    platforms: Optional[List[str]] = None,
    use_proxy_for_browser: bool = False,
    use_proxy_for_download: bool = False,
    download_workers: int = 8,
) -> Dict:
    """
    Thu thập hình ảnh cho một lớp bệnh cây trồng cụ thể từ nhiều nền tảng.

    Tính năng:
      - Tự động tiếp tục từ điểm dừng nếu đã có ảnh/metadata từ lần trước.
      - Ghi metadata JSONL ngay sau mỗi ảnh tải thành công (an toàn khi sập).
      - Lọc URL trùng lặp qua seen_urls (áp dụng cho TẤT CẢ truy vấn của lớp này).
      - Số thứ tự file được đếm liên tục, nguyên tử qua các vòng lặp truy vấn.
      - Hỗ trợ nhiều nền tảng: Bing, Flickr, Pinterest.
      - ThreadPoolExecutor chạy song song tải xuống ảnh (tác vụ I/O-bound).

    Args:
        platforms: Danh sách nền tảng cần dùng, ví dụ ["bing", "flickr", "pinterest"].
                   Mặc định: ["bing"]

    Trả về dict gồm: số ảnh đã tải, số lần thử, metadata từng ảnh.
    """
    if platforms is None:
        platforms = ["bing"]
    
    platforms = [p.lower() for p in platforms]
    
    class_dir = Path(output_root) / class_name
    ensure_dir(str(class_dir))

    # Dọn dẹp file tạm còn sót từ lần chạy trước bị gián đoạn
    _cleanup_temp_files(class_dir)

    # Khôi phục trạng thái từ lần chạy trước (nếu có)
    next_file_idx, downloaded, seen_urls = _load_existing_state(class_dir)

    meta_path = class_dir / "metadata.jsonl"
    attempted = 0

    # Lock bảo vệ ghi metadata đồng thời từ nhiều luồng worker
    metadata_lock = threading.Lock()
    
    # ── CẢI TIẾN: Atomic counter để đảm bảo không bỏ lỗ chỉ số file ──
    # Chỉ tăng counter khi ảnh thực sự được lưu thành công, không phải khi submit task
    file_counter = next_file_idx
    file_counter_lock = threading.Lock()

    def append_metadata(entry: Dict) -> None:
        """Ghi một dòng JSON vào metadata.jsonl, an toàn đa luồng."""
        with metadata_lock:
            with open(meta_path, "a", encoding="utf-8") as _mf:
                _mf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    queries = SEARCH_QUERIES.get(class_name, [class_name])

    # ── CẢI TIẾN: Tải hash của ảnh đã có sẵn để phát hiện trùng lặp ──
    image_hashes: Set[str] = _load_image_hashes(class_dir)
    if image_hashes:
        log.info("[%s] Đã tải %d image hash từ metadata để phát hiện trùng lặp.", class_name, len(image_hashes))

    for q_idx, query in enumerate(queries):
        if downloaded >= max_images:
            break

        log.info(
            "[%s] Truy vấn %d/%d: '%s' (nền tảng: %s)",
            class_name, q_idx + 1, len(queries), query, ", ".join(platforms).upper(),
        )

        # ── Kiểm tra số ảnh available từ mỗi nền tảng trước tải ──
        platform_counts = {}
        for platform in platforms:
            remaining = max_images - downloaded
            if remaining <= 0:
                break
            
            log.info("[%s] Kiểm tra số ảnh available trên %s cho: '%s'",
                    class_name, platform.upper(), query)
            count = count_available_urls(query, platform=platform, max_check=remaining * 2)
            if count > 0:
                platform_counts[platform] = count
                log.info("[%s:%s] Có ~%d ảnh available", class_name, platform.upper(), count)
            else:
                log.warning("[%s:%s] Không thể lấy số lượng ảnh (count=%d)", 
                           class_name, platform.upper(), count)

        # Sắp xếp nền tảng theo số ảnh available (nhiều nhất trước)
        sorted_platforms = sorted(
            platforms,
            key=lambda p: platform_counts.get(p, -1),
            reverse=True
        )
        
        log.info("[%s] Thứ tự ưu tiên nền tảng: %s", class_name, 
                ", ".join(f"{p.upper()}({platform_counts.get(p, 'N/A')})" for p in sorted_platforms))

        # Duyệt qua các nền tảng theo thứ tự ưu tiên (từ nhiều ảnh nhất đến ít nhất)
        for platform in sorted_platforms:
            if downloaded >= max_images:
                break
            
            try:
                if platform == "bing":
                    if scraper_type.lower() == "nodriver":
                        urls_with_meta = run_async(fetch_image_urls_with_nodriver(
                            query, max_results=max_images * 2,
                            headless=headless, use_proxy=use_proxy_for_browser,
                        ))
                    else:
                        urls_with_meta = fetch_image_urls_with_selenium(
                            query, max_results=max_images * 2,
                            headless=headless, use_proxy=use_proxy_for_browser,
                        )
                elif platform == "pinterest":
                    urls_with_meta = fetch_image_urls_with_pinterest(
                        query, max_results=max_images * 2,
                        headless=headless, use_proxy=use_proxy_for_browser,
                    )
                else:
                    log.warning("Nền tảng không hỗ trợ: %s", platform)
                    continue
                    
            except Exception as exc:
                log.warning("Cào dữ liệu từ %s thất bại: %s", platform, exc)
                continue

            if not urls_with_meta:
                continue

            # Lọc URL trùng lặp (kể cả URL đã tải trong lần chạy trước)
            unique: List[Tuple[str, Dict]] = [
                (url, meta) for url, meta in urls_with_meta
                if url not in seen_urls and not seen_urls.add(url)  # type: ignore[func-returns-value]
            ]

            log.info(
                "[%s:%s] Lấy về %d URL → %d URL duy nhất sau khi lọc trùng.",
                platform.upper(), class_name, len(urls_with_meta), len(unique),
            )

            # Lọc URL chứa từ khóa loại trừ (để tránh ảnh từ cây khác)
            before_exclude = len(unique)
            unique = [
                (url, meta) for url, meta in unique
                if not _should_exclude_url(url, query, class_name)
            ]
            excluded_count = before_exclude - len(unique)
            if excluded_count > 0:
                log.info(
                    "[%s:%s] Loại trừ %d URL chứa từ khóa không phù hợp (ngô, sả, mía, v.v.)",
                    class_name, platform.upper(), excluded_count,
                )

            # Chỉ tải số lượng ảnh còn thiếu
            unique = unique[: max_images - downloaded]
            if not unique:
                continue

            log.info("Đang bắt đầu %d lượt tải xuống từ %s (%d luồng worker) ...", 
                     len(unique), platform.upper(), download_workers)

            futures_map: Dict = {}
            
            # ── CẢI TIẾN: Dùng file tạm để tránh lỗ lỗ chỉ số ──
            # Mỗi task tải vào file tạm, chỉ rename thành img_000000.jpg khi thành công

            with ThreadPoolExecutor(max_workers=download_workers) as pool:
                for i, (url, url_meta) in enumerate(unique):
                    # Tạo file tạm với tên random để tránh xung đột
                    tmp_idx = random.randint(0, 999999999)
                    tmp_path = class_dir / f"_tmp_download_{tmp_idx:010d}.jpg"
                    futures_map[pool.submit(download_image, url, tmp_path, 20, use_proxy_for_download)] = (
                        url_meta, tmp_path
                    )

                for future in tqdm(as_completed(futures_map), total=len(futures_map), desc=f"{class_name}:{platform}"):
                    attempted += 1
                    url_meta, tmp_path = futures_map[future]
                    try:
                        success, img_meta = future.result()
                        if success and img_meta:
                            # ── Chỉ đặt tên file cuối cùng nếu tải thành công ──
                            # ── CẢI TIẾN: Kiểm tra trùng lặp dựa trên MD5 hash ──
                            img_hash = img_meta.get("image_hash")
                            if img_hash and img_hash in image_hashes:
                                log.debug(
                                    "Ảnh trùng lặp (hash=%s): %s — bỏ qua",
                                    img_hash[:8], tmp_path.name,
                                )
                                # Xóa ảnh trùng lặp để tiết kiệm dung lượng
                                try:
                                    tmp_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                                continue
                            
                            # Thêm hash vào tập hợp để phát hiện trùng lặp sau này
                            if img_hash:
                                image_hashes.add(img_hash)
                            
                            # ── Atomic naming: lấy index, rename file, cập nhật counter ──
                            with file_counter_lock:
                                final_idx = file_counter
                                file_counter += 1
                            
                            final_path = class_dir / f"img_{final_idx:06d}.jpg"
                            try:
                                tmp_path.rename(final_path)
                            except FileExistsError:
                                # Nếu file đã tồn tại (rất hiếm), xóa tmp và bỏ qua
                                try:
                                    tmp_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                                log.warning("File %s đã tồn tại, bỏ qua task này", final_path.name)
                                continue
                            
                            # Cập nhật metadata với tên file chính xác
                            img_meta["filename"] = final_path.name
                            
                            downloaded += 1
                            # Ghi metadata ngay lập tức sau mỗi ảnh thành công (an toàn khi sập)
                            append_metadata({**url_meta, **img_meta})
                    except Exception as exc:
                        log.warning("Lỗi luồng [%s]: %s", tmp_path.name, exc)
                        # Dọn dẹp file tạm nếu lỗi
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass

            # ── CẢI TIẾN: Chỉ cập nhật next_file_idx sau khi hoàn toàn xong ──
            # Không dùng next_file_idx += len(unique) nữa
            next_file_idx = file_counter

    log.info("[%s] Hoàn thành — đã tải %d/%d ảnh từ %s.", 
             class_name, downloaded, attempted, ", ".join(platforms).upper())
    return {
        "class": class_name,
        "downloaded": downloaded,
        "attempted": attempted,
        "platforms": platforms,
        "output_dir": str(class_dir),
    }


# ── Thu thập toàn bộ bộ dữ liệu ─────────────────────────────────────────────

def crawl_all_stealth(
    output_root: str = "data/raw",
    max_images_per_class: int = 1000,
    classes: Optional[List[str]] = None,
    scraper_type: str = "selenium",
    platforms: Optional[List[str]] = None,
    use_proxy_for_browser: bool = False,
    use_proxy_for_download: bool = False,
    download_workers: int = 8,
) -> Dict:
    """
    Thu thập tuần tự tất cả (hoặc một tập hợp các lớp được chọn) bệnh cây trồng từ nhiều nền tảng.

    Tạo trước các thư mục: raw/, processed/, augmented/ để tránh lỗi đường dẫn sau này.
    Bỏ qua các lớp không có trong DATASET_CLASSES và ghi cảnh báo.
    
    Args:
        platforms: Danh sách nền tảng, mặc định: ["bing", "flickr", "pinterest"]
    """
    if platforms is None:
        platforms = ["bing", "pinterest"]
    
    root = Path(output_root)
    # Tạo trước cấu trúc thư mục dữ liệu
    for d in [root, root.parent / "processed", root.parent / "augmented"]:
        ensure_dir(str(d))

    target = classes if classes is not None else list(DATASET_CLASSES.keys())

    # Cảnh báo và loại bỏ các lớp không hợp lệ
    unknown = [c for c in target if c not in DATASET_CLASSES]
    if unknown:
        log.warning("Bỏ qua (các) lớp không xác định: %s", unknown)
    target = [c for c in target if c in DATASET_CLASSES]

    return {
        class_name: crawl_class_stealth(
            class_name=class_name,
            output_root=str(root),
            max_images=max_images_per_class,
            headless=True,
            scraper_type=scraper_type,
            platforms=platforms,
            use_proxy_for_browser=use_proxy_for_browser,
            use_proxy_for_download=use_proxy_for_download,
            download_workers=download_workers,
        )
        for class_name in target
    }


# ── Cảnh báo đạo đức ─────────────────────────────────────────────────────────

_ETHICAL_WARNING = """
╔══════════════════════════════════════════════════════════════════════╗
║              ⚠  CẢNH BÁO ĐẠO ĐỨC SỬ DỤNG CÔNG CỤ NÀY  ⚠           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Công cụ này thu thập ảnh từ Bing Images phục vụ NGHIÊN CỨU         ║
║  KHOA HỌC VÀ GIÁO DỤC PHI LỢI NHUẬN về bệnh cây trồng.             ║
║                                                                      ║
║  Người dùng có trách nhiệm:                                          ║
║    • Tuân thủ điều khoản sử dụng của Bing (ToS).                    ║
║    • Tôn trọng bản quyền của tác giả ảnh gốc.                       ║
║    • Chỉ sử dụng bộ dữ liệu cho mục đích học thuật, phi thương mại. ║
║    • Không tái phân phối ảnh mà không có sự cho phép của chủ sở hữu.║
║    • Đặt delay đủ lớn để không gây quá tải server.                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _show_ethical_warning() -> None:
    """Hiển thị cảnh báo đạo đức và yêu cầu xác nhận trước khi chạy."""
    print(_ETHICAL_WARNING)
    try:
        ans = input("Bạn có đồng ý với các điều khoản trên không? [yes/no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Chạy tự động (CI/CD) hoặc bị ngắt — mặc định từ chối
        ans = "no"
    if ans not in ("yes", "y"):
        print("Đã hủy. Vui lòng đọc kỹ điều khoản trước khi sử dụng.")
        sys.exit(0)


# ── Giao diện dòng lệnh (CLI) ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _show_ethical_warning()

    parser = argparse.ArgumentParser(
        description="Trình thu thập ảnh bệnh cây trồng ẩn danh với khả năng tiếp tục từ điểm dừng.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--class", dest="class_name", default=None,
        help="Tên một lớp cụ thể để thu thập (mặc định: thu thập tất cả các lớp). "
             f"Các lớp hợp lệ: {', '.join(DATASET_CLASSES.keys())}",
    )
    parser.add_argument(
        "--max", type=int, default=300,
        help="Số ảnh tối đa cần thu thập cho mỗi lớp (mặc định: 300). "
             "Bing thường cung cấp 100-140 ảnh mỗi query, nên với 4 query/lớp có thể đạt 300+.",
    )
    parser.add_argument(
        "--scraper", default="selenium", choices=["selenium", "nodriver"],
        help="Công cụ thu thập sử dụng: selenium (ổn định hơn) hoặc nodriver (CDP, ít bị phát hiện hơn).",
    )
    parser.add_argument(
        "--platforms", nargs="+", default=["bing", "pinterest"],
        help="Danh sách nền tảng cần dùng: bing, pinterest. "
             "Lưu ý: ShutterStock là dịch vụ trả phí nên không hỗ trợ crawl công khai. "
             "(Mặc định: bing pinterest)",
    )
    parser.add_argument(
        "--use-proxy-browser", action="store_true",
        help="Bật proxy xoay vòng cho trình duyệt thu thập URL.",
    )
    parser.add_argument(
        "--use-proxy-download", action="store_true",
        help="Bật proxy xoay vòng cho quá trình tải ảnh.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Số luồng worker tải ảnh song song.",
    )
    parser.add_argument(
        "--output", default=str(Path(__file__).parent.parent / "data" / "raw"),
        help="Thư mục đầu ra để lưu ảnh.",
    )
    args = parser.parse_args()

    # Chuẩn hóa tên nền tảng
    platforms = [p.lower() for p in args.platforms]
    valid_platforms = ["bing", "pinterest"]
    invalid = [p for p in platforms if p not in valid_platforms]
    if invalid:
        log.warning("Nền tảng không hỗ trợ (bỏ qua): %s. Hỗ trợ: %s", invalid, valid_platforms)
        platforms = [p for p in platforms if p in valid_platforms]
    
    if not platforms:
        log.error("Không có nền tảng hợp lệ. Hỗ trợ: %s", valid_platforms)
        sys.exit(1)

    results = crawl_all_stealth(
        output_root=args.output,
        max_images_per_class=args.max,
        classes=[args.class_name] if args.class_name else None,
        scraper_type=args.scraper,
        platforms=platforms,
        use_proxy_for_browser=args.use_proxy_browser,
        use_proxy_for_download=args.use_proxy_download,
        download_workers=args.workers,
    )

    print("\n── Kết quả tổng hợp ──────────────────────────────────")
    for cls, res in results.items():
        print(f"  {cls:<22} {res['downloaded']:>5} ảnh đã tải / {res['attempted']:>5} ảnh thử tải")
        print(f"    Nền tảng: {', '.join(res['platforms']).upper()}")