"""Bộ lọc từ khóa nâng cao chia sẻ giữa Crawler và Data Cleaning.

Hỗ trợ kiểm tra từ khóa tích cực/tiêu cực trên URL, tiêu đề, mô tả
để loại bỏ các ảnh không liên quan (người, món ăn, loài cây khác...).
"""

import re
from typing import Dict, List, Optional, Set
from urllib.parse import unquote


# ── Hàm loại bỏ dấu tiếng Việt ──────────────────────────────────────────────
def remove_vietnamese_diacritics(text: str) -> str:
    accents_map = {
        'a': 'áàảãạăắằẳẵặâấầẩẫậ',
        'A': 'ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ',
        'd': 'đ',
        'D': 'Đ',
        'e': 'éèẻẽẹêếềểễệ',
        'E': 'ÉÈẺẼẸÊẾỀỂỄỆ',
        'i': 'íìỉĩị',
        'I': 'ÍÌỈĨỊ',
        'o': 'óòỏõọôốồổỗộơớờởỡợ',
        'O': 'ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ',
        'u': 'úùủũụưứừửữự',
        'U': 'ÚÙỦŨỤƯỨỪỬỮỰ',
        'y': 'ýỳỷỹỵ',
        'Y': 'ÝỲỶỸỴ'
    }
    result = text
    for char, accented_chars in accents_map.items():
        for accented_char in accented_chars:
            result = result.replace(accented_char, char)
    return result


# ── Định nghĩa từ khóa Loại trừ & Tích cực (đã nới lỏng) ─────────────────
EXCLUDE_KEYWORDS: Dict[str, List[str]] = {
    "common": [
        # Đồ họa, biểu đồ
        "infographic", "diagram", "cartoon", "chart", "graph", "vector", "illustration",
        "icon", "emoji", "logo", "animation", "drawing", "sketch", "painting", "artwork",
        # Hoa (có thể gây nhầm lá)
        "flower", "blossom", "petal",
        # Người, bộ phận cơ thể
        "human", "woman", "man", "girl", "boy", "person", "people", "hand", "finger", "skin",
        "beauty", "face", "portrait", "model", "kid", "baby", "doctor", "scientist", "patient",
        # Món ăn đã nấu chín, bát đĩa
        "recipe", "salad", "cooking", "plate", "dish", "food", "soup", "curry", "pilaf",
        "cooked", "bowl", "spoon", "fork", "kitchen", "delicious", "yummy", "eat",
        # Đồ vật không liên quan
        "wood", "table", "wall", "floor", "concrete", "pot", "vase",
        "garden tool", "glove", "fertilizer", "pest control",
        # Văn bản, trình bày
        "text", "label", "poster", "banner", "infographics", "screenshot", "slide", "presentation",
        # Cây không phải mục tiêu (loại chung)
        "cần sa", "cannabis", "marijuana", "weed", "rose", "fern", "spinach", "cải",
        "publicdomainpictures",
        # ── Loại bỏ đồ uống / món chế biến từ cây trồng (giữ lại hạt thô, quả tươi) ─
        # Đồ uống cà phê
        "coffee cup", "coffee mug", "coffee drink", "espresso", "latte", "cappuccino",
        "ly cà phê", "tách cà phê",
        # Món ăn từ cà chua
        "tomato sauce", "tomato soup", "ketchup", "sinh tố cà chua",
        # Nước ép, lát cắt cam chanh
        "orange juice", "lemon juice", "lime juice", "lemonade",
        "slice of lemon", "slice of orange", "wedge of lemon",
        "lát cam", "lát chanh", "nước cam", "nước chanh",
        # Cơm, món gạo nấu chín (để lọc ảnh đồ ăn)
        "cơm", "steamed rice", "fried rice", "sushi rice",
    ]
}

# Từ khóa tích cực cho từng nhóm cây trồng (bổ sung "leaf", "foliage", "macro")
CROP_POSITIVE_KEYWORDS: Dict[str, List[str]] = {
    "Rice": [
        "rice", "oryza", "paddy", "lúa", "稻", "ข้าว", "straw",
        "blast", "blight", "bacterial", "magnaporthe", "xanthomonas", "pyricularia", "lesion",
        "leaf", "foliage", "macro", "close-up", "closeup", "detail"
    ],
    "Coffee": [
        "coffee", "coffea", "cà phê", "咖啡", "กาแฟ", "cafe", "coffe",
        "rust", "hemileia", "leaf rust", "orange spots", "rust fungus",
        "gỉ sắt", "gi sat", "rỉ sắt", "ri sat", "สนิมใบ", "铁皮病", "叶锈病", "lesion",
        "leaf", "foliage", "macro", "close-up", "closeup", "detail"
    ],
    "Tomato": [
        "tomato", "lycopersicum", "cà chua", "番茄", "西红柿", "มะเขือเทศ",
        "blight", "curl", "curled", "curling", "phytophthora", "infestans",
        "tylcv", "yellow leaf curl", "yellowing", "xoăn lá", "xoan la",
        "sương mai", "suong mai", "卷叶", "黄化卷叶", "ม้วนใบ", "晚疫病", "疫病",
        "โรคหนาวเย็น", "lesion",
        "leaf", "foliage", "macro", "close-up", "closeup", "detail"
    ],
    "Citrus": [
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine",
        "cam", "quýt", "quyt", "chanh", "ส้ม", "มะนาว",
        "canker", "greening", "hlb", "huanglongbing", "xanthomonas", "citri",
        "spots", "mottled", "asymmetric yellowing", "loét", "loet",
        "vàng lá gân xanh", "vang la gan xanh", "vang la hlb",
        "溃疡病", "黄龙病", "แผลสะดือ", "ใบแก้ว", "โรคเหลือง", "lesion",
        "leaf", "foliage", "macro", "close-up", "closeup", "detail"
    ]
}

# Từ khóa loại trừ chéo giữa các họ cây trồng (giữ nguyên)
CROP_EXCLUDE_PLANTS: Dict[str, List[str]] = {
    "Rice": [
        "corn", "maize", "zea_mays", "wheat", "barley", "oat", "rye", "coffee", "coffea",
        "tomato", "citrus", "orange", "lemon", "lime", "cacao", "cocoa", "potato", "pepper",
        "eggplant", "cucumber", "melon", "bean", "spinach", "lettuce", "cabbage", "kale",
        "cannabis", "marijuana", "hemp", "fern", "mimosa", "grape", "rose", "tulip",
        "hydrangea", "lavender", "cassava", "raspberry", "tea", "radish", "hosta", "maple",
        "clover", "banana", "oak", "pine", "palm", "bamboo", "grass", "weed", "basil",
        "mint", "strawberry", "apple", "pear", "peach", "plum", "cherry", "berry",
        "ginger", "garlic", "onion", "bina", "cải", "chuối", "cần sa"
    ],
    "Coffee": [
        "rice", "oryza", "paddy", "lúa", "corn", "maize", "wheat", "barley", "tomato",
        "citrus", "orange", "lemon", "lime", "cacao", "cocoa", "potato", "pepper",
        "eggplant", "cucumber", "melon", "bean", "spinach", "lettuce", "cabbage", "kale",
        "cannabis", "marijuana", "hemp", "fern", "mimosa", "grape", "rose", "tulip",
        "hydrangea", "lavender", "cassava", "raspberry", "tea", "radish", "hosta", "maple",
        "clover", "banana", "oak", "pine", "palm", "bamboo", "grass", "weed", "basil",
        "mint", "strawberry", "apple", "pear", "peach", "plum", "cherry", "berry",
        "ginger", "garlic", "onion"
    ],
    "Tomato": [
        "rice", "oryza", "paddy", "lúa", "corn", "maize", "wheat", "barley", "coffee",
        "coffea", "citrus", "orange", "lemon", "lime", "cacao", "cocoa", "potato", "pepper",
        "eggplant", "cucumber", "melon", "bean", "spinach", "lettuce", "cabbage", "kale",
        "cannabis", "marijuana", "hemp", "fern", "mimosa", "grape", "rose", "tulip",
        "hydrangea", "lavender", "cassava", "raspberry", "tea", "radish", "hosta", "maple",
        "clover", "banana", "oak", "pine", "palm", "bamboo", "grass", "weed", "basil",
        "mint", "strawberry", "apple", "pear", "peach", "plum", "cherry", "berry",
        "ginger", "garlic", "onion"
    ],
    "Citrus": [
        "rice", "oryza", "paddy", "lúa", "corn", "maize", "wheat", "barley", "coffee",
        "coffea", "tomato", "cacao", "cocoa", "potato", "pepper", "eggplant", "cucumber",
        "melon", "bean", "spinach", "lettuce", "cabbage", "kale", "cannabis", "marijuana",
        "hemp", "fern", "mimosa", "grape", "rose", "tulip", "hydrangea", "lavender",
        "cassava", "raspberry", "tea", "radish", "hosta", "maple", "clover", "banana",
        "oak", "pine", "palm", "bamboo", "grass", "weed", "basil", "mint", "strawberry",
        "apple", "pear", "peach", "plum", "cherry", "berry", "ginger", "garlic", "onion"
    ]
}


# ── Compile sẵn Regex ───────────────────────────────────────────────────────
def _compile_word_patterns(words_list: List[str]) -> List[re.Pattern]:
    patterns = []
    for w in words_list:
        w_clean = remove_vietnamese_diacritics(w.lower().strip())
        if not w_clean:
            continue
        pattern = re.compile(r'\b' + re.escape(w_clean) + r'\b')
        patterns.append(pattern)
    return patterns


EXCLUDE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "common": _compile_word_patterns(EXCLUDE_KEYWORDS["common"])
}

CROP_POSITIVE_PATTERNS: Dict[str, List[re.Pattern]] = {
    crop: _compile_word_patterns(keywords)
    for crop, keywords in CROP_POSITIVE_KEYWORDS.items()
}

CROP_EXCLUDE_PATTERNS: Dict[str, List[re.Pattern]] = {
    crop: _compile_word_patterns(keywords)
    for crop, keywords in CROP_EXCLUDE_PLANTS.items()
}


# ── Hàm lọc chính ───────────────────────────────────────────────────────────
def check_image_relevance(
    url: str,
    query: str,
    class_name: str,
    title: str = "",
    description: str = "",
    ignore_positive_check: bool = False
) -> tuple[bool, str]:
    """Kiểm tra độ liên quan của ảnh dựa trên URL, query, title và description.
    
    Trả về (is_relevant, reason).
    """
    try:
        decoded_url = unquote(url)
    except Exception:
        decoded_url = url

    raw_text = f"{decoded_url} {query} {title} {description}".lower()
    text_no_accent = remove_vietnamese_diacritics(raw_text)

    # 1. Loại trừ chung
    for pattern in EXCLUDE_PATTERNS["common"]:
        if pattern.search(text_no_accent):
            return False, f"common_exclude: {pattern.pattern}"

    # Xác định nhóm cây trồng
    crop_type = None
    for crop in ["Rice", "Coffee", "Tomato", "Citrus"]:
        if class_name.lower().startswith(crop.lower()):
            crop_type = crop
            break

    if not crop_type:
        return True, "unknown_crop_pass"

    # 2. Loại trừ chéo cây trồng
    crop_excludes = CROP_EXCLUDE_PATTERNS.get(crop_type, [])
    for pattern in crop_excludes:
        if pattern.search(text_no_accent):
            return False, f"crop_exclude_{crop_type}: {pattern.pattern}"

    # 3. Kiểm tra từ khóa tích cực (dùng cả query, URL, title, description)
    positive_raw = f"{decoded_url} {query} {title} {description}".lower()
    positive_text = remove_vietnamese_diacritics(positive_raw)
    positive_patterns = CROP_POSITIVE_PATTERNS.get(crop_type, [])

    has_positive = any(pattern.search(positive_text) for pattern in positive_patterns)

    if not has_positive:
        if ignore_positive_check:
            return True, "positive_bypassed_by_quality"
        else:
            return False, f"no_positive_keywords_for_{crop_type}"

    return True, "relevant"
