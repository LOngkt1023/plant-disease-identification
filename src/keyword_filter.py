"""Bộ lọc từ khóa nâng cao chia sẻ giữa Crawler và Data Cleaning.

Hỗ trợ kiểm tra từ khóa tích cực/tiêu cực trên URL, tiêu đề, mô tả
để loại bỏ các ảnh không liên quan (người, món ăn, loài cây khác...).
"""

import re
from typing import Dict, List, Optional, Set

# ── Hàm loại bỏ dấu tiếng Việt để so khớp chính xác ──────────────────────────

def remove_vietnamese_diacritics(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu."""
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


# ── Định nghĩa từ khóa Loại trừ & Tích cực ─────────────────────────────────────

EXCLUDE_KEYWORDS: Dict[str, List[str]] = {
    # Chung cho tất cả: loại bỏ infographic, diagram, cartoon, vector, người, món ăn, đồ vật
    "common": [
        "infographic", "diagram", "cartoon", "chart", "graph", "vector", "illustration",
        "icon", "emoji", "logo", "animation", "drawing", "sketch", "painting", "artwork",
        "flower", "blossom", "petal",
        "human", "woman", "man", "girl", "boy", "person", "people", "hand", "finger", "skin",
        "beauty", "face", "portrait", "model", "kid", "baby", "doctor", "scientist", "patient",
        "recipe", "salad", "cooking", "plate", "dish", "food", "soup", "curry", "pilaf",
        "rice cooked", "cooked", "bowl", "spoon", "fork", "kitchen", "delicious", "yummy", "eat",
        "wood", "table", "wall", "floor", "concrete", "ground", "dirt", "soil", "pot", "vase",
        "garden tool", "glove", "fertilizer", "pest control",
        "text", "label", "poster", "banner", "infographics", "screenshot", "slide", "presentation",
        "cần sa", "cannabis", "marijuana", "weed", "rose", "fern", "spinach", "cải", "publicdomainpictures",
        # ── Loại bỏ hạt/quả/sản phẩm chế biến (mục tiêu chỉ là LÁ CÂY) ──────────────
        # Hạt lúa, thóc, gạo, cơm
        "rice grain", "grain of rice", "rice seed", "rice seeds", "grains", "grain",
        "hạt lúa", "hạt gạo", "thóc", "cơm", "white rice", "brown rice", "sushi rice",
        "steamed rice", "fried rice", "raw rice",
        # Hạt/cốc/thức uống cà phê
        "coffee bean", "coffee beans", "coffee cup", "coffee drink", "coffee drink",
        "coffee mug", "coffee powder", "coffee grounds", "espresso", "latte", "cappuccino",
        "hạt cà phê", "ly cà phê", "tách cà phê", "cà phê hạt", "cà phê rang", "cà phê rang xay",
        # Quả cà chua chín / sản phẩm cà chua
        "tomato fruit", "tomato fruits", "tomato sauce", "tomato soup", "ketchup",
        "cherry tomato", "tomatoes", "quả cà chua", "trái cà chua", "cà chua quả",
        "cà chua chín", "sinh tố cà chua",
        # Quả cam/chanh/quýt chín / nước ép
        "orange fruit", "lemon fruit", "lime fruit", "orange slice", "lemon slice",
        "lime slice", "orange juice", "lemon juice", "lime juice", "lemonade",
        "slice of lemon", "slice of orange", "wedge of lemon",
        "quả cam", "trái cam", "cam quả", "quả chanh", "trái chanh", "lát cam", "lát chanh",
        "nước cam", "nước chanh",
        # Từ khóa quả/trái cây chung
        "fruit", "fruits", "quả", "trái cây"
    ]
}

# Từ khóa tích cực của nhóm cây trồng (được bổ sung thêm từ khóa bệnh đặc trưng làm boost)
CROP_POSITIVE_KEYWORDS: Dict[str, List[str]] = {
    "Rice": [
        "rice", "oryza", "paddy", "lúa", "稻", "ข้าว", "straw",
        "blast", "blight", "bacterial", "magnaporthe", "xanthomonas", "pyricularia", "lesion"
    ],
    "Coffee": [
        "coffee", "coffea", "cà phê", "咖啡", "กาแฟ", "cafe", "coffe",
        "rust", "hemileia", "leaf rust", "orange spots", "rust fungus", "gỉ sắt", "gi sat", "rỉ sắt", "ri sat", "สนิมใบ", "铁皮病", "叶锈病", "lesion"
    ],
    "Tomato": [
        "tomato", "lycopersicum", "cà chua", "番茄", "西红柿", "มะเขือเทศ",
        "blight", "curl", "curled", "curling", "phytophthora", "infestans", "tylcv", "yellow leaf curl", "yellowing", "xoăn lá", "xoan la", "sương mai", "suong mai", "卷叶", "黄化卷叶", "ม้วนใบ", "晚疫病", "疫病", "โรคหนาวเย็น", "lesion"
    ],
    "Citrus": [
        "citrus", "orange", "lemon", "lime", "mandarin", "tangerine", "cam", "quýt", "quyt", "chanh", "ส้ม", "มะนาว",
        "canker", "greening", "hlb", "huanglongbing", "xanthomonas", "citri", "spots", "mottled", "asymmetric yellowing", "loét", "loet", "vàng lá gân xanh", "vang la gan xanh", "vang la hlb", "溃疡病", "黄龙病", "แผลสะดือ", "ใบแก้ว", "โรคเหลือง", "lesion"
    ]
}

# Từ khóa loại trừ chéo giữa các họ cây trồng + các loại cây cảnh/cỏ dại phổ biến
CROP_EXCLUDE_PLANTS: Dict[str, List[str]] = {
    "Rice": [
        "corn", "maize", "zea_mays", "wheat", "barley", "oat", "rye", "coffee", "coffea", "tomato", 
        "citrus", "orange", "lemon", "lime", "cacao", "cocoa", "potato", "pepper", "eggplant", 
        "cucumber", "melon", "bean", "spinach", "lettuce", "cabbage", "kale", "cannabis", "marijuana", 
        "hemp", "fern", "mimosa", "grape", "rose", "tulip", "hydrangea", "lavender", "cassava", 
        "raspberry", "tea", "radish", "hosta", "maple", "clover", "banana", "oak", "pine", "palm", 
        "bamboo", "grass", "weed", "basil", "mint", "strawberry", "apple", "pear", "peach", "plum", 
        "cherry", "berry", "ginger", "garlic", "onion", "bina", "cải", "chuối", "cần sa"
    ],
    "Coffee": [
        "rice", "oryza", "paddy", "lúa", "corn", "maize", "wheat", "barley", "tomato", "citrus", 
        "orange", "lemon", "lime", "cacao", "cocoa", "potato", "pepper", "eggplant", "cucumber", 
        "melon", "bean", "spinach", "lettuce", "cabbage", "kale", "cannabis", "marijuana", "hemp", 
        "fern", "mimosa", "grape", "rose", "tulip", "hydrangea", "lavender", "cassava", "raspberry", 
        "tea", "radish", "hosta", "maple", "clover", "banana", "oak", "pine", "palm", "bamboo", 
        "grass", "weed", "basil", "mint", "strawberry", "apple", "pear", "peach", "plum", "cherry", 
        "berry", "ginger", "garlic", "onion"
    ],
    "Tomato": [
        "rice", "oryza", "paddy", "lúa", "corn", "maize", "wheat", "barley", "coffee", "coffea", 
        "citrus", "orange", "lemon", "lime", "cacao", "cocoa", "potato", "pepper", "eggplant", 
        "cucumber", "melon", "bean", "spinach", "lettuce", "cabbage", "kale", "cannabis", "marijuana", 
        "hemp", "fern", "mimosa", "grape", "rose", "tulip", "hydrangea", "lavender", "cassava", 
        "raspberry", "tea", "radish", "hosta", "maple", "clover", "banana", "oak", "pine", "palm", 
        "bamboo", "grass", "weed", "basil", "mint", "strawberry", "apple", "pear", "peach", "plum", 
        "cherry", "berry", "ginger", "garlic", "onion"
    ],
    "Citrus": [
        "rice", "oryza", "paddy", "lúa", "corn", "maize", "wheat", "barley", "coffee", "coffea", 
        "tomato", "cacao", "cocoa", "potato", "pepper", "eggplant", "cucumber", "melon", "bean", 
        "spinach", "lettuce", "cabbage", "kale", "cannabis", "marijuana", "hemp", "fern", "mimosa", 
        "grape", "rose", "tulip", "hydrangea", "lavender", "cassava", "raspberry", "tea", "radish", 
        "hosta", "maple", "clover", "banana", "oak", "pine", "palm", "bamboo", "grass", "weed", 
        "basil", "mint", "strawberry", "apple", "pear", "peach", "plum", "cherry", "berry", "ginger", 
        "garlic", "onion"
    ]
}


# ── Compile sẵn các Regex để tăng hiệu năng tối đa ───────────────────────────

def _compile_word_patterns(words_list: List[str]) -> List[re.Pattern]:
    """Compile các từ khóa thành danh sách regex với word boundary."""
    patterns = []
    for w in words_list:
        w_clean = remove_vietnamese_diacritics(w.lower().strip())
        if not w_clean:
            continue
        # Dùng \b để khớp chính xác từ, tránh khớp một phần từ (ví dụ: 'pea' trong 'peach')
        pattern = re.compile(r'\b' + re.escape(w_clean) + r'\b')
        patterns.append(pattern)
    return patterns


# Tải trước và tối ưu hóa các Regex
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


# ── Hàm lọc chính ────────────────────────────────────────────────────────────

def check_image_relevance(
    url: str,
    query: str,
    class_name: str,
    title: str = "",
    description: str = "",
    ignore_positive_check: bool = False
) -> tuple[bool, str]:
    """Kiểm tra độ liên quan của ảnh dựa trên URL, query, title và description.
    
    Hỗ trợ chuẩn hóa tiếng Việt có dấu sang không dấu trước khi so khớp.
    Ưu tiên lọc tiêu cực trước (Negative). Nếu không dính tiêu cực, kiểm tra tích cực (Positive).
    
    Trả về:
        (is_relevant, reason) -- True nếu ảnh hợp lệ, False kèm lý do nếu bị loại bỏ.
    """
    from urllib.parse import unquote
    
    try:
        decoded_url = unquote(url)
    except Exception:
        decoded_url = url
        
    # Tạo chuỗi văn bản tổng hợp và chuẩn hóa không dấu
    raw_text = f"{decoded_url} {query} {title} {description}".lower()
    text_no_accent = remove_vietnamese_diacritics(raw_text)
    
    # 1. Kiểm tra từ khóa loại trừ chung (Common Excludes)
    for pattern in EXCLUDE_PATTERNS["common"]:
        if pattern.search(text_no_accent):
            return False, f"common_exclude: {pattern.pattern}"
            
    # Xác định nhóm cây trồng (Rice, Coffee, Tomato, Citrus)
    crop_type = None
    for crop in ["Rice", "Coffee", "Tomato", "Citrus"]:
        if class_name.lower().startswith(crop.lower()):
            crop_type = crop
            break
            
    if not crop_type:
        # Nếu không thuộc nhóm cây trồng nào đã biết, cho qua nếu không dính loại trừ chung
        return True, "unknown_crop_pass"
        
    # 2. Kiểm tra từ khóa loại trừ riêng của loài cây trồng (Crop Specific Exclude)
    crop_excludes = CROP_EXCLUDE_PATTERNS.get(crop_type, [])
    for pattern in crop_excludes:
        if pattern.search(text_no_accent):
            return False, f"crop_exclude_{crop_type}: {pattern.pattern}"
            
    # 3. Kiểm tra từ khóa tích cực (Positive Boost)
    # Lọc tích cực chỉ áp dụng trên URL, Title và Description (không bao gồm Query vì Query luôn chứa tên cây)
    positive_raw = f"{decoded_url} {title} {description}".lower()
    positive_text = remove_vietnamese_diacritics(positive_raw)
    
    positive_patterns = CROP_POSITIVE_PATTERNS.get(crop_type, [])
    has_positive = False
    
    for pattern in positive_patterns:
        if pattern.search(positive_text):
            has_positive = True
            break
            
    # Quyết định cuối cùng:
    # - Nếu có từ khóa tích cực: Giữ lại (Đạt).
    # - Nếu KHÔNG có từ khóa tích cực:
    #   + Nếu ignore_positive_check = True (ảnh có chỉ số plant_ratio/confidence cực tốt từ bộ lọc màu sắc): Giữ lại.
    #   + Nếu ignore_positive_check = False: Loại bỏ. Ảnh phải chứa ít nhất 1 từ khóa tích cực
    #     của nhóm cây trồng tương ứng để đảm bảo chất lượng bộ dữ liệu.
    if not has_positive:
        if ignore_positive_check:
            # Chỉ cho qua khi bộ lọc màu sắc đã xác nhận đây là ảnh lá cây chất lượng tốt
            return True, "positive_bypassed_by_quality"
        else:
            # Không có từ khóa tích cực và không được bỏ qua kiểm tra → loại bỏ
            return False, f"no_positive_keywords_for_{crop_type}"
        
    return True, "relevant"
