"""Module sinh query tìm kiếm đa ngôn ngữ cho từng lớp bệnh cây trồng.

Hỗ trợ tất cả ngôn ngữ phổ biến ở vùng trồng cây:
  - Lúa (Rice)    : Tiếng Việt, Thái, Khmer, Bahasa Indonesia/Malaysia,
                    Bengali, Hindi, Trung (Giản thể), Nhật, Hàn, Anh.
  - Cà phê (Coffee): Tiếng Việt, Amharic, Bồ Đào Nha, Tây Ban Nha,
                     Bahasa, Anh.
  - Cà chua (Tomato): Anh, Tây Ban Nha, Ý, Hindi, Tiếng Việt, Trung.
  - Cam chanh (Citrus): Tây Ban Nha, Bồ Đào Nha, Ả Rập, Trung, Thái,
                        Việt, Anh.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------

def _dedupe(items: Iterable[str]) -> List[str]:
    """Loại bỏ query trùng lặp, giữ nguyên thứ tự."""
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def _crop_group(class_name: str) -> Optional[str]:
    for group in ("Rice", "Coffee", "Tomato", "Citrus"):
        if class_name.startswith(group):
            return group
    return None


# ---------------------------------------------------------------------------
# Bộ query đa ngôn ngữ — Rice (Lúa)
# Vùng trồng: Đông Nam Á, Nam Á, Đông Á
# Ngôn ngữ: Việt, Thái, Khmer, Bahasa, Bengali, Hindi, Trung, Nhật, Hàn, Anh
# ---------------------------------------------------------------------------

_RICE_HEALTHY_QUERIES: List[str] = [
    # English
    "rice leaf close-up single",
    "Oryza sativa healthy leaf macro",
    "rice plant green leaf photo",
    "paddy leaf isolated photo",
    # Tiếng Việt
    "lá lúa cận cảnh khỏe mạnh",
    "lá cây lúa đơn",
    "lá lúa xanh chụp gần",
    # Tiếng Thái 🇹🇭
    "ใบข้าว ภาพถ่าย ใกล้ชิด",
    "ใบข้าวนา สีเขียว",
    # Bahasa Indonesia 🇮🇩 / Melayu 🇲🇾
    "daun padi close-up foto",
    "foto daun padi tunggal hijau",
    "daun tanaman padi segar",
    # Tiếng Bengali 🇧🇩
    "ধানের পাতা ক্লোজআপ সবুজ",
    "সুস্থ ধান গাছের পাতা",
    # Tiếng Hindi 🇮🇳
    "धान के स्वस्थ पत्ते की फोटो",
    "चावल के हरे पत्ते क्लोजअप",
    # Tiếng Trung (Giản thể) 🇨🇳
    "水稻叶片 特写 绿色",
    "健康水稻叶 宏观照片",
    # Tiếng Khmer 🇰🇭
    "ស្លឹកស្រូវ រូបភាព",
    # Tiếng Nhật 🇯🇵
    "稲 健康な葉 クローズアップ",
    # Tiếng Hàn 🇰🇷
    "벼 건강한 잎 근접 촬영",
]

_RICE_BLAST_QUERIES: List[str] = [
    # English
    "rice blast disease leaf lesion macro",
    "Magnaporthe oryzae rice leaf spots",
    "rice blast brown spot close-up",
    "pyricularia oryzae leaf symptoms",
    "rice leaf blast fungal disease photo",
    # Tiếng Việt
    "bệnh đạo ôn lá lúa cận cảnh",
    "lá lúa bị đạo ôn Magnaporthe",
    "vết bệnh đạo ôn trên lá lúa",
    # Tiếng Thái 🇹🇭
    "โรคไหม้ข้าว ใบ ภาพถ่าย",
    "ใบข้าวโรคไหม้ จุดสีน้ำตาล",
    # Bahasa Indonesia 🇮🇩
    "penyakit blast padi daun foto",
    "blas padi daun coklat foto",
    # Tiếng Bengali 🇧🇩
    "ধানের ব্লাস্ট রোগ পাতা ক্লোজআপ",
    # Tiếng Hindi 🇮🇳
    "धान का ब्लास्ट रोग पत्ती ক्लोजअप",
    # Tiếng Trung 🇨🇳
    "稻瘟病 叶片 症状 特写",
    "水稻稻瘟病 褐斑 特写",
    # Tiếng Khmer 🇰🇭
    "ស្លឹកស្រូវ ជំងឺ ផ្សិត",
    # Tiếng Nhật 🇯🇵
    "いもち病 葉 症状",
    # Tiếng Hàn 🇰🇷
    "벼 도열병 잎 증상 사진",
]

_RICE_BLIGHT_QUERIES: List[str] = [
    # English
    "rice bacterial blight leaf close-up",
    "Xanthomonas oryzae rice leaf lesion",
    "rice blight yellowing leaf macro",
    "BLB rice leaf wilting photo",
    "bacterial leaf blight paddy photo",
    # Tiếng Việt
    "bệnh bạc lá lúa cận cảnh",
    "lá lúa bị bạc lá Xanthomonas",
    "vết bệnh bạc lá trên lá lúa",
    # Tiếng Thái 🇹🇭
    "โรคขอบใบแห้ง ข้าว ใบ ภาพถ่าย",
    "ใบข้าวโรคขอบใบแห้ง สีเหลือง",
    # Bahasa 🇮🇩
    "hawar daun bakteri padi foto",
    "penyakit hawar padi daun foto",
    # Tiếng Bengali 🇧🇩
    "ধানের পাতা পোড়া রোগ ব্যাকটেরিয়া",
    # Tiếng Hindi 🇮🇳
    "धान का झुलसा रोग पत्ती क्लोजअप",
    # Tiếng Trung 🇨🇳
    "水稻白叶枯病 叶片 特写",
    "稻白叶枯病 症状 照片",
    # Tiếng Nhật 🇯🇵
    "白葉枯病 稲 葉 症状",
    # Tiếng Hàn 🇰🇷
    "벼 흰잎마름병 잎 사진",
]

# ---------------------------------------------------------------------------
# Coffee (Cà phê)
# Vùng trồng: ĐNA, Ethiopia, Brazil, Colombia
# ---------------------------------------------------------------------------

_COFFEE_HEALTHY_QUERIES: List[str] = [
    # English
    "healthy coffee leaf single close-up",
    "Coffea arabica green leaf macro",
    "coffee plant leaf isolated photo",
    "coffee robusta leaf photo",
    # Tiếng Việt
    "lá cà phê cận cảnh khỏe mạnh",
    "lá cây cà phê xanh tươi",
    # Bahasa 🇮🇩
    "daun kopi close-up foto segar",
    "foto daun kopi tunggal hijau",
    # Tiếng Amharic 🇪🇹
    "የቡና ቅጠል ፎቶ ቅርብ",
    # Tiếng Bồ Đào Nha 🇧🇷
    "folha de café saudável close-up",
    "folha de cafeeiro macro foto",
    # Tiếng Tây Ban Nha 🇪🇸🇨🇴
    "hoja de café sana close-up",
    "hoja planta de café macro",
    # Tiếng Trung 🇨🇳
    "咖啡树叶片 绿色 特写",
]

_COFFEE_RUST_QUERIES: List[str] = [
    # English
    "coffee leaf rust disease close-up",
    "Hemileia vastatrix coffee leaf spots",
    "coffee rust orange spots leaf macro",
    "coffee leaf rust fungal symptoms",
    # Tiếng Việt
    "bệnh rỉ sắt cà phê lá cận cảnh",
    "lá cà phê bị rỉ sắt Hemileia",
    "vết bệnh rỉ sắt trên lá cà phê",
    # Bahasa 🇮🇩
    "penyakit karat daun kopi Hemileia foto",
    "daun kopi karat oranye foto",
    # Tiếng Amharic 🇪🇹
    "የቡና ቅጠል ዝገት በሽታ ፎቶ",
    # Tiếng Tây Ban Nha 🇨🇴🇧🇷
    "roya del café hoja síntomas foto",
    "hoja de café enfermedad roya Hemileia",
    # Tiếng Bồ Đào Nha 🇧🇷
    "ferrugem do café folha foto",
    "mancha ferrugem laranja folha cafeeiro",
    # Tiếng Trung 🇨🇳
    "咖啡叶锈病 叶片 症状 特写",
    "叶锈病 咖啡 橙色斑点",
]

# ---------------------------------------------------------------------------
# Tomato (Cà chua)
# Vùng trồng: Toàn cầu
# ---------------------------------------------------------------------------

_TOMATO_HEALTHY_QUERIES: List[str] = [
    # English
    "healthy tomato leaf single close-up",
    "Solanum lycopersicum leaf macro",
    "tomato plant green leaf photo",
    # Tiếng Tây Ban Nha 🇪🇸🇲🇽
    "hoja de tomate sana close-up",
    "hoja planta tomate foto macro",
    # Tiếng Ý 🇮🇹
    "foglia pomodoro sano primo piano",
    # Tiếng Hindi 🇮🇳
    "स्वस्थ टमाटर का पत्ता क्लोजअप",
    # Tiếng Việt
    "lá cà chua cận cảnh khỏe mạnh",
    "lá cây cà chua xanh tươi",
    # Tiếng Trung 🇨🇳
    "番茄叶片 绿色 特写",
    "健康番茄叶 宏观照片",
]

_TOMATO_BLIGHT_QUERIES: List[str] = [
    # English
    "tomato late blight leaf close-up",
    "Phytophthora infestans tomato leaf lesion",
    "tomato early blight Alternaria leaf",
    "tomato blight brown spot macro photo",
    # Tiếng Tây Ban Nha 🇲🇽🇪🇸
    "tizón tardío tomate hoja foto",
    "hoja tomate enfermedad Phytophthora",
    # Tiếng Ý 🇮🇹
    "peronospora pomodoro foglia sintomi",
    # Tiếng Hindi 🇮🇳
    "टमाटर झुलसा रोग पत्ती क्लोजअप",
    # Tiếng Việt
    "bệnh sương mai cà chua lá cận cảnh",
    "lá cà chua bị sương mai Phytophthora",
    # Tiếng Trung 🇨🇳
    "番茄晚疫病 叶片 症状 特写",
    "番茄叶片病害 疫病 照片",
]

_TOMATO_CURL_QUERIES: List[str] = [
    # English
    "tomato leaf curl virus close-up",
    "TYLCV tomato curled leaf photo",
    "tomato leaf curl disease symptoms",
    "tomato yellow leaf curl virus leaf",
    # Tiếng Tây Ban Nha 🇲🇽
    "virus rizado tomate hoja foto",
    "hoja tomate virus enrollamiento",
    # Tiếng Hindi 🇮🇳
    "टमाटर पत्ती मोड़ रोग वायरस",
    # Tiếng Việt
    "bệnh xoăn lá cà chua cận cảnh",
    "lá cà chua xoăn virus TYLCV",
    # Tiếng Trung 🇨🇳
    "番茄黄化卷叶病 叶片 症状",
    "卷叶番茄 病毒 照片",
]

# ---------------------------------------------------------------------------
# Citrus (Cam chanh quýt)
# Vùng trồng: Địa Trung Hải, Mỹ Latinh, ĐNA, Trung Đông
# ---------------------------------------------------------------------------

_CITRUS_HEALTHY_QUERIES: List[str] = [
    # English
    "citrus leaf close-up single",
    "orange tree leaf macro photo",
    "healthy lemon leaf isolated",
    "lime leaf close-up photo",
    # Tiếng Tây Ban Nha 🇪🇸🇲🇽
    "hoja cítrico sano close-up",
    "hoja naranjo limón macro",
    # Tiếng Bồ Đào Nha 🇧🇷
    "folha cítrica saudável close-up foto",
    # Tiếng Ả Rập 🇪🇬🇸🇦
    "ورقة الحمضيات صورة مقربة خضراء",
    # Tiếng Trung 🇨🇳
    "柑橘叶片 绿色 特写",
    # Tiếng Thái 🇹🇭
    "ใบส้ม สุขภาพดี ภาพถ่ายใกล้ชิด",
    # Tiếng Việt
    "lá cam chanh cận cảnh",
    "lá cây có múi khỏe mạnh",
]

_CITRUS_CANKER_QUERIES: List[str] = [
    # English
    "citrus canker disease leaf close-up",
    "Xanthomonas citri leaf lesion macro",
    "orange leaf canker spots photo",
    "citrus bacterial canker symptoms",
    # Tiếng Tây Ban Nha 🇲🇽🇦🇷
    "cancro cítrico hoja lesión foto",
    "hoja cítrico mancha cancro bacteriano",
    # Tiếng Bồ Đào Nha 🇧🇷
    "cancro cítrico folha lesão foto",
    "mancha cancro folha laranjeira",
    # Tiếng Ả Rập 🇪🇬
    "قرحة الحمضيات ورقة صورة مقربة",
    # Tiếng Trung 🇨🇳
    "柑橘溃疡病 叶片 症状 特写",
    "柑橘疮痂病 叶片 照片",
    # Tiếng Việt
    "bệnh loét cam chanh lá cận cảnh",
    "lá cam bị loét Xanthomonas citri",
]

_CITRUS_GREENING_QUERIES: List[str] = [
    # English
    "citrus greening HLB disease leaf",
    "Huanglongbing citrus yellow mottled leaf",
    "citrus greening symptoms blotchy mottling",
    "HLB citrus asymmetric yellowing leaf",
    # Tiếng Tây Ban Nha 🇲🇽🇧🇷
    "enverdecimiento cítrico hoja foto",
    "HLB hoja cítrico amarillamiento asimétrico",
    # Tiếng Bồ Đào Nha 🇧🇷
    "greening cítrico folha sintomas foto",
    "huanglongbing folha citros foto",
    # Tiếng Ả Rập 🇸🇦🇪🇬
    "اخضرار الحمضيات ورقة مصفرة",
    # Tiếng Trung 🇨🇳
    "柑橘黄龙病 叶片 症状 特写",
    "黄龙病 叶片 斑驳黄化 照片",
    # Tiếng Thái 🇹🇭
    "โรคกรีนนิ่งส้ม ใบเหลือง ภาพถ่าย",
    # Tiếng Việt
    "bệnh vàng lá gân xanh HLB cam cận cảnh",
    "lá cam bị HLB vàng lá không đều",
]

# ---------------------------------------------------------------------------
# Bảng tra cứu: class_name → list[query]
# ---------------------------------------------------------------------------

_QUERY_MAP: Dict[str, List[str]] = {
    "Rice_Healthy":    _RICE_HEALTHY_QUERIES,
    "Rice_Blast":      _RICE_BLAST_QUERIES,
    "Rice_Blight":     _RICE_BLIGHT_QUERIES,
    "Coffee_Healthy":  _COFFEE_HEALTHY_QUERIES,
    "Coffee_Rust":     _COFFEE_RUST_QUERIES,
    "Tomato_Healthy":  _TOMATO_HEALTHY_QUERIES,
    "Tomato_Blight":   _TOMATO_BLIGHT_QUERIES,
    "Tomato_Curl":     _TOMATO_CURL_QUERIES,
    "Citrus_Healthy":  _CITRUS_HEALTHY_QUERIES,
    "Citrus_Canker":   _CITRUS_CANKER_QUERIES,
    "Citrus_Greening": _CITRUS_GREENING_QUERIES,
}

# Bảng profile cây trồng (giữ tương thích ngược)
CROP_PROFILE: Dict[str, Dict[str, List[str]]] = {
    "Rice": {
        "local_terms": ["rice", "Oryza sativa", "paddy", "lúa", "水稻", "稻叶", "ข้าว", "beras"],
        "healthy_terms": ["healthy leaf", "green leaf", "single leaf close-up", "leaf macro"],
        "disease_terms": [
            "leaf disease", "leaf lesion", "blast", "bacterial blight",
            "Magnaporthe oryzae", "Xanthomonas oryzae", "đạo ôn", "bạc lá",
        ],
        "countries": ["Vietnam", "India", "China", "Indonesia", "Bangladesh",
                      "Thailand", "Myanmar", "Philippines"],
    },
    "Coffee": {
        "local_terms": ["coffee", "Coffea arabica", "cà phê", "咖啡", "กาแฟ", "cafe"],
        "healthy_terms": ["healthy leaf", "green leaf", "single leaf close-up", "leaf macro"],
        "disease_terms": ["leaf rust", "Hemileia vastatrix", "orange spots", "rỉ sắt"],
        "countries": ["Brazil", "Vietnam", "Colombia", "Indonesia", "Ethiopia", "India"],
    },
    "Tomato": {
        "local_terms": ["tomato", "Solanum lycopersicum", "cà chua", "番茄", "tomate"],
        "healthy_terms": ["healthy leaf", "green leaf", "single leaf close-up", "leaf macro"],
        "disease_terms": ["late blight", "leaf curl", "Phytophthora infestans", "TYLCV"],
        "countries": ["China", "India", "Turkey", "United States", "Italy", "Mexico"],
    },
    "Citrus": {
        "local_terms": ["citrus", "orange leaf", "lemon leaf", "cam", "quýt", "柑橘"],
        "healthy_terms": ["healthy leaf", "green leaf", "single leaf close-up", "leaf macro"],
        "disease_terms": ["canker", "greening", "HLB", "Xanthomonas citri", "loét"],
        "countries": ["China", "Brazil", "India", "Mexico", "Spain", "Egypt"],
    },
}


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------

def build_search_queries(
    class_name: str,
    fallback: Optional[List[str]] = None,
    limit: int = 36,
    shuffle: bool = True,
) -> List[str]:
    """Sinh danh sách query đa ngôn ngữ cho một lớp bệnh cây trồng.

    Args:
        class_name: Tên lớp (``'Rice_Blast'``, ``'Coffee_Rust'``, v.v.).
        fallback:   Query dự phòng nếu ``class_name`` chưa có profile.
        limit:      Số query tối đa trả về (tránh rate-limit).
        shuffle:    Ngẫu nhiên hoá thứ tự để đa dạng nguồn ảnh.

    Returns:
        Danh sách query string sẵn sàng truyền vào hàm tìm kiếm.
    """
    queries = list(_QUERY_MAP.get(class_name, fallback or [class_name]))
    combined = _dedupe(list(fallback or []) + queries)

    if shuffle:
        import random
        random.shuffle(combined)

    return combined[:limit]


def list_supported_classes() -> List[str]:
    """Trả về danh sách tất cả lớp đã có profile query đa ngôn ngữ."""
    return list(_QUERY_MAP.keys())


def get_query_count(class_name: str) -> int:
    """Trả về số lượng query có sẵn cho một lớp."""
    return len(_QUERY_MAP.get(class_name, []))


# ---------------------------------------------------------------------------
# Chạy trực tiếp để xem thống kê
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sinh và xem query đa ngôn ngữ theo lớp bệnh cây trồng."
    )
    parser.add_argument("class_name", nargs="?", help="Tên lớp (bỏ qua để xem tất cả)")
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    if args.class_name:
        print(f"📝 Query cho {args.class_name} (limit={args.limit}):")
        for i, q in enumerate(
            build_search_queries(args.class_name, limit=args.limit, shuffle=not args.no_shuffle), 1
        ):
            print(f"  {i:>2}. {q}")
    else:
        print("📋 Tất cả lớp có profile query đa ngôn ngữ:\n")
        print(f"  {'Lớp':<25} {'Số query':>8}")
        print(f"  {'-'*25} {'-'*8}")
        for cls in list_supported_classes():
            print(f"  {cls:<25} {get_query_count(cls):>8}")