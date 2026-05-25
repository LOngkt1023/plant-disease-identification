#!/usr/bin/env python3
"""Pipeline: Thu thập + Làm sạch ảnh LÁ CÂY BỆNH.

Nguồn ảnh (theo thứ tự ưu tiên):
  1. Bing Images  — dùng a.iusc JSON (đáng tin nhất)
  2. DuckDuckGo   — qua duckduckgo-search API (không cần browser)
  3. Google Images — click thumbnail (chậm nhưng có ảnh unique)

Bộ lọc AI 4 tầng:
  Tầng 0 — Cơ bản        : kích thước ≥ 224, file ≥ 4KB, PIL OK
  Tầng 1 — CLIP lá thật  : "ảnh thật lá cây" vs "hình vẽ/sơ đồ/người/thức ăn"
  Tầng 2 — CLIP ảnh thật : phát hiện illustration/cartoon/diagram
  Tầng 3 — MD5 dedup     : loại trùng lặp
  Tầng 4* — Disease model : chỉ Rice — xác nhận đúng nhãn bệnh

Cài đặt:
    pip install selenium undetected-chromedriver pillow tenacity tqdm requests
    pip install transformers torch torchvision
    pip install duckduckgo-search   # nguồn DDG
    pip install curl_cffi           # tùy chọn, tăng tốc download
"""
from __future__ import annotations
import hashlib, json, logging, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, unquote
import urllib.error, urllib.request

if sys.platform.startswith("win"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Optional ──────────────────────────────────────────────────────────────────
try:
    import curl_cffi.requests as cffi_req; CURL_OK = True
except ImportError:
    CURL_OK = False

try:
    import undetected_chromedriver as uc; UC_OK = True
except ImportError:
    UC_OK = False

try:
    from duckduckgo_search import DDGS; DDG_OK = True
except ImportError:
    DDG_OK = False; log.warning("duckduckgo-search chưa cài: pip install duckduckgo-search")

from selenium import webdriver as sel_wd
from selenium.webdriver.chrome.options import Options as ChromeOpts
from selenium.webdriver.common.by import By
from PIL import Image, ImageOps
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MIN_W, MIN_H, MIN_BYTES = 224, 224, 4_096
CHROME_VERSION = 147          # google-chrome --version → đổi cho khớp

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
]
RETRYABLE = (ConnectionError, TimeoutError, OSError,
             urllib.error.URLError, urllib.error.HTTPError)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & QUERIES  — tập trung từ khóa "ảnh thực tế lá bệnh"
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CLASSES: Dict[str, str] = {
    "Rice_Healthy":    "Lúa khỏe mạnh",
    "Rice_Blast":      "Lúa bệnh đạo ôn",
    "Rice_Blight":     "Lúa bệnh bạc lá",
    "Coffee_Healthy":  "Cà phê khỏe mạnh",
    "Coffee_Rust":     "Cà phê bệnh rỉ sắt",
    "Tomato_Healthy":  "Cà chua khỏe mạnh",
    "Tomato_Blight":   "Cà chua bệnh sương mai",
    "Tomato_Curl":     "Cà chua bệnh xoăn lá",
    "Citrus_Canker":   "Cam bệnh loét",
    "Citrus_Greening": "Cam bệnh vàng lá",
}

# Mỗi lớp có 10+ query, đa ngôn ngữ, ưu tiên "close-up leaf photo"
SEARCH_QUERIES: Dict[str, List[str]] = {
    "Rice_Healthy": [
        "rice healthy leaf close up photo",
        "healthy paddy rice leaf photograph",
        "green rice plant leaf real photo",
        "oryza sativa healthy leaf image",
        "lúa khỏe mạnh lá cây ảnh thực",
        "padi sehat daun foto close up",
        "ใบข้าวสุขภาพดีภาพถ่าย",
        "tanaman padi sehat daun hijau foto",
        "rice leaf healthy macro photograph",
        "healthy rice crop leaf field photo",
    ],
    "Rice_Blast": [
        "rice blast disease leaf photo close up",
        "rice blast lesion brown spot leaf image",
        "Magnaporthe oryzae rice leaf photograph",
        "lúa đạo ôn lá bệnh ảnh thực tế",
        "rice blast fungal infection leaf",
        "bệnh đạo ôn lúa hình ảnh lá",
        "rice leaf blast necrotic lesion photo",
        "penyakit blast padi daun foto",
        "rice leaf blight blast spot real photo",
        "blast disease paddy leaf close up",
    ],
    "Rice_Blight": [
        "rice bacterial blight leaf yellow edge photo",
        "Xanthomonas oryzae rice leaf blight image",
        "lúa bạc lá bệnh lá vàng ảnh",
        "rice leaf blight water soaked lesion",
        "bệnh bạc lá lúa ảnh thực tế",
        "rice sheath blight Rhizoctonia leaf photo",
        "bacterial leaf blight rice close up",
        "hawar daun bakteri padi foto",
        "rice tungro virus yellowing leaf photo",
        "rice brown spot leaf disease image",
    ],
    "Coffee_Healthy": [
        "coffee plant healthy leaf close up photo",
        "Coffea arabica healthy green leaf",
        "coffee robusta healthy leaf image",
        "cà phê khỏe mạnh lá xanh ảnh",
        "daun kopi sehat hijau foto",
        "healthy coffee leaf macro photograph",
        "arabica coffee leaf green shiny photo",
        "coffee plant leaf no disease",
        "feuille café arabica saine photo",
        "hoja café sana verde foto",
    ],
    "Coffee_Rust": [
        "coffee leaf rust disease orange spots photo",
        "Hemileia vastatrix coffee leaf photograph",
        "coffee rust pustule orange powdery leaf",
        "bệnh rỉ sắt cà phê lá ảnh thực",
        "coffee leaf rust close up infection",
        "daun kopi karat foto penyakit",
        "roya café hoja mancha naranja foto",
        "coffee rust fungus spores leaf macro",
        "coffee arabica rust disease leaf image",
        "hemileia coffee orange urediniospores leaf",
    ],
    "Tomato_Healthy": [
        "healthy tomato leaf close up photo",
        "tomato plant green leaf no disease",
        "Solanum lycopersicum healthy leaf",
        "cà chua khỏe mạnh lá xanh ảnh",
        "tomate feuille saine verte photo",
        "tomato leaf healthy macro photograph",
        "green tomato plant leaf real photo",
        "daun tomat sehat hijau foto",
        "tomato healthy crop leaf image",
        "tomato leaf no spots healthy green",
    ],
    "Tomato_Blight": [
        "tomato late blight leaf brown lesion photo",
        "Phytophthora infestans tomato leaf image",
        "tomato early blight Alternaria leaf spot",
        "bệnh sương mai cà chua lá nâu ảnh",
        "tomate mildiou feuille nécrose photo",
        "tomato leaf blight necrosis close up",
        "late blight tomato leaf dark brown",
        "tomato blight infection leaf real photo",
        "penyakit hawar tomat daun foto",
        "tomato leaf blight water soaked spots",
    ],
    "Tomato_Curl": [
        "tomato yellow leaf curl virus TYLCV photo",
        "tomato leaf curl disease symptom image",
        "bệnh xoăn lá cà chua vàng ảnh",
        "TYLCV tomato curled leaf photograph",
        "tomato leaf curling upward yellowing photo",
        "begomovirus tomato leaf curl image",
        "tomato curl virus wrinkled leaf real",
        "tomate virus enroulement feuille photo",
        "tomato leaf curl mosaic virus photo",
        "TYLCV infected tomato leaf close up",
    ],
    "Citrus_Canker": [
        "citrus canker leaf lesion raised spot photo",
        "Xanthomonas axonopodis citrus leaf image",
        "orange lemon leaf canker brown crater",
        "bệnh loét cam quýt lá ảnh thực",
        "cancro cítrico folha mancha foto",
        "citrus bacterial canker leaf close up",
        "lime canker disease leaf photo",
        "citrus canker corky lesion leaf real",
        "kanker jeruk daun foto penyakit",
        "citrus leaf canker halo spot image",
    ],
    "Citrus_Greening": [
        "citrus greening HLB leaf blotchy mottle photo",
        "Huanglongbing citrus leaf yellowing image",
        "bệnh vàng lá greening cam quýt ảnh",
        "HLB citrus asymmetric yellowing leaf",
        "citrus greening mottled chlorotic leaf photo",
        "orange leaf greening disease real photo",
        "citrus HLB blotchy mottle symptom",
        "CVPD greening citrus leaf image",
        "lemon greening disease leaf close up",
        "citrus yellow dragon disease leaf photo",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# CLIP PROMPTS — 2 kiểm tra độc lập
# ─────────────────────────────────────────────────────────────────────────────

# Kiểm tra 1: Có phải LÁ CÂY thật không?
IS_LEAF_POS = [
    "a real close-up photograph of a plant leaf",
    "a macro photo of a diseased plant leaf",
    "a photograph of a leaf with disease spots",
    "a real photo of a green plant leaf",
]
IS_LEAF_NEG = [
    "a diagram chart or infographic about plants",
    "a hand drawn sketch or illustration of a leaf",
    "a cartoon drawing of plants",
    "a photo of a person holding something",
    "a photo of food or vegetables on a plate",
    "a photo of a rice field or farm from far away",
    "a book or document with text and images",
    "a computer generated 3D render of a plant",
    "a scientific diagram of plant anatomy",
]

# Kiểm tra 2: Có phải ẢNH THẬT (photograph) không?  
IS_PHOTO_POS = [
    "a real photograph taken with a camera",
    "a high resolution photo of a real object",
]
IS_PHOTO_NEG = [
    "a hand drawn illustration or painting",
    "a cartoon or anime drawing",
    "a vector graphic or clip art",
    "a 3D render or computer graphics",
    "a watercolor painting or sketch",
]

RICE_DISEASE_LABELS = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
RICE_CLASS_EXPECTED: Dict[str, List[str]] = {
    "Rice_Healthy": ["Healthy"],
    "Rice_Blast":   ["Blast"],
    "Rice_Blight":  ["Bacterial Blight", "Brown Spot", "Tungro"],
}

# ─────────────────────────────────────────────────────────────────────────────
# AI MODELS
# ─────────────────────────────────────────────────────────────────────────────
_mlock = threading.Lock()
_clip_model = _clip_proc = _disease_model = _disease_proc = None
CLIP_OK = DISEASE_OK = False


def _load_models():
    global _clip_model, _clip_proc, _disease_model, _disease_proc, CLIP_OK, DISEASE_OK
    with _mlock:
        if _clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                log.info("⏳ Tải CLIP...")
                _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                _clip_model.eval(); CLIP_OK = True
                log.info("✅ CLIP OK")
            except Exception as e:
                log.warning(f"CLIP lỗi: {e}")

        if _disease_model is None:
            try:
                from transformers import AutoImageProcessor, SiglipForImageClassification
                log.info("⏳ Tải Rice-Disease model...")
                _disease_proc  = AutoImageProcessor.from_pretrained("prithivMLmods/Rice-Leaf-Disease")
                _disease_model = SiglipForImageClassification.from_pretrained("prithivMLmods/Rice-Leaf-Disease")
                _disease_model.eval(); DISEASE_OK = True
                log.info("✅ Disease model OK")
            except Exception as e:
                log.warning(f"Disease model lỗi: {e}")


def _clip_ratio(img: Image.Image, pos: List[str], neg: List[str]) -> float:
    """Tính score = pos_mean / (pos_mean + neg_mean)."""
    if not CLIP_OK:
        return 1.0
    try:
        import torch
        texts = pos + neg
        inp = _clip_proc(text=texts, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = _clip_model(**inp).logits_per_image.softmax(dim=1)[0].tolist()
        pm = sum(probs[:len(pos)]) / len(pos)
        nm = sum(probs[len(pos):]) / len(neg)
        return pm / (pm + nm + 1e-8)
    except Exception:
        return 0.5


def filter_image(img: Image.Image, class_name: str,
                 leaf_thr: float = 0.40, photo_thr: float = 0.45
                 ) -> Tuple[bool, Dict]:
    """Áp dụng bộ lọc CLIP 2 tầng + disease check.
    
    leaf_thr:  ngưỡng "là lá cây thật" (cao hơn → nghiêm hơn)
    photo_thr: ngưỡng "là ảnh thật chụp bằng máy" (loại vẽ tay)
    """
    info: Dict = {}

    # Tầng 1: Lá cây thật?
    leaf_score = _clip_ratio(img, IS_LEAF_POS, IS_LEAF_NEG)
    info["clip_leaf"] = round(leaf_score, 4)
    if leaf_score < leaf_thr:
        info["reject"] = f"not_leaf({leaf_score:.2f})"
        return False, info

    # Tầng 2: Ảnh thật (không phải vẽ tay/cartoon)?
    photo_score = _clip_ratio(img, IS_PHOTO_POS, IS_PHOTO_NEG)
    info["clip_photo"] = round(photo_score, 4)
    if photo_score < photo_thr:
        info["reject"] = f"not_photo({photo_score:.2f})"
        return False, info

    # Tầng 4: Disease label (Rice only)
    if DISEASE_OK and class_name in RICE_CLASS_EXPECTED:
        try:
            import torch
            inp = _disease_proc(images=img, return_tensors="pt")
            with torch.no_grad():
                probs = torch.nn.functional.softmax(
                    _disease_model(**inp).logits, dim=1).squeeze()
            conf, idx = torch.max(probs, dim=0)
            label = RICE_DISEASE_LABELS[int(idx.item())]
            conf  = float(conf.item())
            info["disease_label"] = label
            info["disease_conf"]  = round(conf, 4)
            if label not in RICE_CLASS_EXPECTED[class_name] or conf < 0.30:
                info["reject"] = f"wrong_disease({label},{conf:.2f})"
                return False, info
        except Exception as e:
            log.debug(f"disease check error: {e}")

    return True, info


# ─────────────────────────────────────────────────────────────────────────────
# CHROME DRIVER
# ─────────────────────────────────────────────────────────────────────────────
def _make_driver(headless: bool = True) -> sel_wd.Chrome:
    ua   = random.choice(USER_AGENTS)
    args = [f"--user-agent={ua}", "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-gpu", "--window-size=1920,1080", "--lang=en-US"]
    if headless:
        args.append("--headless=new")

    if UC_OK:
        try:
            opts = ChromeOpts()
            for a in args: opts.add_argument(a)
            drv = uc.Chrome(options=opts, use_subprocess=True,
                            version_main=CHROME_VERSION)
            drv.set_page_load_timeout(30)
            return drv
        except Exception as e:
            log.warning(f"UC thất bại ({e}) → selenium")

    opts = ChromeOpts()
    for a in args: opts.add_argument(a)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    drv = sel_wd.Chrome(options=opts)
    try:
        drv.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
            {"source":"Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})
    except: pass
    drv.set_page_load_timeout(30)
    return drv


class BrowserManager:
    def __init__(self, headless=True):
        self.headless = headless
        self._drv = None
        self._lk  = threading.Lock()

    def get(self):
        with self._lk:
            if self._drv:
                try: self._drv.current_url; return self._drv
                except: self._kill()
            self._drv = _make_driver(self.headless)
            return self._drv

    def _kill(self):
        try: self._drv.quit()
        except: pass
        self._drv = None

    def quit(self):
        with self._lk: self._kill()


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: BING IMAGES  (đáng tin nhất)
# ─────────────────────────────────────────────────────────────────────────────
def _scrape_bing(query: str, driver, max_results: int = 250) -> List[Tuple[str, Dict]]:
    """Dùng a.iusc JSON — cực kỳ ổn định trên Bing."""
    results: List[Tuple[str, Dict]] = []
    seen: Set[str] = set()
    url = f"https://www.bing.com/images/search?q={quote_plus(query)}&first=1&count=150"

    try:
        driver.get(url)
        time.sleep(random.uniform(2.0, 3.0))
    except Exception as e:
        log.warning(f"Bing load lỗi: {e}")
        return results

    last_h = 0
    for _ in range(20):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.8, 1.2))
        for sel in ["input[value='See more images']", "#seemorekey", ".btn_seemore"]:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, sel)
                if btn.is_displayed(): btn.click(); time.sleep(1.0)
            except: pass
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h: break
        last_h = new_h
        if len(results) >= max_results: break

    # Phương án A: a.iusc JSON (chính)
    iusc = driver.find_elements(By.CSS_SELECTOR, "a.iusc")
    for el in iusc:
        if len(results) >= max_results: break
        try:
            m_val = el.get_attribute("m")
            if not m_val: continue
            data = json.loads(m_val)
            u = data.get("murl", "")
            if u and u.startswith("http") and u not in seen:
                seen.add(u)
                results.append((u, {"source":"bing_iusc","query":query,"url":u,
                    "title": data.get("t",""), "timestamp":datetime.now().isoformat()}))
        except: continue

    # Phương án B: img.mimg data-src (fallback)
    if len(results) < max_results // 2:
        for img in driver.find_elements(By.CSS_SELECTOR, "img.mimg"):
            if len(results) >= max_results: break
            try:
                s = img.get_attribute("data-src") or img.get_attribute("src") or ""
                if s.startswith("http") and "th?" not in s and s not in seen:
                    seen.add(s)
                    results.append((s, {"source":"bing_img","query":query,"url":s,
                        "timestamp":datetime.now().isoformat()}))
            except: continue

    log.info(f"[Bing ] '{query[:40]}' → {len(results)} URL")
    return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: DUCKDUCKGO (không cần browser)
# ─────────────────────────────────────────────────────────────────────────────
def _scrape_ddg(query: str, max_results: int = 150) -> List[Tuple[str, Dict]]:
    if not DDG_OK:
        return []
    results = []
    seen: Set[str] = set()
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=max_results):
                u = r.get("image","")
                if u and u.startswith("http") and u not in seen:
                    seen.add(u)
                    results.append((u, {"source":"ddg","query":query,"url":u,
                        "title": r.get("title",""),
                        "width": r.get("width",0), "height": r.get("height",0),
                        "timestamp":datetime.now().isoformat()}))
    except Exception as e:
        log.debug(f"DDG error: {e}")
    log.info(f"[DDG  ] '{query[:40]}' → {len(results)} URL")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: GOOGLE IMAGES (click thumbnail — chậm nhưng có ảnh unique)
# ─────────────────────────────────────────────────────────────────────────────
def _accept_consent(driver):
    for sel in ["button#L2AGLb","button.tHlp8d","[aria-label='Accept all']",
                "form[action*='consent'] button"]:
        try:
            b = driver.find_element(By.CSS_SELECTOR, sel)
            if b.is_displayed(): b.click(); time.sleep(1.5); return
        except: pass


def _scrape_google(query: str, driver, max_results: int = 150) -> List[Tuple[str, Dict]]:
    """Google Images — trích URL bằng regex page source + click thumbnail."""
    results: List[Tuple[str, Dict]] = []
    seen: Set[str] = set()

    def _add(u, src):
        if u and len(u) > 50 and u not in seen:
            seen.add(u)
            results.append((u, {"source":src,"query":query,"url":u,
                "timestamp":datetime.now().isoformat()}))

    url = (f"https://www.google.com/search?q={quote_plus(query)}"
           f"&tbm=isch&hl=en&gl=us&tbs=isz:l,itp:photo")  # itp:photo = chỉ ảnh thật
    try:
        driver.get(url); time.sleep(random.uniform(2.0, 3.0))
    except Exception as e:
        log.warning(f"Google load lỗi: {e}"); return results

    _accept_consent(driver)
    if "consent" in driver.current_url:
        return results

    # Cuộn
    last_h = 0
    for _ in range(15):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.9, 1.4))
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h: break
        last_h = new_h

    # Regex từ page source — tìm pattern "ou":"url" và ["url",w,h]
    try:
        ps = driver.page_source
        for pat in [
            r'"ou"\s*:\s*"(https?://[^"\\]{50,}\.(?:jpg|jpeg|png|webp)[^"\\]*)"',
            r'"murl"\s*:\s*"(https?://[^"\\]{50,})"',
            r'\["(https?://[^"\\]{50,}\.(?:jpg|jpeg|png|webp)[^"\\]*)",\s*\d+,\s*\d+\]',
        ]:
            for m in re.finditer(pat, ps, re.I):
                u = m.group(1)
                if "encrypted-tbn" not in u and "gstatic" not in u:
                    _add(unquote(u), "google_regex")
        log.info(f"[GGL-A] '{query[:35]}' → {len(results)} URL")
    except Exception as e:
        log.debug(f"Google regex: {e}")

    # Click thumbnail nếu regex cho ít URL
    if len(results) < 60:
        THUMB = "img.YQ4gaf, img.Q4LuWd, g-img img"
        PANEL = ["img.sFlh5c","img.iPVvYb","img[jsname='kn3ccd']",
                 ".tvh9oe img",".eHAdSb img"]
        try:
            thumbs = driver.find_elements(By.CSS_SELECTOR, THUMB)
            log.info(f"[GGL-B] {len(thumbs)} thumbnail, click...")
            for th in thumbs[:max_results*2]:
                if len(results) >= max_results: break
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", th)
                    driver.execute_script("arguments[0].click();", th)
                    time.sleep(0.35)
                    for ps in PANEL:
                        try:
                            bi = driver.find_element(By.CSS_SELECTOR, ps)
                            s = bi.get_attribute("src") or ""
                            if s.startswith("https") and "encrypted-tbn" not in s:
                                _add(s, "google_click"); break
                        except: pass
                except: continue
        except Exception as e:
            log.debug(f"Google click: {e}")
        log.info(f"[GGL-B] '{query[:35]}' → {len(results)} URL")

    return results[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SOURCE FETCH
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_all_sources(query: str, browser: BrowserManager,
                       max_per_source: int = 200) -> List[Tuple[str, Dict]]:
    """Gọi cả 3 nguồn, trả về list URL không trùng."""
    all_urls: List[Tuple[str, Dict]] = []
    seen: Set[str] = set()

    def _merge(lst):
        for u, m in lst:
            if u not in seen:
                seen.add(u); all_urls.append((u, m))

    # Bing (primary)
    try:
        _merge(_scrape_bing(query, browser.get(), max_per_source))
    except Exception as e:
        log.warning(f"Bing scrape lỗi: {e}"); browser.quit()

    # DDG (secondary, không cần browser)
    _merge(_scrape_ddg(query, max_per_source // 2))

    # Google (tertiary, chỉ nếu vẫn thiếu)
    if len(all_urls) < max_per_source // 2:
        try:
            _merge(_scrape_google(query, browser.get(), max_per_source // 2))
        except Exception as e:
            log.warning(f"Google scrape lỗi: {e}"); browser.quit()

    log.info(f"[Multi] '{query[:40]}' → {len(all_urls)} URL tổng (3 nguồn)")
    return all_urls


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD + FILTER INLINE
# ─────────────────────────────────────────────────────────────────────────────
_tl = threading.local()

def _get_sess():
    if CURL_OK:
        if not hasattr(_tl,"cs"):
            try: _tl.cs = cffi_req.Session(impersonate="chrome124")
            except: pass
        if hasattr(_tl,"cs"): return _tl.cs
    try:
        import requests
        if not hasattr(_tl,"rs"):
            s = requests.Session(); s.headers["User-Agent"] = random.choice(USER_AGENTS)
            _tl.rs = s
        return _tl.rs
    except ImportError:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1,max=6),
       retry=retry_if_exception_type(RETRYABLE))
def _fetch_raw(url: str, timeout=18) -> bytes:
    ua = random.choice(USER_AGENTS)
    s  = _get_sess()
    if s:
        s.headers["User-Agent"] = ua
        r = s.get(url, timeout=(7,timeout)); r.raise_for_status(); return r.content
    req = urllib.request.Request(url, headers={"User-Agent":ua})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

def _md5(path: Path) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path,"rb") as f:
            for ch in iter(lambda: f.read(65536), b""): h.update(ch)
        return h.hexdigest()
    except: return None


def process_one(url: str, out_path: Path, class_name: str,
                seen_hashes: Set[str], hashes_lk: threading.Lock,
                leaf_thr=0.40, photo_thr=0.45) -> Tuple[bool, Optional[Dict]]:
    """Tải + lọc 1 URL. Trả về (True, meta) nếu sạch."""
    # Tải
    try: raw = _fetch_raw(url)
    except Exception as e:
        log.debug(f"Fetch fail: {e}"); return False, None

    # Tầng 0
    if len(raw) < MIN_BYTES: return False, None
    try:
        img = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
    except: return False, None
    w, h = img.size
    if w < MIN_W or h < MIN_H: return False, None

    # Tầng 1+2: CLIP filter
    ok, info = filter_image(img, class_name, leaf_thr, photo_thr)
    if not ok:
        log.debug(f"Filter reject {info.get('reject','')} | {url[:55]}")
        return False, None

    # Tầng 3: MD5 dedup
    tmp = out_path.parent / f"_t_{out_path.name}"
    try:
        img.save(tmp, "JPEG", quality=95)
        h_md5 = _md5(tmp)
        if h_md5:
            with hashes_lk:
                if h_md5 in seen_hashes:
                    tmp.unlink(missing_ok=True); return False, None
                seen_hashes.add(h_md5)
        tmp.replace(out_path)
    except Exception as e:
        tmp.unlink(missing_ok=True); return False, None

    return True, {
        "url": url, "filename": out_path.name,
        "width": w, "height": h,
        "size_bytes": out_path.stat().st_size,
        "image_hash": h_md5,
        "download_time": datetime.now().isoformat(),
        **info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATE RESUME
# ─────────────────────────────────────────────────────────────────────────────
def _load_state(d: Path):
    existing = sorted(d.glob("img_*.jpg"))
    indices  = [int(p.stem.split("_")[1]) for p in existing
                if p.stem.split("_")[1].isdigit()]
    nxt   = (max(indices)+1) if indices else 0
    cnt   = len(existing)
    urls: Set[str] = set()
    hashes: Set[str] = set()
    meta = d / "metadata.jsonl"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("url"):    urls.add(e["url"])
                    if e.get("image_hash"): hashes.add(e["image_hash"])
                except: pass
    if cnt: log.info(f"  ▶ Resume: {cnt} ảnh sạch sẵn có")
    return nxt, cnt, urls, hashes


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE MỘT LỚP
# ─────────────────────────────────────────────────────────────────────────────
def run_class(class_name: str, output_root="data/clean", target=1000,
              headless=True, workers=10, leaf_thr=0.40, photo_thr=0.45,
              max_url_per_query=250) -> Dict:

    cls_dir = Path(output_root) / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    nxt, clean, seen_urls, seen_hashes = _load_state(cls_dir)
    hashes_lk = threading.Lock()
    meta_lk   = threading.Lock()
    cnt_lk    = threading.Lock()
    counter   = [nxt]
    meta_path = cls_dir / "metadata.jsonl"

    def save(entry):
        with meta_lk:
            with open(meta_path,"a",encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False)+"\n")

    queries = SEARCH_QUERIES.get(class_name, [class_name])
    browser = BrowserManager(headless=headless)
    attempted = 0

    log.info(f"\n{'─'*62}")
    log.info(f"  {class_name}  |  mục tiêu: {target}  |  có sẵn: {clean}")
    log.info(f"{'─'*62}")

    try:
        for qi, query in enumerate(queries):
            if clean >= target: break
            log.info(f"[{class_name}] Query {qi+1}/{len(queries)}: '{query}'")

            try:
                url_list = _fetch_all_sources(query, browser, max_url_per_query)
            except Exception as e:
                log.warning(f"Fetch sources lỗi: {e}"); browser.quit(); url_list = []

            fresh = [(u,m) for u,m in url_list if u not in seen_urls]
            for u,_ in fresh: seen_urls.add(u)
            fresh = fresh[:(target-clean)*5]   # dư 5× để bù lọc
            if not fresh:
                log.info("  Không URL mới, sang query tiếp"); continue

            log.info(f"  → {len(fresh)} URL mới | cần thêm {target-clean} ảnh")

            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for u, mo in fresh:
                    tp = cls_dir / f"_t_{random.randint(0,10**9):010d}.jpg"
                    futures[pool.submit(process_one, u, tp, class_name,
                        seen_hashes, hashes_lk, leaf_thr, photo_thr)] = (mo, tp)

                bar = tqdm(as_completed(futures), total=len(futures),
                           desc=f"  {class_name}", unit="img", leave=False)
                for fut in bar:
                    attempted += 1
                    mo, tp = futures[fut]
                    try: ok, im = fut.result()
                    except: ok, im = False, None

                    if ok and im:
                        with cnt_lk:
                            idx = counter[0]; counter[0] += 1
                        final = cls_dir / f"img_{idx:06d}.jpg"
                        try:
                            tp.rename(final); im["filename"] = final.name
                            save({**mo, **im}); clean += 1
                            bar.set_postfix(clean=clean, need=target-clean)
                        except: tp.unlink(missing_ok=True)
                    else:
                        tp.unlink(missing_ok=True)

                    if clean >= target:
                        for f in futures: f.cancel()
                        break
    finally:
        browser.quit()
        for t in cls_dir.glob("_t_*.jpg"): t.unlink(missing_ok=True)

    rate = f"{clean/attempted*100:.1f}%" if attempted else "N/A"
    log.info(f"[{class_name}] ✅ {clean}/{target} ảnh sạch | {attempted} thử | pass {rate}")
    return {"class":class_name, "clean":clean, "attempted":attempted, "dir":str(cls_dir)}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TOÀN BỘ
# ─────────────────────────────────────────────────────────────────────────────
def run_all(output_root="data/clean", target_per_class=1000,
            classes=None, headless=True, workers=10,
            leaf_thr=0.40, photo_thr=0.45) -> None:

    targets = [c for c in (classes or list(DATASET_CLASSES)) if c in DATASET_CLASSES]
    _load_models()

    log.info(f"\n🌿 PIPELINE — {len(targets)} lớp × {target_per_class} = "
             f"{len(targets)*target_per_class:,} ảnh sạch")
    log.info(f"   CLIP leaf≥{leaf_thr} photo≥{photo_thr} | workers={workers}")

    summary = {}
    for cls in targets:
        summary[cls] = run_class(cls, output_root, target_per_class,
                                 headless, workers, leaf_thr, photo_thr)

    total_c = sum(v["clean"] for v in summary.values())
    total_a = sum(v["attempted"] for v in summary.values())
    print("\n"+"="*70)
    print("  📊 KẾT QUẢ")
    print("-"*70)
    for cls, r in summary.items():
        p   = r["clean"]/target_per_class
        bar = "█"*int(p*22)+"░"*(22-int(p*22))
        flg = "✅" if r["clean"]>=target_per_class else "⚠️ "
        print(f"  {flg} {cls:<22} {r['clean']:>5}/{target_per_class}  [{bar}] {p*100:.0f}%")
    print("-"*70)
    print(f"  Tổng ảnh SẠCH : {total_c:>7,}")
    print(f"  Tổng thử tải  : {total_a:>7,}")
    if total_a: print(f"  Pass rate     : {total_c/total_a*100:.1f}%")
    print(f"  Lưu tại       : {Path(output_root).resolve()}")
    print("="*70+"\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠  Chỉ dùng cho nghiên cứu học thuật, phi thương mại              ║
║  Tuân thủ ToS các nguồn ảnh, không tái phân phối vi phạm bản quyền ║
╚══════════════════════════════════════════════════════════════════════╝""")
    if input("Bạn đồng ý? [yes/no]: ").strip().lower() not in ("yes","y"):
        sys.exit(0)

    p = argparse.ArgumentParser(description="Plant Leaf Disease Crawler + AI Cleaner")
    p.add_argument("--class",   dest="cls",   default=None)
    p.add_argument("--target",  type=int,     default=1000,
                   help="Số ảnh SẠCH/lớp (default 1000)")
    p.add_argument("--workers", type=int,     default=10)
    p.add_argument("--output",  default="../data/clean")
    p.add_argument("--leaf-thr",  type=float, default=0.40,
                   help="CLIP ngưỡng 'là lá cây' (default 0.40)")
    p.add_argument("--photo-thr", type=float, default=0.45,
                   help="CLIP ngưỡng 'ảnh thật' — loại hình vẽ (default 0.45)")
    p.add_argument("--chrome-version", type=int, default=CHROME_VERSION)
    p.add_argument("--show-browser", action="store_true")
    args = p.parse_args()
    CHROME_VERSION = args.chrome_version

    run_all(
        output_root=args.output,
        target_per_class=args.target,
        classes=[args.cls] if args.cls else None,
        headless=not args.show_browser,
        workers=args.workers,
        leaf_thr=args.leaf_thr,
        photo_thr=args.photo_thr,
    )