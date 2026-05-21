#!/usr/bin/env python3
"""Trình thu thập ảnh bệnh cây trồng ẩn danh – Đa nguồn.

Tính năng chính:
- Nguồn tìm kiếm: Bing (Selenium) + Google Images (Selenium) + DuckDuckGo (API)
- ProxyPool (cào proxy miễn phí, kiểm tra sức khỏe, thread-safe)
- Tái sử dụng trình duyệt (BrowserManager)
- Resume, ghi metadata liên tục, đặt tên nguyên tử
- Lọc ảnh: kích thước, plant ratio, edge detection, từ khóa loại trừ

Phụ thuộc bổ sung:
    pip install duckduckgo-search
"""

import json
import logging
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, unquote
import urllib.error
import urllib.request

from .utils import compute_md5
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ----------------------------------------------------------------------
# Cấu hình logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ----------------------------------------------------------------------
# Kiểm tra thư viện
# ----------------------------------------------------------------------
try:
    import curl_cffi.requests as cffi_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False
    log.debug("curl_cffi không có – dùng requests thường")

try:
    import undetected_chromedriver as uc
except ImportError:
    log.warning("undetected_chromedriver chưa cài – chuyển sang selenium thuần")
    from selenium import webdriver as uc
from selenium.webdriver.common.by import By

try:
    from .utils import ensure_dir
except (ImportError, ValueError):
    from utils import ensure_dir

try:
    from .search_profiles import build_search_queries
except (ImportError, ValueError):
    from search_profiles import build_search_queries


SOURCE_ALIASES: Dict[str, str] = {
    "bing": "bing",
    "bing-images": "bing",
    "bingimages": "bing",
    "google": "google",
    "google-images": "google",
    "googleimages": "google",
    "ddg": "ddg",
    "duckduckgo": "ddg",
    "duckduckgo-images": "ddg",
    "duckduckgoimages": "ddg",
}

# ----------------------------------------------------------------------
# Hằng số
# ----------------------------------------------------------------------
RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError,
                        urllib.error.URLError, urllib.error.HTTPError)

# Các lớp dữ liệu
DATASET_CLASSES: Dict[str, str] = {
    "Rice_Healthy": "Lúa khỏe mạnh",
    "Rice_Blast": "Lúa bệnh đạo ôn",
    "Rice_Blight": "Lúa bệnh bạc lá",
    "Coffee_Healthy": "Cà phê khỏe mạnh",
    "Coffee_Rust": "Cà phê bệnh rỉ sắt",
    "Tomato_Healthy": "Cà chua khỏe mạnh",
    "Tomato_Blight": "Cà chua bệnh sương mai",
    "Tomato_Curl": "Cà chua bệnh xoăn lá",
    "Citrus_Canker": "Cam bệnh loét",
    "Citrus_Greening": "Cam bệnh vàng lá",
}

# Search queries are defined centrally in src/search_profiles.py

# Use central keyword/exclude definitions from keyword_filter
try:
    from .keyword_filter import EXCLUDE_PATTERNS, CROP_EXCLUDE_PATTERNS, remove_vietnamese_diacritics
except Exception:
    from keyword_filter import EXCLUDE_PATTERNS, CROP_EXCLUDE_PATTERNS, remove_vietnamese_diacritics

from .config import USER_AGENTS, SEE_MORE_SELECTORS, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT

# ----------------------------------------------------------------------
# Hàm tiện ích
# ----------------------------------------------------------------------
def random_ua() -> str:
    return random.choice(USER_AGENTS)

def _should_exclude_url(url: str, query: str, class_name: str) -> bool:
    """Kiểm tra từ khóa loại trừ trên URL và query."""
    try:
        decoded = unquote(url).lower()
    except Exception:
        decoded = url.lower()
    text = f"{decoded} {query.lower()}"
    # chuẩn hoá không dấu để khớp regex đã compile
    text_no_accent = remove_vietnamese_diacritics(text)
    # Tìm nhóm cây từ class_name
    crop_group = None
    for c in ["Rice", "Coffee", "Tomato", "Citrus"]:
        if class_name.startswith(c):
            crop_group = c
            break
    # Kiểm tra common using compiled patterns
    for pattern in EXCLUDE_PATTERNS.get("common", []):
        if pattern.search(text_no_accent):
            return True
    # Kiểm tra theo nhóm using compiled crop-specific patterns
    if crop_group and crop_group in CROP_EXCLUDE_PATTERNS:
        for pattern in CROP_EXCLUDE_PATTERNS[crop_group]:
            if pattern.search(text_no_accent):
                return True
    return False

def _cleanup_temp_files(class_dir: Path) -> None:
    for tmp in class_dir.glob("_tmp_*.jpg"):
        try:
            tmp.unlink()
        except Exception:
            pass

def _compute_image_hash(img_path: Path) -> Optional[str]:
    try:
        return compute_md5(img_path)
    except Exception:
        return None

def _load_image_hashes(class_dir: Path) -> Set[str]:
    hashes = set()
    meta_path = class_dir / "metadata.jsonl"
    if not meta_path.exists():
        return hashes
    try:
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if "image_hash" in entry:
                            hashes.add(entry["image_hash"])
                    except:
                        pass
    except Exception:
        pass
    return hashes

def _load_existing_state(class_dir: Path) -> Tuple[int, int, Set[str]]:
    existing = sorted(class_dir.glob("img_*.jpg"))
    downloaded = len(existing)
    if existing:
        indices = []
        for p in existing:
            try:
                idx = int(p.stem.split("_")[1])
                indices.append(idx)
            except:
                continue
        next_idx = max(indices) + 1 if indices else downloaded
    else:
        next_idx = 0
    downloaded_urls = set()
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
                    except:
                        pass
    else:
        log.warning(f"Không có metadata.jsonl trong {class_dir}")
    if downloaded > 0:
        log.info(f"  Resume: {downloaded} ảnh, tiêp tục từ img_{next_idx:06d}.jpg")
    return next_idx, downloaded, downloaded_urls

# Proxy pool implementation moved to src/proxy.py
try:
    from .proxy import proxy_pool
except Exception:
    from proxy import proxy_pool

# ----------------------------------------------------------------------
# Session HTTP (thread‑local)
# ----------------------------------------------------------------------
_thread_local = threading.local()

def _get_session():
    if CURL_CFFI_AVAILABLE:
        if not hasattr(_thread_local, "cffi_session"):
            try:
                session = cffi_requests.Session(impersonate="chrome120")
                _thread_local.cffi_session = session
                return session
            except Exception:
                pass
        else:
            return _thread_local.cffi_session
    try:
        import requests
        if not hasattr(_thread_local, "session"):
            session = requests.Session()
            session.headers.update({"User-Agent": random_ua()})
            _thread_local.session = session
        return _thread_local.session
    except ImportError:
        return None

# ----------------------------------------------------------------------
# Browser manager cho undetected_chromedriver
# ----------------------------------------------------------------------
def _init_uc_chrome(headless: bool, proxy: Optional[str], ua: str) -> uc.Chrome:
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    try:
        return uc.Chrome(options=options, version_main=None)
    except Exception as e:
        match = re.search(r"Current browser version is (\d+)\.", str(e))
        if match:
            major = int(match.group(1))
            return uc.Chrome(options=options, version_main=major)
        raise

class BrowserManager:
    def __init__(self, headless: bool = True, use_proxy: bool = False):
        self.driver = None
        self.headless = headless
        self.use_proxy = use_proxy

    def get_driver(self):
        if self.driver is not None:
            try:
                self.driver.current_url
            except Exception:
                self.quit()
        if self.driver is None:
            proxy = proxy_pool.get() if self.use_proxy else None
            if proxy:
                log.info(f"Browser dùng proxy: {proxy}")
            ua = random_ua()
            self.driver = _init_uc_chrome(self.headless, proxy, ua)
            self.driver.set_page_load_timeout(30)
        return self.driver

    def quit(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

# ----------------------------------------------------------------------
# Fetch URLs from Bing (Selenium)
# ----------------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS))
def fetch_image_urls_with_selenium(
    query: str,
    max_results: int = 100,
    headless: bool = True,
    use_proxy: bool = False,
    browser_manager: Optional[BrowserManager] = None,
) -> List[Tuple[str, Dict]]:
    urls_with_meta = []
    driver = None
    own_driver = False
    try:
        if browser_manager:
            driver = browser_manager.get_driver()
        else:
            proxy = proxy_pool.get() if use_proxy else None
            ua = random_ua()
            driver = _init_uc_chrome(headless, proxy, ua)
            driver.set_page_load_timeout(30)
            own_driver = True

        driver.get(f"https://www.bing.com/images/search?q={quote_plus(query)}")
        time.sleep(2)
        # Cuộn trang
        last_h = driver.execute_script("return document.body.scrollHeight")
        for _ in range(20):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.0, 1.5))
            try:
                for sel in SEE_MORE_SELECTORS:
                    for btn in driver.find_elements(By.CSS_SELECTOR, sel):
                        if btn.is_displayed() and btn.is_enabled():
                            btn.click()
                            time.sleep(1.2)
                            break
            except:
                pass
            new_h = driver.execute_script("return document.body.scrollHeight")
            if new_h == last_h:
                break
            last_h = new_h

        # Phương án 1: a.iusc JSON
        iusc_elements = driver.find_elements(By.CSS_SELECTOR, "a.iusc")
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
                    title = data.get("t", "") or data.get("desc", "")
                    urls_with_meta.append((murl, {"source": "bing_iusc", "query": query,
                                                  "timestamp": datetime.now().isoformat(),
                                                  "url": murl, "title": title}))
            except:
                continue

        # Phương án 2: click thumbnail + fallback
        if len(urls_with_meta) < max_results:
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
            for thumb in thumbnails[:max_results * 2]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    thumb.click()
                    time.sleep(0.4)
                    for sel in ["img.n1ddgdc", "img.sib_r", ".imgpt img"]:
                        for actual in driver.find_elements(By.CSS_SELECTOR, sel):
                            src = actual.get_attribute("src") or actual.get_attribute("data-src")
                            if src and src.startswith("http") and "th?" not in src and len(src) > 80:
                                urls_with_meta.append((src, {"source": "bing_click", "query": query,
                                                              "timestamp": datetime.now().isoformat(),
                                                              "url": src}))
                                break
                        else:
                            continue
                        break
                except:
                    continue
        log.info(f"Thu được {len(urls_with_meta)} URL")
    finally:
        if own_driver and driver:
            try:
                driver.quit()
            except:
                pass
    return urls_with_meta


# ----------------------------------------------------------------------
# Fetch URLs từ DuckDuckGo (không cần browser)
# ----------------------------------------------------------------------
def fetch_image_urls_duckduckgo(
    query: str,
    max_results: int = 100,
) -> List[Tuple[str, Dict]]:
    """Lấy URL ảnh từ DuckDuckGo Images (không cần Selenium).

    Yêu cầu: ``pip install duckduckgo-search``
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.warning("duckduckgo-search chưa cài – bỏ qua nguồn DDG. "
                    "Cài bằng: pip install duckduckgo-search")
        return []

    urls_with_meta: List[Tuple[str, Dict]] = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(
                keywords=query,
                max_results=max_results,
            ))
        for r in results:
            img_url = r.get("image", "")
            if img_url and img_url.startswith("http"):
                urls_with_meta.append((img_url, {
                    "source": "duckduckgo",
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "url": img_url,
                    "title": r.get("title", ""),
                }))
        log.info(f"[DDG] Thu được {len(urls_with_meta)} URL cho '{query}'")
    except Exception as e:
        log.warning(f"[DDG] Lỗi khi tìm kiếm '{query}': {e}")
    return urls_with_meta


# ----------------------------------------------------------------------
# Fetch URLs từ Google Images (Selenium)
# ----------------------------------------------------------------------
def fetch_image_urls_google(
    query: str,
    max_results: int = 100,
    headless: bool = True,
    use_proxy: bool = False,
    browser_manager: Optional[BrowserManager] = None,
) -> List[Tuple[str, Dict]]:
    """Lấy URL ảnh từ Google Images qua Selenium."""
    urls_with_meta: List[Tuple[str, Dict]] = []
    driver = None
    own_driver = False
    try:
        if browser_manager:
            driver = browser_manager.get_driver()
        else:
            proxy = proxy_pool.get() if use_proxy else None
            ua = random_ua()
            driver = _init_uc_chrome(headless, proxy, ua)
            driver.set_page_load_timeout(30)
            own_driver = True

        search_url = (
            f"https://www.google.com/search?q={quote_plus(query)}"
            f"&tbm=isch&hl=vi&gl=vn"
        )
        driver.get(search_url)
        time.sleep(2)

        # Cuộn trang để tải thêm ảnh
        last_h = driver.execute_script("return document.body.scrollHeight")
        for _ in range(15):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.2, 1.8))
            # Bấm nút "Xem thêm kết quả" nếu có
            try:
                for btn_sel in [
                    "input.mye4qd", "[data-ved] button", "a.frGj1b",
                ]:
                    for btn in driver.find_elements(By.CSS_SELECTOR, btn_sel):
                        if btn.is_displayed() and btn.is_enabled():
                            btn.click()
                            time.sleep(1)
                            break
            except Exception:
                pass
            new_h = driver.execute_script("return document.body.scrollHeight")
            if new_h == last_h:
                break
            last_h = new_h

        # Chiết xuất URL ảnh từ thuộc tính src / data-src của img
        img_elements = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf, img.rg_i")
        for el in img_elements:
            if len(urls_with_meta) >= max_results:
                break
            try:
                src = (
                    el.get_attribute("src")
                    or el.get_attribute("data-src")
                    or ""
                )
                # Bỏ qua thumbnail base64 nhỏ
                if src.startswith("http") and len(src) > 80 and "gstatic" not in src:
                    alt = el.get_attribute("alt") or ""
                    urls_with_meta.append((src, {
                        "source": "google",
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "url": src,
                        "title": alt,
                    }))
            except Exception:
                continue

        # Phương án 2: click thumbnail để lấy URL full-size
        if len(urls_with_meta) < max_results // 2:
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "div.bRMDJf img")
            for thumb in thumbnails[:max_results * 2]:
                if len(urls_with_meta) >= max_results:
                    break
                try:
                    thumb.click()
                    time.sleep(0.5)
                    for sel in ["img.n3VNCb", "img.r48jcc", ".tvh0pb img"]:
                        for actual in driver.find_elements(By.CSS_SELECTOR, sel):
                            src = (
                                actual.get_attribute("src")
                                or actual.get_attribute("data-src") or ""
                            )
                            if src.startswith("http") and len(src) > 80:
                                alt = actual.get_attribute("alt") or ""
                                urls_with_meta.append((src, {
                                    "source": "google_click",
                                    "query": query,
                                    "timestamp": datetime.now().isoformat(),
                                    "url": src,
                                    "title": alt,
                                }))
                                break
                        else:
                            continue
                        break
                except Exception:
                    continue

        log.info(f"[Google] Thu được {len(urls_with_meta)} URL cho '{query}'")
    except Exception as e:
        log.warning(f"[Google] Lỗi: {e}")
    finally:
        if own_driver and driver:
            try:
                driver.quit()
            except Exception:
                pass
    return urls_with_meta


# ----------------------------------------------------------------------
# Fetch đa nguồn: Bing + Google + DuckDuckGo
# ----------------------------------------------------------------------
def _normalize_sources(sources: Optional[List[str]]) -> List[str]:
    """Chuẩn hoá alias nguồn về tên canonical: bing, google, ddg."""
    if sources is None:
        sources = ["bing", "google", "ddg"]

    normalized: List[str] = []
    for source in sources:
        key = source.lower().strip().replace("_", "-").replace(" ", "")
        canonical = SOURCE_ALIASES.get(key, key)
        if canonical in ("bing", "google", "ddg") and canonical not in normalized:
            normalized.append(canonical)

    return normalized or ["bing", "google", "ddg"]


def fetch_image_urls_multi_source(
    query: str,
    max_results: int = 100,
    headless: bool = True,
    use_proxy: bool = False,
    browser_manager: Optional[BrowserManager] = None,
    sources: Optional[List[str]] = None,
) -> List[Tuple[str, Dict]]:
    """Lấy URL ảnh từ nhiều nguồn tìm kiếm song song, tự động dedup.

    Args:
        query:          Từ khoá tìm kiếm.
        max_results:    Số URL tối đa mong muốn (mỗi nguồn sẽ cố lấy số này).
        headless:       Chạy trình duyệt ẩn hay hiện.
        use_proxy:      Dùng proxy cho browser (Bing/Google).
        browser_manager: BrowserManager dùng chung (nếu có).
        sources:        Danh sách nguồn cần dùng.
                        Hợp lệ: ``'bing'``, ``'google'``, ``'ddg'``.
                        Mặc định: tất cả 3 nguồn.

    Returns:
        Danh sách ``(url, metadata_dict)`` không trùng URL.
    """
    sources = _normalize_sources(sources)

    all_results: List[Tuple[str, Dict]] = []
    seen_urls: Set[str] = set()

    def _bing():
        if "bing" not in sources:
            return []
        try:
            return fetch_image_urls_with_selenium(
                query, max_results=max_results,
                headless=headless, use_proxy=use_proxy,
                browser_manager=browser_manager,
            )
        except Exception as e:
            log.warning(f"[Bing] Lỗi: {e}")
            return []

    def _google():
        if "google" not in sources:
            return []
        try:
            return fetch_image_urls_google(
                query, max_results=max_results,
                headless=headless, use_proxy=use_proxy,
                browser_manager=browser_manager,
            )
        except Exception as e:
            log.warning(f"[Google] Lỗi: {e}")
            return []

    def _ddg():
        if "ddg" not in sources:
            return []
        return fetch_image_urls_duckduckgo(query, max_results=max_results)

    # Đơn giản hoá: chạy tuần tự để tránh xung đột browser
    for fn in [_ddg, _bing, _google]:       # DDG trước vì không cần browser
        for url, meta in fn():
            if url not in seen_urls:
                seen_urls.add(url)
                all_results.append((url, meta))

    log.info(
        f"[Multi-source] Tổng {len(all_results)} URL duy nhất "
        f"từ nguồn: {', '.join(sources)}"
    )
    return all_results


# ----------------------------------------------------------------------
# Download image với retry, validate, edge detection
# ----------------------------------------------------------------------
def _validate_image_quality(img: Image.Image, class_name: str = "") -> Tuple[bool, float]:
    try:
        w, h = img.size
        aspect = w / h if h > 0 else 1.0
        if aspect > 5.0 or aspect < 0.2:
            return False, 0.1
        confidence = 0.5
        # Aspect ratio theo nhóm
        if class_name.startswith("Rice"):
            if 0.7 <= aspect <= 4.5:
                confidence += 0.2
            else:
                confidence -= 0.1
        elif class_name.startswith("Coffee"):
            if 0.8 <= aspect <= 2.8:
                confidence += 0.2
        elif class_name.startswith("Tomato"):
            if 0.5 <= aspect <= 3.5:
                confidence += 0.2
        elif class_name.startswith("Citrus"):
            if 0.8 <= aspect <= 3.0:
                confidence += 0.2
        else:
            if 0.5 <= aspect <= 3.0:
                confidence += 0.2

        # Plant ratio (màu xanh + vàng/nâu)
        img_small = img.resize((64, 64))
        from PIL import ImageStat
        stat = ImageStat.Stat(img_small)
        stdev = sum(stat.stddev) / len(stat.stddev) if stat.stddev else 0
        if stdev < 8:
            confidence -= 0.3
        elif stdev > 45:
            confidence += 0.1

        img_hsv = img_small.convert("HSV")
        hsv_data = list(img_hsv.getdata())
        green = sum(1 for p in hsv_data if 55 <= p[0] <= 160)
        yellow = sum(1 for p in hsv_data if 10 <= p[0] < 55)
        plant = (green + yellow) / 4096.0
        if plant > 0.15:
            confidence += 0.2
        elif plant < 0.06:
            confidence -= 0.55

        # Edge detection (diagram, text)
        from PIL import ImageFilter
        img_edge = img.resize((128, 128)).convert("L").filter(ImageFilter.FIND_EDGES)
        pixels = list(img_edge.getdata())
        w, h = 128, 128
        block = 16
        total = (w//block)*(h//block)
        dense = 0
        for by in range(0, h, block):
            for bx in range(0, w, block):
                cnt = 0
                for y in range(by, min(by+block, h)):
                    row = y * w
                    for x in range(bx, min(bx+block, w)):
                        if pixels[row + x] > 25:
                            cnt += 1
                if cnt / (block*block) > 0.2:
                    dense += 1
        dense_ratio = dense / total
        if plant > 0.5:
            pass
        elif dense_ratio > 0.4 and plant < 0.3:
            return False, 0.05
        if dense_ratio > 0.1:
            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))
        return confidence >= 0.35, confidence
    except Exception:
        return True, 0.5

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5),
       retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS))
def _download_image_core(url: str, output_path: Path, timeout_s: int = 20,
                         use_proxy: bool = False, class_name: str = "") -> Tuple[bool, Optional[Dict]]:
    proxy = proxy_pool.get() if use_proxy else None
    ua = random_ua()
    session = _get_session()
    try:
        if session is not None:
            proxies = {"http": proxy, "https": proxy} if proxy else None
            session.headers["User-Agent"] = ua
            resp = session.get(url, proxies=proxies, timeout=(10, timeout_s))
            resp.raise_for_status()
            raw = resp.content
        else:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            if proxy:
                handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
                with urllib.request.build_opener(handler).open(req, timeout=timeout_s) as r:
                    raw = r.read()
            else:
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    raw = r.read()
    except RETRYABLE_EXCEPTIONS:
        if proxy:
            proxy_pool.remove(proxy)
        raise
    if len(raw) < 1024:
        return False, None
    try:
        from PIL import ImageOps
        img = Image.open(BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except:
        return False, None
    w, h = img.size
    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
        return False, None
    ok, conf = _validate_image_quality(img, class_name)
    if not ok:
        return False, None
    tmp_path = output_path.parent / f"_tmp_{output_path.name}"
    try:
        with open(tmp_path, "wb") as f:
            img.save(f, "JPEG", quality=95)
        tmp_path.replace(output_path)
    except:
        try:
            tmp_path.unlink(missing_ok=True)
        except:
            pass
        raise
    img_hash = _compute_image_hash(output_path)
    return True, {
        "filename": output_path.name, "width": w, "height": h,
        "size_bytes": output_path.stat().st_size, "url": url,
        "proxy_used": proxy or "Direct", "user_agent": ua,
        "download_time": datetime.now().isoformat(), "confidence": conf,
        "image_hash": img_hash
    }

def download_image(url: str, output_path: Path, timeout_s: int = 20,
                   use_proxy: bool = False, class_name: str = "") -> Tuple[bool, Optional[Dict]]:
    try:
        return _download_image_core(url, output_path, timeout_s, use_proxy, class_name)
    except Exception as e:
        log.debug(f"Download error {url}: {e}")
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return False, None

# ----------------------------------------------------------------------
# Crawl một lớp
# ----------------------------------------------------------------------
def crawl_class_stealth(
    class_name: str,
    output_root: str = "data/raw",
    max_images: int = 1000,
    headless: bool = True,
    use_proxy_for_browser: bool = False,
    use_proxy_for_download: bool = False,
    download_workers: int = 16,
    sources: Optional[List[str]] = None,
) -> Dict:
    """Crawl ảnh cho một lớp từ nhiều nguồn tìm kiếm.

    Args:
        sources: Danh sách nguồn dùng để tìm kiếm.
                 Hợp lệ: ``'bing'``, ``'google'``, ``'ddg'``.
                 Mặc định: ``['bing', 'google', 'ddg']``.
    """
    class_dir = Path(output_root) / class_name
    ensure_dir(str(class_dir))
    _cleanup_temp_files(class_dir)

    next_idx, downloaded, seen_urls = _load_existing_state(class_dir)
    meta_path = class_dir / "metadata.jsonl"
    attempted = 0

    metadata_lock = threading.Lock()
    file_counter = next_idx
    file_counter_lock = threading.Lock()
    image_hashes = _load_image_hashes(class_dir)

    def append_meta(entry: Dict):
        with metadata_lock:
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    browser_mgr = BrowserManager(headless=headless, use_proxy=use_proxy_for_browser)
    queries = build_search_queries(class_name)

    try:
        for q_idx, query in enumerate(queries):
            if downloaded >= max_images:
                break
            log.info(f"[{class_name}] Query {q_idx+1}/{len(queries)}: '{query}'")
            urls_with_meta = fetch_image_urls_multi_source(
                query, max_results=max_images * 2,
                headless=headless, use_proxy=use_proxy_for_browser,
                browser_manager=browser_mgr,
                sources=sources,
            )
            if not urls_with_meta:
                continue

            unique = []
            for url, meta in urls_with_meta:
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique.append((url, meta))
            # Lọc từ khóa loại trừ
            unique = [(u, m) for u, m in unique if not _should_exclude_url(u, query, class_name)]
            unique = unique[:max_images - downloaded]
            if not unique:
                continue

            log.info(f"Bắt đầu tải {len(unique)} ảnh với {download_workers} workers")
            futures = {}
            with ThreadPoolExecutor(max_workers=download_workers) as pool:
                for url, meta in unique:
                    tmp_path = class_dir / f"_tmp_download_{random.randint(0,1e9):010d}.jpg"
                    futures[pool.submit(download_image, url, tmp_path, 20,
                                        use_proxy_for_download, class_name)] = (meta, tmp_path)
                for fut in tqdm(as_completed(futures), total=len(futures), desc=class_name):
                    attempted += 1
                    meta, tmp_path = futures[fut]
                    try:
                        success, img_meta = fut.result()
                        if success and img_meta:
                            img_hash = img_meta.get("image_hash")
                            if img_hash and img_hash in image_hashes:
                                tmp_path.unlink(missing_ok=True)
                                continue
                            if img_hash:
                                image_hashes.add(img_hash)
                            with file_counter_lock:
                                final_idx = file_counter
                                file_counter += 1
                            final_path = class_dir / f"img_{final_idx:06d}.jpg"
                            tmp_path.rename(final_path)
                            img_meta["filename"] = final_path.name
                            downloaded += 1
                            append_meta({**meta, **img_meta})
                        else:
                            tmp_path.unlink(missing_ok=True)
                    except Exception as e:
                        log.warning(f"Lỗi task: {e}")
                        tmp_path.unlink(missing_ok=True)
    finally:
        browser_mgr.quit()

    log.info(f"[{class_name}] Hoàn thành: {downloaded}/{attempted} ảnh")
    return {"class": class_name, "downloaded": downloaded, "attempted": attempted,
            "output_dir": str(class_dir)}

# ----------------------------------------------------------------------
# Crawl tất cả
# ----------------------------------------------------------------------
def crawl_all_stealth(
    output_root: str = "data/raw",
    max_images_per_class: int = 1000,
    classes: Optional[List[str]] = None,
    use_proxy_for_browser: bool = False,
    use_proxy_for_download: bool = False,
    download_workers: int = 16,
    sources: Optional[List[str]] = None,
) -> Dict:
    root = Path(output_root)
    for d in [root, root.parent / "processed", root.parent / "augmented"]:
        ensure_dir(str(d))
    target = classes if classes else list(DATASET_CLASSES.keys())
    target = [c for c in target if c in DATASET_CLASSES]
    return {
        c: crawl_class_stealth(
            c, output_root, max_images_per_class, True,
            use_proxy_for_browser, use_proxy_for_download,
            download_workers, sources=sources,
        )
        for c in target
    }

# ----------------------------------------------------------------------
# Cảnh báo và CLI
# ----------------------------------------------------------------------
_ETHICAL_WARNING = """
╔══════════════════════════════════════════════════════════════════════╗
║   ⚠  CẢNH BÁO: Chỉ dùng cho nghiên cứu học thuật, phi thương mại  ║
║   Tuân thủ ToS của Bing, không tái phân phối ảnh vi phạm bản quyền ║
╚══════════════════════════════════════════════════════════════════════╝
"""

def _show_ethical_warning():
    print(_ETHICAL_WARNING)
    ans = input("Bạn đồng ý? [yes/no]: ").strip().lower()
    if ans not in ("yes", "y"):
        print("Đã hủy.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    _show_ethical_warning()
    parser = argparse.ArgumentParser(
        description="Thu thập ảnh bệnh cây trồng từ nhiều nguồn (Bing, Google, DuckDuckGo)."
    )
    parser.add_argument("--class", dest="class_name",
                        help="Tên lớp cần crawl (bỏ qua để crawl tất cả)")
    parser.add_argument("--max", type=int, default=300,
                        help="Số ảnh tối đa mỗi lớp (mặc định: 300)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Số luồng tải song song (mặc định: 16)")
    parser.add_argument("--use-proxy-browser", action="store_true",
                        help="Dùng proxy cho trình duyệt")
    parser.add_argument("--use-proxy-download", action="store_true",
                        help="Dùng proxy khi tải ảnh")
    parser.add_argument("--output",
                        default=str(Path(__file__).parent.parent / "data/raw"),
                        help="Thư mục lưu ảnh")
    parser.add_argument(
        "--sources", default="bing,google,ddg",
        help="Nguồn tìm kiếm, phân cách bằng dấu phẩy: bing,google,ddg "
             "(alias hỗ trợ: bing-images, google-images, duckduckgo)"
             " (mặc định: tất cả 3 nguồn)",
    )
    args = parser.parse_args()
    sources_list = _normalize_sources([s.strip() for s in args.sources.split(",") if s.strip()])
    log.info(f"Nguồn tìm kiếm đã chọn: {sources_list}")
    results = crawl_all_stealth(
        output_root=args.output,
        max_images_per_class=args.max,
        classes=[args.class_name] if args.class_name else None,
        use_proxy_for_browser=args.use_proxy_browser,
        use_proxy_for_download=args.use_proxy_download,
        download_workers=args.workers,
        sources=sources_list,
    )
    print("\n── Kết quả tổng hợp ──")
    for cls, res in results.items():
        print(f"  {cls}: {res['downloaded']} ảnh đã tải / {res['attempted']} thử tải")