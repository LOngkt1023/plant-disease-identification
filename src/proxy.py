import json
import logging
import random
import re
import threading
from typing import List, Optional
import urllib.request

log = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class ProxyPool:
    _HEALTH_URL = "https://httpbin.org/ip"
    _HEALTH_TIMEOUT = 5
    _MIN_HEALTHY = 3
    _MAX_CANDIDATES = 40

    def __init__(self):
        self.active_pool: List[str] = []
        self._lock = threading.Lock()

    def _scrape_candidates(self) -> List[str]:
        candidates = []
        try:
            req = urllib.request.Request(
                "https://free-proxy-list.net/",
                headers={"User-Agent": USER_AGENTS[0]},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8")
            for ip, port in re.findall(r"<td>(\d{1,3}(?:\.\d{1,3}){3})</td><td>(\d+)</td>", html)[:self._MAX_CANDIDATES]:
                candidates.append(f"http://{ip}:{port}")
        except Exception as e:
            log.warning(f"Cào proxy thất bại: {e}")
        return candidates

    def _is_alive(self, proxy: str) -> bool:
        try:
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            opener = urllib.request.build_opener(handler)
            req = urllib.request.Request(self._HEALTH_URL, headers={"User-Agent": USER_AGENTS[0]})
            with opener.open(req, timeout=self._HEALTH_TIMEOUT) as resp:
                if resp.status != 200:
                    return False
                body = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(body)
                return "origin" in data
        except Exception:
            return False

    def _refresh(self):
        log.info("Proxy pool rỗng – đang cập nhật...")
        candidates = self._scrape_candidates()
        if not candidates:
            log.warning("Không tìm thấy proxy nào")
            return
        healthy = []
        for p in candidates:
            if self._is_alive(p):
                healthy.append(p)
                log.info(f"  [OK] {p}")
                if len(healthy) >= self._MIN_HEALTHY:
                    break
        self.active_pool = healthy
        log.info(f"Pool hoạt động: {len(self.active_pool)} proxy")

    def get(self) -> Optional[str]:
        with self._lock:
            if not self.active_pool:
                self._refresh()
            return random.choice(self.active_pool) if self.active_pool else None

    def remove(self, proxy: str):
        with self._lock:
            try:
                self.active_pool.remove(proxy)
                log.info(f"Đã loại proxy {proxy}")
            except ValueError:
                pass


proxy_pool = ProxyPool()
