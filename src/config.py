"""Central configuration for crawler project."""
from typing import List

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Selectors to click "See more images" on Bing-like pages
SEE_MORE_SELECTORS: List[str] = [
    "input[type='button'][value='See more images']",
    "input[value='See more images']", "#seemorekey", ".b_seemore", ".mop",
]

# Minimum image dimensions accepted by downloader
MIN_IMAGE_WIDTH: int = 200
MIN_IMAGE_HEIGHT: int = 200
