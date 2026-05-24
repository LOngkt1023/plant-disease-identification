"""Central configuration for crawler project."""
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_SRC_DIR     = Path(__file__).parent
PROJECT_ROOT = _SRC_DIR.parent

# Dataset directories (new structure under dataset_v2/)
DATASET_DIR        = PROJECT_ROOT / "dataset_v2"
DATA_RAW_DIR       = DATASET_DIR / "raw"
DATA_CLEAN_DIR     = DATASET_DIR / "clean"
DATA_REJECTED_DIR  = DATASET_DIR / "rejected"
DATA_REVIEW_DIR    = DATASET_DIR / "review"
DATA_PROCESSED_DIR = DATASET_DIR / "processed"
METADATA_DIR       = DATASET_DIR / "metadata"
SPLITS_DIR         = DATASET_DIR / "splits"

# Metadata file paths
RAW_METADATA_CSV   = METADATA_DIR / "raw_metadata.csv"
CLEAN_METADATA_CSV = METADATA_DIR / "clean_metadata.csv"
CRAWL_LOG_CSV      = METADATA_DIR / "crawl_log.csv"

# ---------------------------------------------------------------------------
# Thresholds — 3-level filtering strategy
# ---------------------------------------------------------------------------
#   Level 1 (KEEP  → clean/)   : clip_score > CLIP_KEEP  AND disease_score > DISEASE_KEEP
#   Level 2 (REVIEW → review/) : clip_score ∈ [CLIP_REVIEW, CLIP_KEEP]
#                                 OR  disease_score ∈ [DISEASE_REVIEW, DISEASE_KEEP]
#   Level 3 (REJECT → rejected/): clip_score < CLIP_REVIEW  OR  file invalid / duplicate
CLIP_KEEP_THRESHOLD     : float = 0.6    # must be above to auto-accept
CLIP_REVIEW_THRESHOLD   : float = 0.35   # below this → reject as irrelevant
DISEASE_KEEP_THRESHOLD  : float = 0.5    # high-confidence disease label
DISEASE_REVIEW_THRESHOLD: float = 0.3    # low-medium confidence → send to review

# ---------------------------------------------------------------------------
# Crawler settings
# ---------------------------------------------------------------------------
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Selectors to click "See more images" on Bing-like pages
SEE_MORE_SELECTORS: List[str] = [
    "input[type='button'][value='See more images']",
    "input[value='See more images']", "#seemorekey", ".b_seemore", ".mop",
]

# Minimum image dimensions accepted by downloader
MIN_IMAGE_WIDTH : int = 200
MIN_IMAGE_HEIGHT: int = 200