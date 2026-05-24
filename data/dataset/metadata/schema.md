# Dataset Metadata Schema

The following fields should be included in the metadata files (`raw_metadata.parquet`, `clean_metadata.parquet`):

| Field | Description |
|-------|-------------|
| `image_id` | Unique identifier for the image |
| `class_name` | Target class name (e.g., Rice_Healthy) |
| `plant_type` | Type of plant (e.g., Rice) |
| `disease_name` | Name of the disease (e.g., Blast) |
| `source_url` | Original URL of the image |
| `source_domain` | Domain of the source URL |
| `search_keyword` | Keyword used to find the image |
| `crawl_time` | Timestamp when the image was crawled |
| `raw_path` | Path to the raw image file |
| `clean_path` | Path to the cleaned image file |
| `width` | Image width in pixels |
| `height` | Image height in pixels |
| `file_size` | File size in bytes |
| `clip_score` | CLIP similarity score (if applicable) |
| `disease_score` | Model confidence score for disease detection |
| `hash_md5` | MD5 hash of the image file for deduplication |
| `status` | Current status (raw, clean, rejected, review) |
| `reject_reason` | Reason for rejection (if status is rejected) |

## Files in this directory:
- `raw_metadata.parquet`: Metadata for all crawled images.
- `clean_metadata.parquet`: Metadata for images that passed cleaning/filtering.
- `crawl_log.csv`: Log of crawling sessions.