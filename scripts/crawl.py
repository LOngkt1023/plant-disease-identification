#!/usr/bin/env python3
"""Script entrypoint for crawling plant disease images using the new dataset_v2 structure."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.crawler_stealth import crawl_all_stealth
from src.config.disease_classes import CLASS_NAMES
from src.config import DATA_RAW_DIR

def main():
    parser = argparse.ArgumentParser(description="Crawl plant disease images for dataset_v2.")
    parser.add_argument("--max-images-per-class", type=int, default=100, 
                        help="Target number of images per class (default: 100 for testing)")
    parser.add_argument("--classes", nargs="+", help="Specific classes to crawl (default: all 16 classes)")
    parser.add_argument("--workers", type=int, default=16, help="Number of download workers")
    parser.add_argument("--use-proxy-browser", action="store_true", help="Use proxy for browser")
    parser.add_argument("--use-proxy-download", action="store_true", help="Use proxy for downloads")
    parser.add_argument("--output-dir", type=str, default=str(DATA_RAW_DIR), help="Output directory for raw images")
    
    args = parser.parse_args()
    
    target_classes = args.classes if args.classes else CLASS_NAMES
    
    print(f"Starting crawl for {len(target_classes)} classes...")
    print(f"Target: {args.max_images_per_class} images per class")
    print(f"Output directory: {args.output_dir}")
    
    results = crawl_all_stealth(
        output_root=args.output_dir,
        max_images_per_class=args.max_images_per_class,
        classes=target_classes,
        use_proxy_for_browser=args.use_proxy_browser,
        use_proxy_for_download=args.use_proxy_download,
        download_workers=args.workers
    )
    
    print("\n" + "="*50)
    print("CRAWL SUMMARY")
    print("="*50)
    total_downloaded = 0
    for cls, res in results.items():
        downloaded = res.get('downloaded', 0)
        total_downloaded += downloaded
        print(f"{cls:<30}: {downloaded:>5} images")
    print("-" * 50)
    print(f"{'TOTAL':<30}: {total_downloaded:>5} images")
    print("="*50)

if __name__ == "__main__":
    main()