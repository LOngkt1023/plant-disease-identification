"""Data cleaning and preprocessing pipeline for Plant Disease Identification."""

import argparse
import os
import sys
import shutil
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add root folder to sys.path to run correctly
sys.path.append(str(Path(__file__).parent.parent))

from src.config.disease_classes import CLASS_NAMES, PLANT_DISEASE_CLASSES
from src.utils import compute_md5
from src.preprocessing import resize_with_padding

def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess crawled images.")
    parser.add_argument("--raw-dir", type=str, default="dataset_v2/raw", help="Path to raw image directories.")
    parser.add_argument("--clean-dir", type=str, default="dataset_v2/clean", help="Path to clean output directory.")
    parser.add_argument("--review-dir", type=str, default="dataset_v2/review", help="Path to review output directory.")
    parser.add_argument("--rejected-dir", type=str, default="dataset_v2/rejected", help="Path to rejected output directory.")
    parser.add_argument("--processed-dir", type=str, default="dataset_v2/processed", help="Path to preprocessed 224x224 output directory.")
    parser.add_argument("--metadata-dir", type=str, default="dataset_v2/metadata", help="Path to store metadata files.")
    parser.add_argument("--clip-filter", action="store_true", help="Use CLIP model for filtering.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    review_dir = Path(args.review_dir)
    rejected_dir = Path(args.rejected_dir)
    processed_dir = Path(args.processed_dir)
    metadata_dir = Path(args.metadata_dir)

    for d in [clean_dir, review_dir, rejected_dir, processed_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STARTING DATA CLEANING & PREPROCESSING PIPELINE")
    print(f"Raw source: {raw_dir}")
    print(f"Clean output: {clean_dir}")
    print(f"Review output: {review_dir}")
    print(f"Rejected output: {rejected_dir}")
    print(f"Processed (224x224) output: {processed_dir}")
    print("==================================================")

    # Initialize CLIP filter if requested
    clip_model = None
    clip_processor = None
    if args.clip_filter:
        print("Loading CLIP model for semantic relevance filtering...")
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()
            print("CLIP model loaded successfully.")
        except Exception as e:
            print(f"Failed to load CLIP. Running without CLIP filter. Error: {e}")
            clip_model = None

    seen_hashes = {}  # md5 -> first class_name it was found in
    raw_records = []
    clean_records = []

    # Read existing raw metadata if exists to retain search_keyword, source_url etc.
    raw_meta_file = metadata_dir / "raw_metadata.parquet"
    existing_raw_df = None
    if raw_meta_file.exists():
        try:
            existing_raw_df = pd.read_parquet(raw_meta_file)
            print(f"Loaded existing raw metadata: {len(existing_raw_df)} records.")
        except Exception as e:
            print(f"Could not load existing raw metadata parquet: {e}")

    # Step 1: Scan all raw class folders
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_raw_dir = raw_dir / class_name
        if not class_raw_dir.exists():
            print(f"Class folder not found: {class_name}, skipping.")
            continue

        # Get class metadata config
        class_info = PLANT_DISEASE_CLASSES[class_name]
        plant = class_info["plant"]
        disease = class_info["disease"]

        # Gather images
        img_files = sorted(
            list(class_raw_dir.glob("*.jpg")) +
            list(class_raw_dir.glob("*.jpeg")) +
            list(class_raw_dir.glob("*.png"))
        )

        print(f"\nProcessing {class_name}: {len(img_files)} images found.")
        
        # Load local metadata for this folder if exists
        local_meta_map = {}
        local_meta_path = class_raw_dir / "metadata.jsonl"
        if local_meta_path.exists():
            try:
                with open(local_meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            if "filename" in obj:
                                local_meta_map[obj["filename"]] = obj
            except Exception as e:
                print(f"Error loading local metadata.jsonl: {e}")

        for img_path in tqdm(img_files, desc=f"{class_name}"):
            filename = img_path.name
            file_size = img_path.stat().st_size
            
            # Lookup original metadata properties if available
            orig_meta = local_meta_map.get(filename, {})
            if not orig_meta and existing_raw_df is not None:
                # Fallback to parquet lookup
                match = existing_raw_df[existing_raw_df["image_id"] == filename.split(".")[0]]
                if not match.empty:
                    orig_meta = match.iloc[0].to_dict()

            source_url = orig_meta.get("url", orig_meta.get("source_url", "unknown"))
            source_domain = orig_meta.get("source_domain", "unknown")
            search_keyword = orig_meta.get("query", orig_meta.get("search_keyword", "unknown"))
            crawl_time = orig_meta.get("crawl_time", "unknown")
            
            image_id = img_path.stem
            
            # Record raw metadata entry
            raw_rec = {
                "image_id": image_id,
                "class_name": class_name,
                "class_idx": class_idx,
                "plant": plant,
                "disease": disease,
                "source_url": source_url,
                "source_domain": source_domain,
                "search_keyword": search_keyword,
                "crawl_time": crawl_time,
                "raw_path": str(img_path.resolve()),
                "width": 0,
                "height": 0,
                "file_size": file_size,
                "status": "raw",
                "error_message": ""
            }

            # --- PRE-VERIFY & MD5 HASHING ---
            # 1. Check if corrupted / empty
            if file_size < 1024:
                raw_rec["status"] = "rejected"
                raw_rec["error_message"] = "File size too small (<1KB)"
                raw_records.append(raw_rec)
                shutil.copy2(img_path, rejected_dir / class_name / filename if (rejected_dir / class_name).exists() else (rejected_dir / class_name).mkdir(parents=True, exist_ok=True) or rejected_dir / class_name / filename)
                continue
            
            try:
                with Image.open(img_path) as pil_img:
                    pil_img.verify()
                # Open again to get size
                with Image.open(img_path) as pil_img:
                    width, height = pil_img.size
                    pil_img_rgb = pil_img.convert("RGB") # Just make sure conversion is fine
            except Exception as e:
                raw_rec["status"] = "rejected"
                raw_rec["error_message"] = f"Corrupted file: {e}"
                raw_records.append(raw_rec)
                (rejected_dir / class_name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, rejected_dir / class_name / filename)
                continue

            raw_rec["width"] = width
            raw_rec["height"] = height
            
            # Calculate MD5 hash
            md5_hash = compute_md5(img_path)
            
            # Duplicate detection
            if md5_hash in seen_hashes:
                raw_rec["status"] = "rejected"
                raw_rec["error_message"] = f"Duplicate hash of {seen_hashes[md5_hash]}"
                raw_records.append(raw_rec)
                (rejected_dir / class_name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, rejected_dir / class_name / filename)
                continue
            
            seen_hashes[md5_hash] = f"{class_name}/{filename}"

            # --- SEMANTIC FILTERING (CLIP) ---
            clip_score = 1.0
            status = "clean"
            reject_reason = ""
            
            if clip_model is not None:
                try:
                    import torch
                    with Image.open(img_path) as pil_img:
                        pil_img_rgb = pil_img.convert("RGB")
                        # Perform CLIP matching against typical leaf descriptors vs background
                        prompts = [
                            f"a clear close up photo of a {plant} leaf showing {disease} symptoms",
                            "a leaf of a crop plant showing symptoms of disease",
                            "a healthy green plant leaf",
                            "an abstract pattern or non-plant object or unrelated document text"
                        ]
                        inputs = clip_processor(text=prompts, images=pil_img_rgb, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1)
                        # We sum the probability of the first 3 plant/leaf terms
                        clip_score = float(probs[0][0].item() + probs[0][1].item() + probs[0][2].item())
                except Exception as e:
                    print(f"Error computing CLIP score: {e}")
                    clip_score = 0.5  # default/fallback score on error
            
            # Thresholding rules
            if clip_score >= 0.60:
                status = "clean"
                dest_dir = clean_dir / class_name
            elif clip_score >= 0.35:
                status = "review"
                reject_reason = "low_clip_score_suspected"
                dest_dir = review_dir / class_name
            else:
                status = "rejected"
                reject_reason = "not_relevant_leaf"
                dest_dir = rejected_dir / class_name

            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / filename)
            
            raw_rec["status"] = status
            raw_rec["error_message"] = reject_reason
            raw_records.append(raw_rec)

            if status == "clean":
                # --- PREPROCESSING (224x224 resize with padding) ---
                proc_class_dir = processed_dir / class_name
                proc_class_dir.mkdir(parents=True, exist_ok=True)
                proc_path = proc_class_dir / filename
                
                try:
                    with Image.open(img_path) as pil_img:
                        pil_img_rgb = pil_img.convert("RGB")
                        proc_img = resize_with_padding(pil_img_rgb, (224, 224))
                        proc_img.save(proc_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"Error resizing image {filename}: {e}")
                    continue
                
                # Append clean metadata entry
                clean_rec = {
                    "image_id": image_id,
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "plant": plant,
                    "disease": disease,
                    "source_url": source_url,
                    "source_domain": source_domain,
                    "search_keyword": search_keyword,
                    "crawl_time": crawl_time,
                    "raw_path": str(img_path.resolve()),
                    "clean_path": str((clean_dir / class_name / filename).resolve()),
                    "processed_path": str(proc_path.resolve()),
                    "width": width,
                    "height": height,
                    "file_size": file_size,
                    "md5_hash": md5_hash,
                    "clip_score": round(clip_score, 4),
                    "disease_score": 1.0,  # default / placeholder
                    "status": "clean",
                    "reject_reason": ""
                }
                clean_records.append(clean_rec)

    # Save to parquet/csv files
    if raw_records:
        raw_df = pd.DataFrame(raw_records)
        raw_df.to_parquet(metadata_dir / "raw_metadata.parquet", index=False)
        print(f"Saved raw metadata parquet: {len(raw_df)} records.")
    
    if clean_records:
        clean_df = pd.DataFrame(clean_records)
        clean_df.to_parquet(metadata_dir / "clean_metadata.parquet", index=False)
        clean_df.to_csv(metadata_dir / "clean_metadata.csv", index=False, encoding="utf-8")
        print(f"Saved clean metadata parquet and CSV: {len(clean_df)} records.")
        
        # Output summary counts
        print("\nSummary of clean images per class:")
        summary = clean_df["class_name"].value_counts().to_frame("clean_images")
        print(summary)
        summary.to_csv(metadata_dir / "crawl_summary.csv", encoding="utf-8")
        print(f"Summary saved to {metadata_dir / 'crawl_summary.csv'}")

    print("\nPreprocess and data cleaning complete!")

if __name__ == "__main__":
    main()