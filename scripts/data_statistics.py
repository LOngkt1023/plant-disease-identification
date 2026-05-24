#!/usr/bin/env python3
"""
Script to generate data statistics and visualizations for dataset_v2.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.disease_classes import CLASS_NAMES, PLANT_DISEASE_CLASSES
from src.config import DATA_DIR, DATA_RAW_DIR, DATA_CLEAN_DIR, DATA_METADATA_DIR

def generate_statistics(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating statistics in {output_path}...")
    
    # 1. Load Metadata if exists
    raw_meta_path = DATA_METADATA_DIR / "raw_metadata.parquet"
    clean_meta_path = DATA_METADATA_DIR / "clean_metadata.parquet"
    
    raw_df = pd.DataFrame()
    if raw_meta_path.exists():
        raw_df = pd.read_parquet(raw_meta_path)
        print(f"Loaded raw metadata: {len(raw_df)} rows")
    
    clean_df = pd.DataFrame()
    if clean_meta_path.exists():
        clean_df = pd.read_parquet(clean_meta_path)
        print(f"Loaded clean metadata: {len(clean_df)} rows")

    # 2. Physical counts (fallback if metadata missing)
    stats_data = []
    for cls in CLASS_NAMES:
        raw_count = 0
        if (DATA_RAW_DIR / cls).exists():
            raw_count = len(list((DATA_RAW_DIR / cls).glob("*.*")))
            
        clean_count = 0
        if (DATA_CLEAN_DIR / cls).exists():
            clean_count = len(list((DATA_CLEAN_DIR / cls).glob("*.*")))
            
        stats_data.append({
            "class_name": cls,
            "plant": PLANT_DISEASE_CLASSES[cls]["plant"],
            "disease": PLANT_DISEASE_CLASSES[cls]["disease"],
            "raw_count": raw_count,
            "clean_count": clean_count
        })
    
    summary_df = pd.DataFrame(stats_data)
    summary_df.to_csv(output_path / "data_summary.csv", index=False)
    
    # 3. Visualizations
    sns.set_theme(style="whitegrid")
    
    # Class Distribution
    plt.figure(figsize=(12, 8))
    plot_df = summary_df.melt(id_vars="class_name", value_vars=["raw_count", "clean_count"], 
                              var_name="Status", value_name="Count")
    sns.barplot(data=plot_df, x="Count", y="class_name", hue="Status")
    plt.title("Image Distribution by Class")
    plt.tight_layout()
    plt.savefig(output_path / "class_distribution.png")
    
    # Plant Distribution
    plt.figure(figsize=(10, 6))
    plant_stats = summary_df.groupby("plant")[["raw_count", "clean_count"]].sum().reset_index()
    plant_plot = plant_stats.melt(id_vars="plant", value_vars=["raw_count", "clean_count"])
    sns.barplot(data=plant_plot, x="value", y="plant", hue="variable")
    plt.title("Image Distribution by Plant")
    plt.tight_layout()
    plt.savefig(output_path / "plant_distribution.png")

    # Disease Distribution
    plt.figure(figsize=(10, 6))
    disease_stats = summary_df.groupby("disease")[["raw_count", "clean_count"]].sum().reset_index()
    disease_plot = disease_stats.melt(id_vars="disease", value_vars=["raw_count", "clean_count"])
    sns.barplot(data=disease_plot, x="value", y="disease", hue="variable")
    plt.title("Image Distribution by Disease Type")
    plt.tight_layout()
    plt.savefig(output_path / "disease_distribution.png")

    # Status Distribution (Clean vs Raw)
    if not summary_df.empty:
        plt.figure(figsize=(8, 8))
        total_raw = summary_df["raw_count"].sum()
        total_clean = summary_df["clean_count"].sum()
        if total_raw > 0:
            plt.pie([total_clean, total_raw - total_clean], labels=["Clean", "Filtered/Raw"], 
                    autopct='%1.1f%%', colors=["#2ecc71", "#e74c3c"])
            plt.title("Overall Data Quality (Clean vs Filtered)")
            plt.savefig(output_path / "status_distribution.png")

    # Metadata-based plots
    if not raw_df.empty:
        # Top Source Domains
        if "source_domain" in raw_df.columns:
            plt.figure(figsize=(10, 6))
            top_domains = raw_df["source_domain"].value_counts().head(15)
            sns.barplot(x=top_domains.values, y=top_domains.index)
            plt.title("Top 15 Source Domains")
            plt.tight_layout()
            plt.savefig(output_path / "source_domain_distribution.png")
            
        # Keyword Distribution
        if "search_keyword" in raw_df.columns:
            plt.figure(figsize=(10, 8))
            top_keywords = raw_df["search_keyword"].value_counts().head(20)
            sns.barplot(x=top_keywords.values, y=top_keywords.index)
            plt.title("Top 20 Search Keywords")
            plt.tight_layout()
            plt.savefig(output_path / "keyword_distribution.png")

        # Image Size Distribution
        if "width" in raw_df.columns and "height" in raw_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=raw_df.sample(min(2000, len(raw_df))), x="width", y="height", alpha=0.5)
            plt.title("Image Resolution Distribution (Sampled)")
            plt.savefig(output_path / "image_size_distribution.png")

    print(f"Statistics generation complete. Files saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data statistics for dataset_v2")
    parser.add_argument("--output-dir", type=str, default="outputs/data_statistics", 
                        help="Directory to save statistics and plots")
    args = parser.parse_args()
    
    generate_statistics(args.output_dir)