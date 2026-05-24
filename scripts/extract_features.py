#!/usr/bin/env python3
"""
Script to extract deep features from ResNet50 and MobileNetV2 pretrained backbones.
Also creates 2D PCA and t-SNE visualizations of the features.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.disease_classes import CLASS_NAMES, CLASS_TO_IDX, PLANT_DISEASE_CLASSES
from src.models import get_resnet50, get_mobilenet_v2

class FeatureExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        if self.split_dir.exists():
            for class_name in CLASS_NAMES:
                class_path = self.split_dir / class_name
                if class_path.exists():
                    for img_path in class_path.glob("*.*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            self.image_paths.append(img_path)
                            self.labels.append(CLASS_TO_IDX[class_name])
                            self.class_names.append(class_name)
                            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a blank image in case of error
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, str(img_path), class_name

def extract_features(model_name, splits_dir, output_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for extracting features with {model_name}")
    
    # 1. Load Model and strip final classifier layer
    if model_name == "resnet50":
        model = get_resnet50(num_classes=16, pretrained=True)
        # ResNet50: global pool is before fc
        # Hook or slice the model:
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "mobilenet_v2":
        model = get_mobilenet_v2(num_classes=16, pretrained=True)
        # MobileNetV2: classifier is at .classifier
        # feature extractor is .features, plus global average pool
        class MobileNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.features = original_model.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                return x
        feature_extractor = MobileNetFeatureExtractor(model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Process Splits: train, val, test
    splits_dir = Path(splits_dir)
    all_features = []
    metadata_records = []
    
    for split in ["train", "val", "test"]:
        split_path = splits_dir / split
        if not split_path.exists():
            print(f"Split path {split_path} not found. Skipping.")
            continue
            
        dataset = FeatureExtractionDataset(split_path, transform=transform)
        if len(dataset) == 0:
            print(f"No images in split {split}. Skipping.")
            continue
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Extracting features for split: {split} ({len(dataset)} images)...")
        
        with torch.no_grad():
            for imgs, labels, paths, class_names in dataloader:
                imgs = imgs.to(device)
                features = feature_extractor(imgs)
                # Flatten
                features = torch.flatten(features, 1)
                features_np = features.cpu().numpy()
                
                for i in range(len(paths)):
                    all_features.append(features_np[i])
                    
                    p = Path(paths[i])
                    class_name = class_names[i]
                    plant = PLANT_DISEASE_CLASSES[class_name]["plant"]
                    disease = PLANT_DISEASE_CLASSES[class_name]["disease"]
                    
                    metadata_records.append({
                        "image_path": str(p.relative_to(project_root) if p.is_relative_to(project_root) else p),
                        "class_name": class_name,
                        "class_idx": CLASS_TO_IDX[class_name],
                        "plant": plant,
                        "disease": disease,
                        "split": split
                    })
                    
    if not all_features:
        print("No features extracted.")
        return
        
    features_arr = np.array(all_features)
    metadata_df = pd.DataFrame(metadata_records)
    
    # Save Outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    features_file = output_path / f"{model_name}_features.npy"
    metadata_file = output_path / f"{model_name}_metadata.csv"
    
    np.save(features_file, features_arr)
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Saved features to {features_file} Shape: {features_arr.shape}")
    print(f"Saved metadata to {metadata_file} Rows: {len(metadata_df)}")
    
    # 4. Visualization (PCA and t-SNE)
    visualize_features(features_arr, metadata_df, model_name)

def visualize_features(features, df_meta, model_name, plot_dir="outputs/features"):
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    
    num_samples = len(features)
    if num_samples < 2:
        print("Too few samples to perform dimensionality reduction.")
        return
        
    print(f"Performing PCA on {num_samples} samples...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Add coordinates to temp df
    df_plot = df_meta.copy()
    df_plot["PCA_1"] = features_pca[:, 0]
    df_plot["PCA_2"] = features_pca[:, 1]
    
    # PCA Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_plot, x="PCA_1", y="PCA_2", 
        hue="class_name", style="plant",
        palette="tab20", alpha=0.8
    )
    plt.title(f"PCA Visualisation of Deep Features ({model_name.upper()})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    pca_file = plot_path / f"{model_name}_pca.png"
    plt.savefig(pca_file)
    plt.close()
    print(f"Saved PCA plot to {pca_file}")
    
    # t-SNE Plot (if at least 10 samples)
    if num_samples >= 10:
        print(f"Performing t-SNE on {num_samples} samples...")
        # Cap max iterations and perplexity based on size
        perplexity = min(30, num_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        features_tsne = tsne.fit_transform(features)
        
        df_plot["tSNE_1"] = features_tsne[:, 0]
        df_plot["tSNE_2"] = features_tsne[:, 1]
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=df_plot, x="tSNE_1", y="tSNE_2", 
            hue="class_name", style="plant",
            palette="tab20", alpha=0.8
        )
        plt.title(f"t-SNE Visualisation of Deep Features ({model_name.upper()})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        tsne_file = plot_path / f"{model_name}_tsne.png"
        plt.savefig(tsne_file)
        plt.close()
        print(f"Saved t-SNE plot to {tsne_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract deep features from split dataset_v2")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "mobilenet_v2"],
                        help="Pretrained backbone to extract features from")
    parser.add_argument("--splits-dir", type=str, default="dataset_v2/splits",
                        help="Directory containing train/val/test splits")
    parser.add_argument("--output-dir", type=str, default="dataset_v2/features",
                        help="Directory to save extracted features and metadata")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for feature extraction")
    args = parser.parse_args()
    
    extract_features(
        model_name=args.model,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )