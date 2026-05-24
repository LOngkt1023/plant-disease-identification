# 🌿 Nhận Dạng Bệnh Cây Trồng (Plant Disease Classification)

> **Đề tài:** Xây dựng hệ thống nhận dạng bệnh cây trồng từ ảnh lá sử dụng học sâu (Deep Learning).  
> Dữ liệu tự crawl từ internet. Huấn luyện tối thiểu 2 mô hình (ResNet50 & MobileNetV2).  
> Báo cáo tiểu luận tối thiểu 35 trang.

---

## 🎯 Mục Tiêu Dự Án

- Tự crawl >10.000 ảnh bệnh cây từ Google Images
- Làm sạch và gán nhãn theo **16 lớp** cây trồng / bệnh cây
- Huấn luyện **ResNet50** và **MobileNetV2** bằng Transfer Learning
- Đạt **Test Accuracy ≥ 85%**
- Xuất báo cáo đầy đủ: confusion matrix, classification report, loss/accuracy curve

---

## 🗂️ Bộ 16 Class

| Class | Cây | Bệnh |
|---|---|---|
| Rice_Healthy | Lúa | Khỏe mạnh |
| Rice_LeafBlast | Lúa | Đạo ôn lá |
| Rice_BrownSpot | Lúa | Đốm nâu |
| Tomato_Healthy | Cà chua | Khỏe mạnh |
| Tomato_EarlyBlight | Cà chua | Cháy lá sớm |
| Tomato_LateBlight | Cà chua | Mốc sương |
| Tomato_LeafMold | Cà chua | Mốc lá |
| Tomato_SeptoriaLeafSpot | Cà chua | Đốm lá Septoria |
| Potato_Healthy | Khoai tây | Khỏe mạnh |
| Potato_EarlyBlight | Khoai tây | Cháy lá sớm |
| Potato_LateBlight | Khoai tây | Mốc sương |
| Corn_Healthy | Ngô | Khỏe mạnh |
| Corn_CommonRust | Ngô | Gỉ sắt phổ thông |
| Corn_NorthernLeafBlight | Ngô | Cháy lá phía bắc |
| Apple_Healthy | Táo | Khỏe mạnh |
| Apple_Scab | Táo | Bệnh ghẻ lá |

---

## 📂 Cấu Trúc Dự Án

```text
plant-disease-identification/
├── dataset_v2/                       # Dataset phiên bản 2 (16 class)
│   ├── raw/{class_name}/             # Ảnh thô từ crawl
│   ├── clean/{class_name}/           # Ảnh sạch qua filter
│   ├── review/{class_name}/          # Ảnh nghi ngờ, cần review thủ công
│   ├── rejected/{class_name}/        # Ảnh bị loại
│   ├── processed/{class_name}/       # Ảnh đã resize 224x224
│   ├── splits/
│   │   ├── train/{class_name}/       # 70% train
│   │   ├── val/{class_name}/         # 15% validation
│   │   └── test/{class_name}/        # 15% test
│   ├── metadata/
│   │   ├── raw_metadata.parquet
│   │   ├── clean_metadata.parquet
│   │   ├── split_metadata.parquet
│   │   └── crawl_summary.csv
│   └── features/
│       ├── resnet50_features.npy
│       ├── resnet50_metadata.csv
│       ├── mobilenet_v2_features.npy
│       └── mobilenet_v2_metadata.csv
│
├── src/
│   ├── config/
│   │   └── disease_classes.py        # ⭐ Config trung tâm: 16 class, keywords
│   ├── config.py                     # Paths và settings
│   ├── dataset.py                    # Dataset & DataLoader
│   ├── models.py                     # ResNet50, MobileNetV2
│   ├── preprocessing.py              # Resize, normalize
│   ├── crawler_stealth.py            # Crawler anti-bot
│   ├── keyword_filter.py             # Lọc keyword
│   └── utils.py                      # Helper functions
│
├── scripts/
│   ├── crawl.py                      # Crawl ảnh từ Google Images
│   ├── preprocess.py                 # Làm sạch + resize ảnh
│   ├── split_dataset.py              # Tách train/val/test 70/15/15
│   ├── data_statistics.py            # Thống kê + biểu đồ dữ liệu
│   ├── extract_features.py           # Trích xuất features + PCA/t-SNE
│   ├── train.py                      # Train ResNet50 / MobileNetV2
│   ├── evaluate.py                   # Evaluate trên test set
│   └── sanity_check.py               # Kiểm tra nhanh toàn bộ pipeline
│
├── notebooks/
│   ├── 01_data_crawling.ipynb        # Notebook crawl + EDA cơ bản
│   ├── 02_eda_preprocessing.ipynb    # EDA nâng cao + preprocessing
│   └── 03_model_training.ipynb       # Training loop + visualize
│
├── outputs/
│   ├── resnet50/                     # Kết quả train ResNet50
│   ├── mobilenet_v2/                 # Kết quả train MobileNetV2
│   ├── data_statistics/              # Biểu đồ thống kê dữ liệu
│   └── features/                    # PCA/t-SNE plots
│
├── docs/
│   └── refactor_status.md            # Trạng thái refactor
├── requirements.txt
├── SETUP_GUIDE.md
└── README.md
```

---

## ⚙️ Cài Đặt Môi Trường

```bash
conda create -n crop-disease python=3.11 -y
conda activate crop-disease
pip install -r requirements.txt
```

---

## 🔄 Pipeline Đầy Đủ

### 1. Crawl Dữ Liệu

```bash
# Crawl thử (20 ảnh/class)
python scripts/crawl.py --max-images-per-class 20

# Crawl thật (1500 ảnh/class → ~24.000 ảnh thô)
python scripts/crawl.py --max-images-per-class 1500
```

Ảnh được lưu vào: `dataset_v2/raw/{class_name}/`

### 2. Làm Sạch & Preprocess

```bash
python scripts/preprocess.py
```

Output:
- `dataset_v2/clean/` — ảnh hợp lệ
- `dataset_v2/review/` — ảnh nghi ngờ
- `dataset_v2/rejected/` — ảnh bị loại

Threshold:
- `clip_score >= 0.60` → clean
- `0.35 <= clip_score < 0.60` → review
- `clip_score < 0.35` → rejected

### 3. Chia Tập Train/Val/Test

```bash
python scripts/split_dataset.py
```

Tỷ lệ: 70% Train / 15% Val / 15% Test — stratified split theo class.

Output: `dataset_v2/splits/train|val|test/{class_name}/`

### 4. Thống Kê Dữ Liệu

```bash
python scripts/data_statistics.py
```

Output vào `outputs/data_statistics/`:
- `class_distribution.png`
- `plant_distribution.png`
- `disease_distribution.png`
- `status_distribution.png`
- `source_domain_distribution.png`
- `keyword_distribution.png`
- `image_size_distribution.png`
- `data_summary.csv`

### 5. Trích Xuất Features

```bash
python scripts/extract_features.py --model resnet50
python scripts/extract_features.py --model mobilenet_v2
```

Output vào `dataset_v2/features/`:
- `resnet50_features.npy` / `resnet50_metadata.csv`
- `mobilenet_v2_features.npy` / `mobilenet_v2_metadata.csv`

Plots vào `outputs/features/`:
- `resnet50_pca.png`, `resnet50_tsne.png`
- `mobilenet_v2_pca.png`, `mobilenet_v2_tsne.png`

### 6. Huấn Luyện Mô Hình

```bash
# Train ResNet50 (full 30 epoch)
python scripts/train.py --model resnet50 --epochs 30 --batch-size 32

# Train MobileNetV2
python scripts/train.py --model mobilenet_v2 --epochs 30 --batch-size 32

# Train nhanh thử (2 epoch)
python scripts/train.py --model resnet50 --epochs 2 --batch-size 16
python scripts/train.py --model mobilenet_v2 --epochs 2 --batch-size 16
```

Output vào `outputs/{model_name}/`:
- `best_checkpoint.pt`
- `last_checkpoint.pt`
- `train_history.csv` / `train_history.json`

### 7. Đánh Giá Mô Hình

```bash
python scripts/evaluate.py --model resnet50
python scripts/evaluate.py --model mobilenet_v2
```

Output vào `outputs/{model_name}/`:
- `classification_report.txt`
- `classification_report.csv`
- `confusion_matrix.png`
- `predictions.csv`

---

## ✅ Kết Quả Mong Đợi

| Chỉ số | Mục tiêu |
|---|---|
| Tổng ảnh sạch | > 10.000 ảnh |
| Số class | 16 |
| Train Accuracy | > 90% |
| Test Accuracy | **≥ 85%** |
| F1-score (macro) | > 0.85 |

---

## 🔍 Kiểm Tra Nhanh (Sanity Check)

```bash
python scripts/sanity_check.py
```

Kiểm tra:
- `NUM_CLASSES == 16`
- `CLASS_TO_IDX` / `IDX_TO_CLASS` đúng
- Mỗi class có keywords
- ResNet50 output shape `[2, 16]`
- MobileNetV2 output shape `[2, 16]`
- Không có md5 overlap giữa train/val/test
- Các file script quan trọng tồn tại

---

## 📌 Lưu Ý Quan Trọng

- **Không hard-code class** rải rác. Tất cả class và keywords được quản lý tại `src/config/disease_classes.py`.
- Dataset cũ (nếu có) trong `data/` **không bị xóa**. Dataset mới dùng `dataset_v2/`.
- Không dùng augmentation để tính số lượng crawl gốc.
- Augmentation chỉ áp dụng trong DataLoader lúc train.

---

## 🏗️ Chiến Lược Fine-tuning

```
Giai đoạn 1 (3–5 epoch):
  - Freeze backbone
  - Chỉ train classifier head
  - Learning rate: 1e-3

Giai đoạn 2 (25 epoch):
  - Unfreeze 1–2 block cuối backbone
  - Fine-tune toàn bộ
  - Learning rate: 1e-4 → 1e-5 (ReduceLROnPlateau)
```

---

## 📖 Cấu Trúc Module Chính

| File | Mô tả |
|---|---|
| `src/config/disease_classes.py` | ⭐ Config trung tâm: 16 class, keywords, plant/disease info |
| `src/config.py` | Paths, device, training hyperparameters |
| `src/models.py` | `get_resnet50()`, `get_mobilenet_v2()`, `build_model()` |
| `src/dataset.py` | `PlantDiseaseDataset`, DataLoader builders |
| `src/preprocessing.py` | Resize với padding, normalize theo ImageNet |
| `scripts/crawl.py` | Crawl ảnh từ Google Images theo keywords |
| `scripts/preprocess.py` | Lọc ảnh lỗi, resize, tính hash, xuất metadata |
| `scripts/split_dataset.py` | Stratified split 70/15/15 |
| `scripts/data_statistics.py` | Thống kê + biểu đồ phân phối dữ liệu |
| `scripts/extract_features.py` | Deep feature extraction + PCA/t-SNE |
| `scripts/train.py` | Training loop đầy đủ với checkpoint + history |
| `scripts/evaluate.py` | Đánh giá test set: F1, confusion matrix, report |
| `scripts/sanity_check.py` | Kiểm tra nhanh toàn bộ pipeline |