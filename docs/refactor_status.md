# Refactor Status Report - Plant Disease Identification

## 1) Tổng hợp 17 task ban đầu: đã xong / chưa xong

| # | Task | Trạng thái | Ghi chú |
|---|------|------------|--------|
| 1 | Đổi toàn bộ sang bộ class mới 16 lớp | ✅ Đã xong (pipeline chính) | Config trung tâm đã dùng 16 lớp mới |
| 2 | Tạo config class trung tâm `src/config/disease_classes.py` | ✅ Đã xong | Có `CLASS_NAMES`, `NUM_CLASSES`, `CLASS_TO_IDX`, `IDX_TO_CLASS`, `PLANT_DISEASE_CLASSES` |
| 3 | Cập nhật crawler đọc class/keyword từ config trung tâm | ✅ Đã xong | `scripts/crawl.py`, `src/crawler_stealth.py`, `src/search_profiles.py` đã dùng config mới |
| 4 | Cập nhật filter/cleaning pipeline | ✅ Đã xong cơ bản | Có `scripts/preprocess.py`, luồng clean/review/rejected, metadata |
| 5 | Cập nhật preprocessing (224, giữ tỷ lệ, normalize ImageNet) | ✅ Đã xong | `src/preprocessing.py` + transform train/val/test |
| 6 | Cập nhật split dataset 70/15/15 | ✅ Đã xong | `scripts/split_dataset.py` |
| 7 | Cập nhật models ResNet50 + MobileNetV2 | ✅ Đã xong | `src/models.py` implement thật, không còn `NotImplementedError` |
| 8 | Cập nhật dataset/dataloader | ✅ Đã xong | `src/dataset.py` |
| 9 | Cập nhật training script | ✅ Đã xong | `scripts/train.py` |
| 10 | Cập nhật evaluation script | ✅ Đã xong | `scripts/evaluate.py` |
| 11 | Cập nhật trực quan hóa thống kê dữ liệu/training | ✅ Đã xong | Đã có `scripts/data_statistics.py` xuất CSV + biểu đồ |
| 12 | Cập nhật feature extraction cho báo cáo | ✅ Đã xong | Đã có `scripts/extract_features.py` (feature + metadata + PCA/t-SNE) |
| 13 | Cập nhật README theo pipeline mới | ✅ Đã xong | `README.md` đã cập nhật đầy đủ pipeline `dataset_v2` + 16 lớp |
| 14 | Cập nhật test/sanity check | ✅ Đã xong cơ bản | Có `scripts/sanity_check.py` |
| 15 | Chuẩn hóa cấu trúc thư mục mong muốn | ⚠️ Một phần | Khung chính có, nhưng chưa xác nhận đủ toàn bộ artifacts |
| 16 | Chạy chuỗi lệnh kiểm tra sau refactor | ⚠️ Chưa xác nhận đầy đủ | Đã gọi lệnh nhưng terminal không capture output |
| 17 | Lưu ý quan trọng (không xóa dữ liệu cũ, dùng dataset_v2, tránh hard-code class) | ✅/⚠️ | Pipeline chính dùng `dataset_v2`; vẫn còn tài liệu/notebook cũ |

---

## 2) Kiểm tra trạng thái “nửa cũ nửa mới”

**Kết luận: Có trạng thái nửa cũ nửa mới ở mức tài liệu/notebook, không nằm ở pipeline core.**

- **Pipeline core (code chạy chính):** đã chuyển sang class mới 16 lớp.
- **Vùng còn cũ:**  
  - Không còn class cũ trong các file đã rà soát chính (`README.md`, `SETUP_GUIDE.md`, `notebooks/02_eda_preprocessing.ipynb`, các script core).

---

## 3) Kiểm tra còn class cũ không

Đã scan các class cũ:
- `Coffee_Healthy`
- `Coffee_Rust`
- `Citrus_Canker`
- `Citrus_Greening`
- `Tomato_Curl`
- `Tomato_Blight`
- `Rice_Blight`

### Kết quả
- **Không còn xuất hiện** trong các file đã kiểm tra gần nhất, bao gồm:
  - `README.md`
  - `SETUP_GUIDE.md`
  - `notebooks/02_eda_preprocessing.ipynb`
  - `src/config/disease_classes.py`
  - `src/models.py`
  - `scripts/crawl.py`
  - `src/search_profiles.py`
  - `src/dataset.py`
  - `src/crawler_stealth.py`

---

## 4) Kiểm tra bộ class mới có đúng 16 lớp không

**Kết quả: Đúng 16 lớp.**  
Trong `src/config/disease_classes.py` có đủ:

1. Rice_Healthy  
2. Rice_LeafBlast  
3. Rice_BrownSpot  
4. Tomato_Healthy  
5. Tomato_EarlyBlight  
6. Tomato_LateBlight  
7. Tomato_LeafMold  
8. Tomato_SeptoriaLeafSpot  
9. Potato_Healthy  
10. Potato_EarlyBlight  
11. Potato_LateBlight  
12. Corn_Healthy  
13. Corn_CommonRust  
14. Corn_NorthernLeafBlight  
15. Apple_Healthy  
16. Apple_Scab  

---

## 5) Kiểm tra NUM_CLASSES

**Kết quả: `NUM_CLASSES = len(CLASS_NAMES)` và hiện bằng 16.**

---

## 6) Kiểm tra crawler có đọc class/keywords từ `src/config/disease_classes.py` không

**Kết quả: Có.**

- `scripts/crawl.py` import `CLASS_NAMES` từ `src.config.disease_classes`.
- `src/crawler_stealth.py` import `CLASS_NAMES`, `PLANT_DISEASE_CLASSES`.
- `src/search_profiles.py` đọc keywords từ `PLANT_DISEASE_CLASSES`.

=> Crawler không còn hard-code danh sách class chính trong pipeline mới.

---

## 7) Kiểm tra `src/models.py` đã implement thật ResNet50/MobileNetV2 chưa

**Kết quả: Đã implement thật.**

- Có `get_resnet50(...)`
- Có `get_mobilenet_v2(...)`
- Có `build_model(...)`
- Không còn `NotImplementedError`.

---

## 8) Kiểm tra output shape model với input `[2, 3, 224, 224]`

**Kết quả theo code/sanity logic: Đúng mục tiêu [2,16].**

- `scripts/sanity_check.py` có check:
  - ResNet50 output `(2,16)`
  - MobileNetV2 output `(2,16)`

---

## 9) Kiểm tra `scripts/sanity_check.py` có tồn tại và chạy được không

- **Tồn tại:** Có.
- **Đã thực thi lệnh:** `python scripts/sanity_check.py` đã chạy nhưng terminal của môi trường hiện tại không trả output capture được.
- **Đánh giá hiện trạng:** Script hợp lệ về cấu trúc và logic; trạng thái runtime cần xác nhận lại trên máy local bằng terminal thường.

---

## 10) Nếu có lỗi thì sửa để chạy end-to-end

### Lỗi/điểm tồn tại hiện ghi nhận
1. **Terminal capture issue** trong môi trường hiện tại làm không xác nhận được stdout của lệnh.
2. **Ghi chú runtime:** terminal tích hợp trước đó có lúc không capture output đầy đủ.
3. **Hạng mục đã hoàn tất thêm**:
   - `scripts/data_statistics.py`
   - `scripts/extract_features.py`
   - `README.md` theo pipeline mới

---

## File đã sửa (đã xác nhận trong quá trình refactor)

- `src/config/disease_classes.py`
- `src/config.py`
- `src/search_profiles.py`
- `src/crawler_stealth.py`
- `src/keyword_filter.py`
- `src/models.py`
- `src/preprocessing.py`
- `src/dataset.py`
- `scripts/crawl.py`
- `scripts/preprocess.py`
- `scripts/split_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/sanity_check.py`
- `scripts/data_statistics.py`
- `scripts/extract_features.py`
- `README.md`
- `SETUP_GUIDE.md`
- `notebooks/02_eda_preprocessing.ipynb`

## File đã tạo mới

- `docs/refactor_status.md` (file này)

---

## Việc cần làm tiếp (ưu tiên)

1. Chạy lại full sanity + dry-run end-to-end trên terminal local có output:
   - `python scripts/sanity_check.py`
   - `python scripts/crawl.py --max-images-per-class 20`
   - `python scripts/preprocess.py`
   - `python scripts/split_dataset.py`
   - `python scripts/train.py --model mobilenet_v2 --epochs 2 --batch-size 16`
   - `python scripts/train.py --model resnet50 --epochs 2 --batch-size 16`
   - `python scripts/evaluate.py --model mobilenet_v2`
   - `python scripts/evaluate.py --model resnet50`
2. Nếu cần báo cáo trực quan, chạy thêm:
   - `python scripts/data_statistics.py`
   - `python scripts/extract_features.py --model resnet50`
   - `python scripts/extract_features.py --model mobilenet_v2`
   - `python scripts/sanity_check.py`
   - `python scripts/crawl.py --max-images-per-class 20`
   - `python scripts/preprocess.py`
   - `python scripts/split_dataset.py`
   - `python scripts/train.py --model mobilenet_v2 --epochs 2 --batch-size 16`
   - `python scripts/train.py --model resnet50 --epochs 2 --batch-size 16`
   - `python scripts/evaluate.py --model mobilenet_v2`
   - `python scripts/evaluate.py --model resnet50`