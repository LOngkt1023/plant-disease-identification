# 🌿 Nhận Dạng Bệnh Cây Trồng (Plant Disease Classification)

> **Mục tiêu:** Quản lý vòng đời dữ liệu học sâu chuyên nghiệp và thiết lập quy trình chuẩn để thu thập dữ liệu, làm sạch và huấn luyện mạng Nơ-ron sâu tự động phân loại các loại bệnh trên cây trồng.

---

## 🚀 Các Tính Năng Nổi Bật
- **Hệ Thống Crawler Anti-Bot:** Thu thập dữ liệu thông minh trên diện rộng sử dụng các cơ chế bypass anti-bot, rotate proxies và mô phỏng trình duyệt (`crawler_stealth.py`, `proxy.py`, `search_profiles.py`).
- **Data Pipeline Chuyên Sâu:** Tự động hoá xử lý rác dữ liệu: phân giải URL, xoá trùng lặp, dùng dải phân vị loại bỏ file dị thường, xử lý siêu dữ liệu metadata lỗi (`convert_metadata.py`, `filter_metadata.py`, `keyword_filter.py`, `data_cleaning.py`).
- **Trung Tâm Deep Learning:** Các pipeline chuyển từ quá trình tiền xử lý ảnh và dán nhãn (`preprocessing.py`) sang kiến trúc mạng tối ưu cho Transfer Learning với thư viện PyTorch (`models.py`).

---

## 📂 Kiến Trúc Dự Án (Project Structure)

Dự án được phân cấp rõ ràng giữa khu vực nghiên cứu/thử nghiệm (Notebooks), mã nguồn hệ thống sản xuất (Scripts), và kho lưu trữ vật lý (Data/Models):

```text
plant-disease-identification/
│
├── data/                           # 📁 Kho dữ liệu được quản lý theo luồng
│   ├── raw/                        # Dữ liệu ảnh thô tải từ mạng về
│   ├── processed/                  # Dữ liệu ảnh sạch sau khi lọc rác & làm chuẩn
│   └── augmented/                  # Dữ liệu đã được Data Augmentation
│
├── notebooks/                      # 📁 Tầng Notebook tương tác & thử nghiệm nghiên cứu
│   ├── 01_data_crawling.ipynb      # Thử nghiệm các bộ cào dữ liệu và proxy
│   ├── 02_eda_preprocessing.ipynb  # Khám phá EDA, vẽ biểu đồ, thử thuật toán lọc ảnh
│   └── 03_model_training.ipynb     # Thử nghiệm thiết kế mạng CV và huấn luyện nháp
│
├── src/                            # 📁 Tầng Mã Nguồn Lõi (Core Source Code)
│   ├── config.py                   # Cấu hình chung và siêu tham số mạng (Hyperparams)
│   ├── crawler_stealth.py          # Bộ crawler chống phát hiện (nodriver/undetected)
│   ├── search_profiles.py          # Chỉ định từ khóa & các cấu hình trang lấy ảnh
│   ├── proxy.py                    # Rotate/Quản lý kết nối Proxy & User-agents
│   ├── convert_metadata.py         # Chuyển đổi định dạng siêu dữ liệu tải về
│   ├── filter_metadata.py          # Lọc metadata (Bỏ các bản ghi thiếu URL, size)
│   ├── keyword_filter.py           # Module thuật toán xử lý nhãn và từ khóa sai
│   ├── data_cleaning.py            # Làm sạch File vật lý (Deduplicate, Outliers IQR)
│   ├── preprocessing.py            # Resize, Format, Crop ảnh hàng loạt
│   ├── storage.py                  # Module IO xử lý đọc/ghi ở tốc độ cao
│   ├── utils.py                    # Helper (hàm logging, timer, check directory)
│   └── models.py                   # Backbone architectures: ResNet, MobileNet...
│
├── reports/                        # 📁 Lưu báo cáo và hình ảnh học thuật
│   ├── evaluation.txt              # Chỉ số F1, Recall, Precision của mạng sâu
│
├── requirements.txt                # Thư viện quy chuẩn của dự án (PyTorch, Selenium...)
├── SETUP_GUIDE.md                  # Hướng dẫn chi tiết setup Môi trường (Conda/venv)
└── README.md                       # Tài liệu hướng dẫn bộ quy tắc chung
```

---

## 🛠️ Hướng Dẫn Khởi Chạy (Quick Start)

### 1. Khởi Tạo Môi Trường Làm Việc
Vui lòng tham khảo kịch bản lỗi ở `SETUP_GUIDE.md`. Cài đặt nhanh với Minconda (Conda):
```bash
conda create -n crop-disease python=3.11 -y
conda activate crop-disease
pip install -r requirements.txt
```

### 2. Quy Trình Vận Hành 4 Bước Tiêu Chuẩn (Workflow)
Sự thành công của mô hình máy học phần lớn ở sự tuân thủ quy trình Sandbox To Production:
1. **Nghiên Cứu Thử Nghiệm (Notebooks):** Dùng thư mục `notebooks/` để trải nghiệm logic cào ảnh, phân tích trực quan EDA, và kịch bản mạng Neural.
2. **Triển Khai Hoàn Thiện (Src/):** Đoạn mã chạy ngon trên Notebook sẽ được đúc kết thành các Class & Function để bỏ vào `src/` theo đúng chức trách.
3. **Thực Thi Công Đoạn Dữ Liệu:**
   - **Crawling:** `python src/crawler_stealth.py` -> Đưa ảnh về `data/raw/`
   - **Làm Sạch (Filtering):** `python src/data_cleaning.py` kết hợp các module Metadata -> `data/processed/`.
   - **Chống Dị Biến (Preprocessing):** Cân bằng kích cỡ qua `src/preprocessing.py`.
4. **Bắt Đầu Huấn Luyện (Training):** 
   - Tiến hành thiết lập trong `src/config.py` và chạy kiến trúc tại `src/models.py`. Có thể phân tán Train Script hoặc Jupyter cho Server GPU. Nhận báo cáo thành tích vào `reports/`.

---

## 🧭 Cơ Chế Xuyên Thủng Hệ Thống Anti-Bot
Với lượng lớn dữ liệu (ví dụ 10.000 ảnh/layer), các bộ công cụ thông thường sẽ bị ngắt luồng (Blocked/Rate Limited). Dự án phòng hộ bằng:
- **Tàng Hình Cấp Độ Trình Duyệt:** Giấu toàn bộ thuộc tính `navigator.webdriver` thông qua `nodriver` hoặc `undetected-chromedriver`.
- **Hành Vi Sinh Học (Jitter Delays):** Không có cỗ máy nào click chuột cứ 1.5s/lần trọn đời. Bot thêm độ trễ ngẫu nhiên cùng các thao tác scroll ngắt quãng tại `search_profiles.py`.
- **Địa Chỉ Hoán Đổi Liên Tục:** Thay đổi định tuyến hệ thống mạng Residential thông qua logic cấu hình tại `proxy.py`.

---

## 🧹 Triết Lý "Rác Vào, Rác Ra" Và Cách Làm Sạch
Mọi dòng code ML đều chết đứng nếu Dataset không đạt chuẩn. Bộ lọc trong `data_cleaning.py` áp tiêu chuẩn khắt khe cho từng tấm ảnh:
1. **Chống Trùng Lặp Cấp Vật Lý (Deduplication):** Cấu thành chuỗi hàm băm (Hash SHA-256) xử tử đối với các URL hoặc content bị lập lại.
2. **Khuyết Thiếu Răn Đe:** Các records Metadata nếu thiếu link hoặc size sẽ tự động gạch bỏ.
3. **Khoanh Vùng Outliers Bằng IQR:** Nếu bức hình chỉ có `1KB` hoặc thuộc cỡ Panorama bản trích lục (vượt 1.5 lần dải nội phân vị IQR), nó là rác nhiễu và lập tức bị Drop.
4. **Hậu Chuẩn Cấu Trúc File:** Tất cả được `Image.open()` (Pillow/OpenCV) giải mã thử. Nghi ngờ hình lỗi/Corrupt sẽ bị Delete vật lý và dọn dẹp khỏi DataFrame. File sau đó cùng thống nhất ép về `*.jpg` cho Data Loader của PyTorch vào việc.