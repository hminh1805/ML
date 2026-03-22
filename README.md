# 🐾 Đồ án Học máy: Phân loại Ảnh Chó và Mèo (Dogs vs. Cats)



## 🎯 Mục tiêu dự án
1. [cite_start]**Tiền xử lý & EDA:** Hiểu rõ bản chất dữ liệu, kích thước, và phân phối nhãn[cite: 252, 253].
2. [cite_start]**Trích xuất đặc trưng:** Ứng dụng mô hình Pretrained (ResNet/VGG) để chuyển đổi ảnh thô thành vector đặc trưng[cite: 293, 294, 295, 296, 297].
3. [cite_start]**Huấn luyện mô hình Cổ điển:** Tự triển khai/tinh chỉnh các thuật toán phân lớp tuyến tính (SVM, Logistic Regression)[cite: 298].
4. [cite_start]**Đánh giá & So sánh:** Đánh giá hiệu năng các mô hình thông qua F1-score, Precision, Recall và ma trận nhầm lẫn[cite: 230].

---

## 📂 Cấu trúc thư mục

\`\`\`text
📦 BTL_MachineLearning
 ┣ 📂 dataset_full/          # Thư mục chứa 25.000 ảnh gốc (Không push lên Git)
 ┣ 📂 minidataset/           # Tập dữ liệu đã lấy mẫu (Train: 2000, Test: 400)
 ┃ ┣ 📂 train/               # Ảnh huấn luyện (chia 2 thư mục cats/ và dogs/)
 ┃ ┗ 📂 test/                # Ảnh kiểm tra (chia 2 thư mục cats/ và dogs/)
 ┣ 📜 make_minidata.py       # Script tự động lấy mẫu ngẫu nhiên/cân bằng
 ┣ 📜 eda.py                 # Script Phân tích dữ liệu khám phá (Xuất biểu đồ)
 ┣ 📜 requirements.txt       # Danh sách thư viện cần thiết
 ┣ 📜 .gitignore             # File cấu hình Git
 ┗ 📜 README.md              # Tài liệu hướng dẫn dự án
\`\`\`

---

## ⚙️ Hướng dẫn cài đặt và Chạy dự án

### 1. Thiết lập môi trường
Khuyến nghị sử dụng môi trường ảo (`venv`) để tránh xung đột thư viện:

\`\`\`bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate
# Kích hoạt môi trường (Mac/Linux)
source venv/bin/activate

# Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
\`\`\`

### 2. Chuẩn bị Dữ liệu (Data Preparation)
Dự án sử dụng bộ dữ liệu [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).
1. Tải bộ dữ liệu gốc từ Kaggle và giải nén thư mục `PetImages` (gồm 2 thư mục con `Cat` và `Dog`) vào thư mục gốc của dự án.
2. Chạy script lấy mẫu để tạo ra tập `minidataset` cân bằng (tốc độ train nhanh hơn, tránh thiên vị):

\`\`\`bash
python make_minidata.py
\`\`\`
*Lưu ý: Mặc định script sẽ lấy 1000 ảnh Train và 200 ảnh Test cho mỗi nhãn.*

### 3. Phân tích dữ liệu khám phá (EDA)
Để kiểm tra phân phối dữ liệu, kích thước trung bình của ảnh và xuất các biểu đồ trực quan, chạy lệnh:

\`\`\`bash
python eda.py
\`\`\`
Script sẽ tự động tạo ra 3 file báo cáo hình ảnh:
- `eda_distribution.png`: Biểu đồ cột thể hiện sự cân bằng nhãn.
- `eda_dimensions.png`: Biểu đồ phân tán (scatter) kích thước các ảnh đầu vào.
- `eda_samples.png`: Lưới hình ảnh trực quan hóa mẫu dữ liệu.

---

## 🔗 Liên kết tài liệu cuối kỳ
- **Báo cáo chi tiết (PDF):** `[Sẽ cập nhật link hoặc đường dẫn file]`
- **Slide Thuyết trình:** `[Sẽ cập nhật link hoặc đường dẫn file]`
- **Google Colab Notebook:** `[Sẽ cập nhật link chạy online]`