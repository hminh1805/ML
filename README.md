# 🐾 Bài tập lớn Học máy: Phân loại Ảnh Chó và Mèo (Dogs vs. Cats)



## 🎯 Mục tiêu dự án
1. **Tiền xử lý & EDA:** Hiểu rõ bản chất dữ liệu, kích thước, và phân phối nhãn.
2. **Trích xuất đặc trưng:** Ứng dụng mô hình Pretrained (ResNet/VGG) để chuyển đổi ảnh thô thành vector đặc trưng.
3. **Huấn luyện mô hình Cổ điển:** Tự triển khai/tinh chỉnh các thuật toán phân lớp tuyến tính (SVM, Logistic Regression).
4. **Đánh giá & So sánh:** Đánh giá hiệu năng các mô hình thông qua F1-score, Precision, Recall và ma trận nhầm lẫn.

---

## 📂 Cấu trúc thư mục

```text
📦 BTL_MachineLearning
 ┣ 📂 PetImages/          # Thư mục chứa 25.000 ảnh gốc (Không push lên Git)
 ┣ 📂 minidataset/           # Tập dữ liệu đã lấy mẫu (Train: 2000, Test: 400)
 ┃ ┣ 📂 train/               # Ảnh huấn luyện (chia 2 thư mục cats/ và dogs/)
 ┃ ┗ 📂 test/                # Ảnh kiểm tra (chia 2 thư mục cats/ và dogs/)
 ┣ 📜 make_minidataset.py       # Script tự động lấy mẫu ngẫu nhiên/cân bằng
 ┣ 📜 checkdata.py                 # Script Phân tích dữ liệu khám phá (Xuất biểu đồ)
 ┣ 📜 requirements.txt       # Danh sách thư viện cần thiết
 ┣ 📜 .gitignore             # File cấu hình Git
 ┗ 📜 README.md              # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Hướng dẫn cài đặt và Chạy dự án

### 1. Thiết lập môi trường
Khuyến nghị sử dụng môi trường ảo (`venv`) để tránh xung đột thư viện:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate
# Kích hoạt môi trường (Mac/Linux)
source venv/bin/activate

# Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

### 2. Chuẩn bị Dữ liệu (Data Preparation)
Dự án sử dụng bộ dữ liệu [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).
1. Tải bộ dữ liệu gốc từ Kaggle và giải nén thư mục `PetImages` (gồm 2 thư mục con `Cat` và `Dog`) vào thư mục gốc của dự án.
2. Chạy script lấy mẫu để tạo ra tập `minidataset` cân bằng (tốc độ train nhanh hơn, tránh thiên vị):

```bash
python make_minidataset.py
```
*Lưu ý: Mặc định script sẽ lấy 1000 ảnh Train và 200 ảnh Test cho mỗi nhãn.*

### 3. Phân tích dữ liệu khám phá (EDA)
Để kiểm tra phân phối dữ liệu, kích thước trung bình của ảnh và xuất các biểu đồ trực quan, chạy lệnh:

```bash
python checkdata.py
```
Script sẽ tự động tạo ra 3 file báo cáo hình ảnh:
- `pt_distribution.png`: Biểu đồ cột thể hiện sự cân bằng nhãn.
- `pt_dimensions.png`: Biểu đồ phân tán (scatter) kích thước các ảnh đầu vào.
- `pt_samples.png`: Lưới hình ảnh trực quan hóa mẫu dữ liệu.

---

## 🔗 Liên kết tài liệu cuối kỳ
- **Báo cáo chi tiết (PDF):** `[Sẽ cập nhật link hoặc đường dẫn file]`
- **Slide Thuyết trình:** `[Sẽ cập nhật link hoặc đường dẫn file]`
- **Google Colab Notebook:** `[Sẽ cập nhật link chạy online]`


# ĐỒ ÁN HỌC MÁY: PHÂN LOẠI ẢNH CHÓ VÀ MÈO (DOGS VS. CATS)
Hiện thực các thuật toán Machine Learning từ con số không (From Scratch)

## 1. Tổng quan dự án
Dự án này là một hệ thống học máy hoàn chỉnh (End-to-End Machine Learning Pipeline) được xây dựng nhằm giải quyết bài toán phân loại ảnh nhị phân (Dogs vs. Cats) từ tập dữ liệu của Kaggle.

Điểm đặc biệt của dự án nằm ở việc không sử dụng các thư viện tích hợp sẵn (như Scikit-Learn hay TensorFlow) cho phần lõi thuật toán. Thay vào đó, các mô hình phân loại như Logistic Regression, Support Vector Machine (SVM), và Random Forest được lập trình hoàn toàn thủ công (from scratch) bằng Python và Numpy. Dự án thể hiện sự am hiểu sâu sắc về toán học cơ bản của các thuật toán, cũng như kỹ năng tối ưu hóa hệ thống, xử lý dữ liệu lớn và triển khai ứng dụng thực tế.

## 2. Các đặc điểm kỹ thuật cốt lõi
Tiền xử lý hình ảnh và Trích xuất đặc trưng: Sử dụng kỹ thuật Center Crop để chuẩn hóa không gian ảnh hình vuông (128x128), kết hợp với Histogram of Oriented Gradients (HOG) để trích xuất đặc trưng hình học.

Giảm chiều dữ liệu (Dimensionality Reduction): Ứng dụng Principal Component Analysis (PCA) để nén các vector HOG khổng lồ (hơn 8000 chiều) xuống mức tối ưu (giữ lại 95% phương sai thông tin), giúp loại bỏ nhiễu và tối ưu hóa tốc độ huấn luyện.

Thuật toán tự xây dựng (Custom Algorithms): Hiện thực thủ công Logistic Regression (sử dụng thuật toán Gradient Descent), SVM (tối ưu hóa Hinge Loss có điều chuẩn L2), và Random Forest (tự hoàn thiện kiến trúc cây quyết định CART với cơ chế song song hóa bằng multiprocessing và Numba).

Đối chuẩn hệ thống (Benchmarking): Tích hợp hệ thống đối chiếu hiệu năng trực tiếp với thư viện tiêu chuẩn công nghiệp Scikit-Learn để đánh giá độ chính xác và thời gian hội tụ.

Triển khai ứng dụng Web (Deployment): Đóng gói mô hình thành ứng dụng web tương tác thời gian thực bằng Streamlit với cơ chế lưu trữ bộ nhớ đệm (In-memory Caching) giúp thời gian suy luận (inference) đạt mức độ trễ cực thấp.

## 3. Yêu cầu hệ thống và Cài đặt
Hệ thống yêu cầu phiên bản Python 3.8 trở lên. Để đảm bảo môi trường độc lập và tránh xung đột thư viện, khuyến nghị sử dụng môi trường ảo (virtual environment).

Thiết lập môi trường:

```Bash
python -m venv venv

# Kích hoạt môi trường trên Windows:
.\venv\Scripts\activate

# Kích hoạt môi trường trên Mac/Linux:
source venv/bin/activate
```

Cài đặt các gói thư viện phụ thuộc:
```Bash
pip install -r requirements.txt
```

## 4. Hướng dẫn thực thi dự án
Để chạy dự án từ đầu, người dùng cần tải tập dữ liệu Kaggle Dogs vs. Cats và đặt 2 thư mục Cat và Dog vào bên trong thư mục PetImages. Sau đó, thực thi các lệnh sau theo đúng thứ tự:

Bước 1: Tạo phân vùng dữ liệu
Sử dụng script này để lấy mẫu ngẫu nhiên, tạo ra các tập Train/Valid/Test cân bằng hoặc tạo tập mini dataset để chạy kiểm thử.

Bash
python make_minidataset.py
Bước 2: Trích xuất đặc trưng HOG
Pha tiền xử lý hình ảnh, cắt trung tâm và trích xuất vector HOG cho toàn bộ hình ảnh được chi định trong file JSON.

```Bash
python extract_feature.py
```

Bước 3: Ép chiều dữ liệu với PCA
Sử dụng tập đặc trưng HOG vừa tạo để huấn luyện mô hình PCA, giữ lại 95% phương sai và lưu lại khuôn PCA để tái sử dụng sau này.

```Bash
python pca.py
```

Bước 4: Phân tích dữ liệu khám phá (EDA)
Xuất các biểu đồ trực quan hóa để đánh giá chất lượng dữ liệu như: Ma trận tương quan, biểu đồ tích lũy phương sai (Scree Plot), biểu đồ phân tán không gian 2D/3D. Kết quả được lưu tại thư mục eda_results.

```Bash
python eda.py
```
Bước 5: Huấn luyện và Đánh giá mô hình
Kích hoạt toàn bộ quá trình huấn luyện cho 3 mô hình tự code thông qua file main.py. Script sẽ tự động in ra bảng đánh giá hiệu năng (Accuracy, Precision, Recall, F1-Score, Thời gian) và lưu các mô hình dưới dạng file .pkl.

```Bash
python main.py
```

Bước 6: Chạy ứng dụng Web
Khởi chạy giao diện người dùng trên trình duyệt, cho phép tải ảnh lên và sử dụng các mô hình đã huấn luyện để dự đoán thời gian thực.

```Bash
streamlit run app.py
```

## 5. Kết quả thực nghiệm
Trên tập dữ liệu huấn luyện thực tế, hệ thống phân loại tuyến tính (Logistic Regression tự hiện thực) đạt độ chính xác hơn 74.3% kèm theo tốc độ suy luận chưa đến 0.01 giây cho toàn bộ tập kiểm thử, vượt trội hơn về tốc độ so với các mô hình quần thể (Ensemble) trong cùng không gian đặc trưng PCA. Các số liệu chi tiết được tổng hợp trong thư mục reports sau khi chạy quá trình đánh giá.