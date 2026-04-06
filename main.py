import numpy as np
import time
from myModel.randomforest import MyRandomForest  # Đổi tên file import cho đúng với file của ông nha

print("🚀 ĐANG LOAD DỮ LIỆU HOG-PCA THẬT...")
try:
    X_train = np.load('feature_train_hog.npy')
    y_train = np.load('label_train.npy')
    X_test = np.load('feature_test_hog.npy')
    y_test = np.load('label_test.npy')
    print(f"Đã load xong! Kích thước Train gốc: {X_train.shape}")
except FileNotFoundError:
    print("Không tìm thấy file .npy! Ông nhớ để chung thư mục nhé.")
    exit()

# ---------------------------------------------------------

print(f"\nĐang lấy {len(y_train)} ảnh để dạy rừng...")

# Tính số lượng cột cho mỗi cây (Căn bậc 2 của tổng số cột PCA)
n_features = int(np.sqrt(X_train.shape[1]))

# Khởi tạo rừng: 20 cây, mỗi cây cao tối đa 10 tầng
forest = MyRandomForest(n_trees=20, max_depth=7, min_samples_split= 5, n_features=n_features)

start_time = time.time()
forest.fit(X_train, y_train)
print(f"⏳ Thời gian trồng rừng: {time.time() - start_time:.2f} giây")

print("\n🔍 Đang đem Khu rừng đi dự đoán tập Test...")
predictions = forest.predict(X_test)

# Đánh giá
accuracy = np.sum(predictions == y_test) / len(y_test) * 100
print(f"🎉 Độ chính xác trên tập thực tế thu nhỏ: {accuracy:.2f}%")