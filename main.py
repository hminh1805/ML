import time
import numpy as np
from myModel.randomforest import MyRandomForest  # Đổi tên file import cho đúng với file của ông nha
from myModel.logisticRegression import LogisticRegression  # Đổi tên file import cho đúng với file của ông nha
from myModel.SVM import SVM  # Đổi tên file import cho đúng với file của ông nha

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

# =========================================================
# RANDOM FOREST
# =========================================================
# print(f"\nĐang lấy {len(y_train)} ảnh để dạy rừng...")

# # Tính số lượng cột cho mỗi cây (Căn bậc 2 của tổng số cột PCA)
# n_features = int(np.sqrt(X_train.shape[1]))

# # Khởi tạo rừng: 20 cây, mỗi cây cao tối đa 10 tầng
# forest = MyRandomForest(n_trees=20, max_depth=7, min_samples_split= 5, n_features=n_features)

# start_time = time.time()
# forest.fit(X_train, y_train)
# print(f"⏳ Thời gian trồng rừng: {time.time() - start_time:.2f} giây")

# print("\n🔍 Đang đem Khu rừng đi dự đoán tập Test...")
# rf_predictions = forest.predict(X_test)

# # Đánh giá
# rf_accuracy = np.sum(rf_predictions == y_test) / len(y_test) * 100
# print(f"🎉 Độ chính xác trên tập thực tế thu nhỏ của Mô hình Random Forest: {rf_accuracy:.2f}%")

# =========================================================
# LOGISTIC REGRESSION
# =========================================================
logisticRegression = print("\n Huấn luyện mô hình LOGISTIC REGRESSION...")

logistic = LogisticRegression( learning_rate=0.1, epochs=1000)

start_time = time.time()

logistic.fit(X_train, y_train)
print(f"⏳ Thời gian train Logistic Regression: {time.time() - start_time:.2f} giây")

lr_predictions = logistic.predict(X_test)
lr_accucary = np.mean(lr_predictions == y_test) * 100
print(f"🎯 Logistic Regression Accuracy: {lr_accucary:.2f}%")

# =========================================================
# SVM
# =========================================================
# logisticRegression = print("\n Huấn luyện mô hình SVM...")

# svm = SVM( learning_rate=0.1, epochs=1000)

# start_time = time.time()
# svm.fit(X_train, y_train)
# print(f"⏳ Thời gian train SVM: {time.time() - start_time:.2f} giây")

# svm_predictions = svm.predict(X_test)
# svm_accucary = np.mean(svm_predictions == y_test) * 100
# print(f"🎯 Logistic Regression Accuracy: {svm_accucary:.2f}%")

# =========================================================
# So sánh
# =========================================================
print("\n📊 SO SÁNH KẾT QUẢ")
# print(f"Random Forest      : {rf_accuracy:.2f}%")
print(f"Logistic Regression: {lr_accucary:.2f}%")
# print(f"SVM                : {svm_accucary:.2f}%")