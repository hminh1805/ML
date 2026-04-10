import time
import numpy as np
from myModel.randomforest import MyRandomForest  # Đổi tên file import cho đúng với file của ông nha
from myModel.logisticRegression import LogisticRegression  # Đổi tên file import cho đúng với file của ông nha
from myModel.SVM import SVM  # Đổi tên file import cho đúng với file của ông nha
from make_minidataset import generate_dataset_indices
from extract_feature import run_hog
from pca import run_pca

#generate_dataset_indices(source_dir='PetImages', mode='mini')

#run_hog()
run_pca()

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
# f_predictions = forest.predict(X_test)

# # Đánh giá
# f_accuracy = np.sum(f_predictions == y_test) / len(y_test) * 100
# print(f"🎉 Độ chính xác trên tập thực tế thu nhỏ: {f_accuracy:.2f}%")

# =========================================================
# LOGISTIC REGRESSION
# =========================================================
# print("\n Huấn luyện mô hình LOGISTIC REGRESSION...")

# logistic = LogisticRegression( learning_rate=0.1, epochs=1000)

# start_time = time.time()

# logistic.fit(X_train, y_train)
# print(f"⏳ Thời gian train Logistic Regression: {time.time() - start_time:.2f} giây")

# lr_predictions = logistic.predict(X_test)
# lr_accucary = np.mean(lr_predictions == y_test) * 100
# print(f"🎯 Logistic Regression Accuracy: {lr_accucary:.2f}%")

# =========================================================
# SVM
# =========================================================
print("\n Huấn luyện mô hình SVM...")

svm = SVM( learning_rate=0.001, epochs=1000)

start_time = time.time()
svm.fit(X_train, y_train)
print(f"⏳ Thời gian train SVM: {time.time() - start_time:.2f} giây")

svm_predictions = svm.predict(X_test)
svm_accuracy = svm.score(X_test, y_test) * 100
# svm_accucary = np.mean(svm_predictions == y_test) * 100
print(f"🎯 SVM Accuracy: {svm_accuracy:.2f}%")

# =========================================================
# So sánh
# =========================================================
print("\n📊 SO SÁNH KẾT QUẢ")
# print(f"Random Forest      : {f_accuracy:.2f}%")
# print(f"Logistic Regression: {lr_accucary:.2f}%")
print(f"SVM                : {svm_accuracy:.2f}%")



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(y_test, preds_list, names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (preds, name) in enumerate(zip(preds_list, names)):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Mèo (0)', 'Chó (1)'], 
                    yticklabels=['Mèo (0)', 'Chó (1)'])
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Dự đoán')
        axes[i].set_ylabel('Thực tế')
    plt.tight_layout()
    plt.show()

# Gọi hàm sau khi đã có đủ 3 kết quả dự đoán
plot_confusion_matrices(y_test, 
                        [svm_predictions], 
                        ['SVM'])

def plot_decision_boundaries(X_train, y_train, names):
    # Chỉ lấy 2 chiều đầu tiên để vẽ 2D
    X_reduced = X_train[:, :2]
    
    # Tạo lưới (grid) để dự đoán màu nền
    models_mini = [
        MyRandomForest(n_trees=10, max_depth=5, n_features=2), # n_features phải <= 2
        LogisticRegression(learning_rate=0.1, epochs=500),
        SVM(learning_rate=0.01, epochs=500)
    ]
    h = .02  # độ phân giải của lưới
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (model, name) in enumerate(zip(models_mini, names)):
        # Train lại model với dữ liệu 2D
        # Lưu ý: fit lại trên bản sao để không hỏng model chính
        model.fit(X_reduced, y_train)
        
        # Dự đoán trên toàn bộ lưới
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Vẽ màu nền (vùng quyết định)
        axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        
        # Vẽ các điểm dữ liệu thực tế (chỉ vẽ 200 điểm cho đỡ rối mắt)
        scatter = axes[i].scatter(X_reduced[:200, 0], X_reduced[:200, 1], c=y_train[:200], 
                                  edgecolors='k', cmap='coolwarm', s=30)
        axes[i].set_title(f'Decision Boundary: {name}')
        
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Gọi hàm vẽ
plot_decision_boundaries(X_train, y_train, ['SVM'])