import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog
from sklearn.decomposition import PCA

# Định nghĩa các biến toàn cục (Sửa lại đường dẫn cho khớp máy bạn)
SOURCE_DIR = "PetImages"
IMG_SIZE = (128, 64)

def run_eda():
    print("🚀 ĐANG KHỞI CHẠY PHÂN TÍCH DỮ LIỆU CHUYÊN SÂU (EDA)...\n")
    os.makedirs('eda_results', exist_ok=True)

    # ==========================================
    # 1. TRỰC QUAN HÓA BỘ LỌC HOG (Sự khác biệt giữa Chó và Mèo)
    # ==========================================
    print("1. Trực quan hóa đặc trưng HOG...")
    # Lấy tạm 1 ảnh chó và 1 ảnh mèo (Bạn tự trỏ đường dẫn đúng nhé)
    cat_path = os.path.join(SOURCE_DIR, "Cat", "0.jpg")
    dog_path = os.path.join(SOURCE_DIR, "Dog", "0.jpg")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    for i, path in enumerate([cat_path, dog_path]):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, IMG_SIZE)
        # Chạy HOG và lấy ảnh visualize (Lưu ý cờ visualize=True)
        _, hog_image = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), 
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        
        title = "MÈO" if i == 0 else "CHÓ"
        axes[i, 0].imshow(img_resized, cmap='gray')
        axes[i, 0].set_title(f"Ảnh gốc thu nhỏ - {title}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(hog_image, cmap='hot') # Dùng colormap hot để làm nổi bật đường viền
        axes[i, 1].set_title(f"Đặc trưng HOG - {title}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('eda_results/hog_visualization.png', dpi=300)
    print("-> Đã lưu ảnh HOG vào: eda_results/hog_visualization.png")

    # ==========================================
    # 2. PHÂN TÍCH THÔNG TIN GIỮ LẠI CỦA PCA (Scree Plot)
    # ==========================================
    print("\n2. Phân tích PCA (Scree Plot)...")
    try:
        # Load PCA object đã lưu
        pca = joblib.load('models/pca_model.pkl')
        
        plt.figure(figsize=(8, 5))
        # np.cumsum tính tổng dồn phần trăm phương sai
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='.', linestyle='-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='Ngưỡng 95% thông tin')
        
        plt.title('Biểu đồ Tích lũy Phương sai (PCA Explained Variance)', fontweight='bold')
        plt.xlabel('Số lượng chiều (Components)')
        plt.ylabel('Tỉ lệ thông tin tích lũy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eda_results/pca_scree_plot.png', dpi=300)
        print("-> Đã lưu biểu đồ PCA Scree Plot vào: eda_results/pca_scree_plot.png")
    except FileNotFoundError:
         print("-> [Bỏ qua] Không tìm thấy model PCA tại models/pca_model.pkl")

    # ==========================================
    # 3. PHÂN BỐ KHÔNG GIAN DỮ LIỆU (PCA 2D Scatter Plot)
    # ==========================================
    print("\n3. Vẽ ranh giới phân loại trong không gian 2D...")
    try:
        # Vẽ biểu đồ Scatter Plot 2 chiều đầu tiên để xem chó/mèo tách biệt không
        X_pca = np.load('feature_train_pca.npy')
        y = np.load('label_train.npy')
        
        plt.figure(figsize=(10, 8))
        
        # Vẽ class 0 (Mèo)
        plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', alpha=0.5, label='Mèo (0)', marker='o', s=10)
        # Vẽ class 1 (Chó)
        plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', alpha=0.5, label='Chó (1)', marker='x', s=10)

        plt.title('Phân bố không gian của 2 đặc trưng PCA lớn nhất', fontweight='bold')
        plt.xlabel('Thành phần chính số 1 (PC1)')
        plt.ylabel('Thành phần chính số 2 (PC2)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('eda_results/pca_scatter_2d.png', dpi=300)
        print("-> Đã lưu biểu đồ phân bố không gian vào: eda_results/pca_scatter_2d.png")
    except FileNotFoundError:
         print("-> [Bỏ qua] Không tìm thấy file dữ liệu PCA.")

    print("\n✅ HOÀN TẤT PHÂN TÍCH CHUYÊN SÂU!")

if __name__ == "__main__":
    run_eda()