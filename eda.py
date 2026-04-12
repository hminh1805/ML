import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog
from sklearn.decomposition import PCA
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
import pandas as pd
import glob



# Định nghĩa các biến toàn cục (Sửa lại đường dẫn cho khớp máy bạn)
SOURCE_DIR = "PetImages"
IMG_SIZE = (128, 128)

# Cài đặt font chữ và style chung cho đẹp
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class EDA:
    def plot_class_balance(self,y_train, y_valid, y_test):
        # Đếm số lượng nhãn 0 (Mèo) và 1 (Chó) trong từng tập
        datasets = {'Train': y_train, 'Valid': y_valid, 'Test': y_test}
        
        data = []
        for name, y in datasets.items():
            meo_count = np.sum(y == 0)
            cho_count = np.sum(y == 1)
            data.extend([[name, 'Mèo (0)', meo_count], [name, 'Chó (1)', cho_count]])
            
        df = pd.DataFrame(data, columns=['Tập dữ liệu', 'Class', 'Số lượng'])
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='Tập dữ liệu', y='Số lượng', hue='Class', palette=['blue', 'red'])
        plt.title('Phân bố số lượng Chó/Mèo trong các tập dữ liệu')
        plt.savefig('eda_results/class.png', dpi=300)
        plt.show()

    def plot_image_dimensions(self,source_dir, sample_size=1000):
        widths, heights = [], []
        
        # Lấy ngẫu nhiên một lượng ảnh để vẽ cho nhanh
        all_images = glob.glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)
        sample_imgs = np.random.choice(all_images, min(sample_size, len(all_images)), replace=False)
        
        for img_path in sample_imgs:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except:
                continue

        plt.figure(figsize=(10, 6))
        plt.scatter(widths, heights, alpha=0.3, color='purple', s=10)
        plt.title('Phân bố Kích thước ảnh gốc (Width vs Height)')
        plt.xlabel('Chiều rộng (Pixels)')
        plt.ylabel('Chiều cao (Pixels)')
        
        # Vẽ đường chéo (Tỷ lệ 1:1 - ảnh vuông)
        max_val = max(max(widths), max(heights))
        plt.plot([0, max_val], [0, max_val], 'k--', label='Tỷ lệ 1:1 (Vuông)')
        plt.legend()
        plt.savefig('eda_results/img_dim.png', dpi=300)
        plt.show()
        
    def plot_pixel_intensity(self,source_dir, sample_size=500):
        cat_pixels, dog_pixels = [], []
        
        cats = glob.glob(os.path.join(source_dir, 'Cat', '*.jpg'))[:sample_size]
        dogs = glob.glob(os.path.join(source_dir, 'Dog', '*.jpg'))[:sample_size]
        
        for path, pixel_list in zip([cats, dogs], [cat_pixels, dog_pixels]):
            for img_path in path:
                try:
                    # Mở ảnh, chuyển sang Grayscale (L), resize nhỏ lại tính cho lẹ
                    img = Image.open(img_path).convert('L').resize((64, 64))
                    pixel_list.extend(np.array(img).flatten())
                except:
                    continue
                    
        plt.figure(figsize=(10, 6))
        plt.hist(cat_pixels, bins=50, alpha=0.5, color='blue', label='Mèo', density=True)
        plt.hist(dog_pixels, bins=50, alpha=0.5, color='red', label='Chó', density=True)
        plt.title('Phân bố Cường độ sáng (Pixel Intensity) của Ảnh gốc')
        plt.xlabel('Giá trị Pixel (0: Đen -> 255: Trắng)')
        plt.ylabel('Tần suất')
        plt.legend()
        plt.savefig('eda_results/pca_pixel.png', dpi=300)
        plt.show()


    def plot_mean_image(self,source_dir, img_size=(100, 100), sample_size=1000):
        def get_mean_img(folder):
            paths = glob.glob(os.path.join(source_dir, folder, '*.jpg'))[:sample_size]
            imgs = []
            for p in paths:
                try:
                    img = np.array(Image.open(p).convert('L').resize(img_size))
                    imgs.append(img)
                except: pass
            return np.mean(imgs, axis=0)

        mean_cat = get_mean_img('Cat')
        mean_dog = get_mean_img('Dog')

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mean_cat, cmap='gray')
        axes[0].set_title('Ảnh Mèo Trung Bình')
        axes[0].axis('off')
        
        axes[1].imshow(mean_dog, cmap='gray')
        axes[1].set_title('Ảnh Chó Trung Bình')
        axes[1].axis('off')
        plt.savefig('eda_results/pca_mean_img.png', dpi=300)
        plt.show()
        
    def plot_pca_correlation(self,X_pca, num_components=15):
        # Chỉ lấy top components đầu tiên để vẽ cho rõ
        X_subset = X_pca[:, :num_components]
        corr_matrix = np.corrcoef(X_subset, rowvar=False)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    annot=False, xticklabels=[f'PC{i+1}' for i in range(num_components)], 
                    yticklabels=[f'PC{i+1}' for i in range(num_components)])
        plt.title(f'Ma trận Tương quan của {num_components} Đặc trưng PCA lớn nhất')
        plt.savefig('eda_results/pca_corr.png', dpi=300)
        plt.show()

    def plot_pca_pairplot(self,X_pca, y_train, num_components=4):
        # Lấy top N thành phần và gán nhãn
        df = pd.DataFrame(X_pca[:, :num_components], columns=[f'PC{i+1}' for i in range(num_components)])
        df['Nhãn'] = ['Chó' if label == 1 else 'Mèo' for label in y_train]
        
        sns.pairplot(df, hue='Nhãn', palette={'Mèo': 'blue', 'Chó': 'red'}, 
                    plot_kws={'alpha':0.4, 's':10})
        plt.suptitle('Phân tán Không gian của Top Đặc trưng PCA', y=1.02)
        plt.savefig('eda_results/pac_pair.png', dpi=300)
        plt.show()

    def plot_custom_feature_importance(self,forest_model, top_n=20):
        feature_counts = {}
        
        # Hàm đệ quy đi dọc các cành cây để đếm
        def traverse_and_count(node):
            if node.is_leaf_node():
                return
            # Nếu là node cành, tăng biến đếm cho đặc trưng đó
            feat = node.feature_idx
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
            traverse_and_count(node.left)
            traverse_and_count(node.right)
            
        # Duyệt qua toàn bộ rừng
        for tree in forest_model.trees:
            traverse_and_count(tree.root)
            
        # Sắp xếp và vẽ
        sorted_feats = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f'PC {f[0]}' for f in sorted_feats]
        counts = [f[1] for f in sorted_feats]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=counts, y=features, palette='viridis')
        plt.title(f'Top {top_n} Đặc trưng Được chia nhánh nhiều nhất trong Rừng')
        plt.xlabel('Số lần được chọn làm nhát cắt')
        plt.ylabel('Đặc trưng (Cột PCA)')
        plt.savefig('eda_results/feat_importance.png', dpi=300)
        plt.show()
        
    def plot_edge_cases(self,model, X_valid, y_valid, valid_image_paths, num_images=5):
        # Dự đoán xác suất (Cần model có hàm predict_proba hoặc decision_function)
        # Giả sử mô hình của ông có predict_proba trả về xác suất class 1 (Chó)
        try:
            probs = model.predict_proba(X_valid)[:, 1] 
        except:
            print("Mô hình không hỗ trợ xuất xác suất!")
            return

        # Độ bối rối = Khoảng cách từ xác suất đến 0.5 (Càng nhỏ càng lú)
        confusion_scores = np.abs(probs - 0.5)
        
        # Lấy ra index của top những ảnh lú nhất
        most_confused_idx = np.argsort(confusion_scores)[:num_images]
        
        fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
        for i, idx in enumerate(most_confused_idx):
            img_path = valid_image_paths[idx] # Chỗ này ông phải lưu lại mảng chứa đường dẫn ảnh tập Valid nhé
            true_label = "Chó" if y_valid[idx] == 1 else "Mèo"
            pred_prob = probs[idx]
            
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'Thực tế: {true_label}\nXác suất đoán Chó: {pred_prob:.2f}', 
                                color='red' if np.round(pred_prob) != y_valid[idx] else 'green')
            except: pass
        plt.suptitle('Top các ảnh mô hình "lú" nhất (Xác suất sát ranh giới 50/50)')
        plt.savefig('eda_results/edge_case.png', dpi=300)
        plt.show()
        
    def plot_tree_depth_distribution(self,forest_model):
        depths = []
        
        # Hàm đệ quy tìm đáy của từng cây
        def get_max_depth(node, current_depth=0):
            if node.is_leaf_node():
                return current_depth
            left_depth = get_max_depth(node.left, current_depth + 1)
            right_depth = get_max_depth(node.right, current_depth + 1)
            return max(left_depth, right_depth)

        for tree in forest_model.trees:
            depths.append(get_max_depth(tree.root))
            
        plt.figure(figsize=(8, 5))
        sns.countplot(x=depths, palette='magma')
        plt.title('Phân bố Chiều cao thực tế (Depth) của quần thể cây')
        plt.xlabel('Chiều cao (Tầng)')
        plt.ylabel('Số lượng cây')
        
        # Vẽ đường trung bình
        mean_depth = np.mean(depths)
        plt.axvline(mean_depth - min(depths), color='r', linestyle='dashed', linewidth=2, label=f'Trung bình: {mean_depth:.1f}')
        plt.legend()
        plt.savefig('eda_results/tree_depth.png', dpi=300)
        plt.show()

# Cách gọi (Sau khi đã forest.fit xong):
# plot_tree_depth_distribution(forest)

# Cách gọi: Ông cần truyền thêm mảng chứa danh sách đường dẫn ảnh của tập validation vào
# plot_edge_cases(logistic_model, X_valid, y_valid, list_valid_paths)

# Cách gọi (Sau khi đã forest.fit xong): 
# plot_custom_feature_importance(forest)

# Cách gọi: plot_tsne(X_train_pca, y_train)
# Cách gọi: plot_pca_pairplot(X_train_pca, y_train)
# Cách gọi: plot_pca_correlation(X_train_pca)

# Cách gọi: plot_mean_image('PetImages')
# Cách gọi: plot_pixel_intensity('PetImages')

# Cách gọi: plot_image_dimensions('PetImages')
# Cách gọi: plot_class_balance(y_train, y_valid, y_test)



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