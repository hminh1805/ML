import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn tới thư mục minidataset mà ông vừa tạo
DATASET_DIR = "minidataset"

def run_pt():
    print(" ĐANG KHỞI CHẠY PHÂN TÍCH DỮ LIỆU (pt)...\n")
    
    # ==========================================
    # 1. PHÂN PHỐI NHÃN (CLASS DISTRIBUTION)
    # ==========================================
    train_cats = len(os.listdir(os.path.join(DATASET_DIR, "train", "cats")))
    train_dogs = len(os.listdir(os.path.join(DATASET_DIR, "train", "dogs")))
    test_cats = len(os.listdir(os.path.join(DATASET_DIR, "test", "cats")))
    test_dogs = len(os.listdir(os.path.join(DATASET_DIR, "test", "dogs")))
    
    print("1. THỐNG KÊ SỐ LƯỢNG:")
    print(f"- Tập Train: {train_cats} Mèo | {train_dogs} Chó")
    print(f"- Tập Test : {test_cats} Mèo | {test_dogs} Chó")
    
    # Vẽ biểu đồ cột phân phối nhãn
    labels = ['Train - Cats', 'Train - Dogs', 'Test - Cats', 'Test - Dogs']
    counts = [train_cats, train_dogs, test_cats, test_dogs]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts, color=['#ff9999','#66b3ff','#ff9999','#66b3ff'])
    plt.title('Phân phối dữ liệu (Class Distribution)', fontsize=14, fontweight='bold')
    plt.ylabel('Số lượng ảnh')
    
    # Thêm con số trên đầu cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('pt_distribution.png', dpi=300) # Lưu luôn ra file ảnh
    print("-> Đã lưu biểu đồ phân phối vào file 'pt_distribution.png'")
    
    # ==========================================
    # 2. KHẢO SÁT KÍCH THƯỚC VÀ KÊNH MÀU
    # ==========================================
    print("\n2. KHẢO SÁT KÍCH THƯỚC ẢNH (Đang quét tập Train...):")
    train_cat_dir = os.path.join(DATASET_DIR, "train", "cats")
    sample_images = os.listdir(train_cat_dir)[:500] # Lấy thử 500 tấm để đo cho lẹ
    
    heights = []
    widths = []
    
    for img_name in sample_images:
        img_path = os.path.join(train_cat_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w, c = img.shape
            heights.append(h)
            widths.append(w)
    
    print(f"- Số kênh màu: {c} kênh (Ảnh RGB)")
    print(f"- Chiều cao trung bình: {int(np.mean(heights))}px (Dao động từ {np.min(heights)}px đến {np.max(heights)}px)")
    print(f"- Chiều rộng trung bình: {int(np.mean(widths))}px (Dao động từ {np.min(widths)}px đến {np.max(widths)}px)")
    print("=> KẾT LUẬN: Kích thước ảnh rất lộn xộn, BẮT BUỘC phải Resize và chuyển ảnh xám trước khi đưa vào mô hình.")

    # Vẽ biểu đồ phân tán (Scatter Plot) kích thước ảnh
    plt.figure(figsize=(8, 5))
    plt.scatter(widths, heights, alpha=0.5, color='purple')
    plt.title('Phân bố kích thước ảnh thực tế (Width vs Height)', fontsize=14, fontweight='bold')
    plt.xlabel('Chiều rộng (Pixels)')
    plt.ylabel('Chiều cao (Pixels)')
    plt.tight_layout()
    plt.savefig('pt_dimensions.png', dpi=300)
    print("-> Đã lưu biểu đồ kích thước vào file 'pt_dimensions.png'")

    # ==========================================
    # 3. TRỰC QUAN HÓA (VISUALIZATION)
    # ==========================================
    print("\n3. XUẤT ẢNH TRỰC QUAN...")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6)) # Vẽ lưới 2 hàng 3 cột
    
    # Lấy đại 3 ảnh chó, 3 ảnh mèo
    cat_samples = [os.path.join(DATASET_DIR, "train", "cats", f) for f in os.listdir(os.path.join(DATASET_DIR, "train", "cats"))[:3]]
    dog_samples = [os.path.join(DATASET_DIR, "train", "dogs", f) for f in os.listdir(os.path.join(DATASET_DIR, "train", "dogs"))[:3]]
    all_samples = cat_samples + dog_samples
    
    for i, ax in enumerate(axes.flat):
        img = cv2.imread(all_samples[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV đọc BGR, phải chuyển qua RGB để hiện đúng màu
        ax.imshow(img_rgb)
        label = "CAT" if i < 3 else "DOG"
        ax.set_title(f"Label: {label}", fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('pt_samples.png', dpi=300)
    print("-> Đã lưu ảnh mẫu vào file 'pt_samples.png'\n")
    print("HOÀN TẤT PHÂN TÍCH!")

if __name__ == "__main__":
    run_pt()