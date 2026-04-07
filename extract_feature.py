import os
import cv2
import numpy as np
import json
from skimage.feature import hog

#
SOURCE_DIR = "PetImages" 
INDEX_FILE = "data_splits/mini_dataset.json"
IMG_SIZE = (128,64)  # Kích thước ảnh sau khi resize

def get_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    img_resized = cv2.resize(img, IMG_SIZE)
    
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return hog_features

def extract_features(set_name,data_list):
    features = []
    labels = []
    classes = ['cats', 'dogs']
    
    print(f"Đang trích xuất HOG cho tập '{set_name}' ({len(data_list)} ảnh)...")
    
    for rel_path, label in data_list:
        # Nối đường dẫn gốc với phần đường dẫn tương đối lưu trong JSON
        # Ví dụ: "PetImages" + "Cat/1.jpg"
        full_image_path = os.path.join(SOURCE_DIR, rel_path)
        
        hog_feature = get_hog_features(full_image_path)
        if hog_feature is not None:
            features.append(hog_feature)
            labels.append(label)
            
    return np.array(features), np.array(labels)


def run_hog():
    print("🚀 ĐANG KHỞI CHẠY HOG (hog)...\n")
    
    # 1. Đọc "bản đồ" index
    if not os.path.exists(INDEX_FILE):
        print(f"Lỗi: Không tìm thấy file {INDEX_FILE}. Hãy chạy file make_dataset trước!")
        return

    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)
    
    # 2. Xử lý tập Train
    feature_train_hog, label_train_hog = extract_features("train", index_data['train'])
    
    # 3. Xử lý tập Valid (Đã bổ sung)
    feature_valid_hog, label_valid_hog = extract_features("valid", index_data['valid'])
    
    # 4. Xử lý tập Test
    feature_test_hog, label_test_hog = extract_features("test", index_data['test'])

    print("\nKẾT QUẢ SAU KHI CHẠY HOG:")
    print(f"- Số lượng ảnh Train: {feature_train_hog.shape[0]} tấm")
    print(f"- Số lượng ảnh Valid: {feature_valid_hog.shape[0]} tấm")
    print(f"- Số lượng ảnh Test:  {feature_test_hog.shape[0]} tấm")
    print(f"- Chiều dài vector HOG của MỖI tấm ảnh: {feature_train_hog.shape[1]} con số")

    # --- LƯU LẠI DỮ LIỆU THÔ ĐỂ DÀNH ---
    np.save('feature_train_hog.npy', feature_train_hog)
    np.save('feature_valid_hog.npy', feature_valid_hog) # Lưu valid
    np.save('feature_test_hog.npy', feature_test_hog)
    
    np.save('label_train.npy', label_train_hog)
    np.save('label_valid.npy', label_valid_hog) # Lưu valid
    np.save('label_test.npy', label_test_hog)

    print("\n💾 Đã lưu xong các file .npy gốc (Chưa qua PCA)!")

if __name__ == "__main__":
    run_hog()
