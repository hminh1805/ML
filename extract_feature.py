import os
import cv2
import numpy as np
from skimage.feature import hog

#
DATA_DIR = "minidataset"
IMG_SIZE = (68,64)  # Kích thước ảnh sau khi resize

def get_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    img_resized = cv2.resize(img, IMG_SIZE)
    
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return hog_features

def extract_features(set_name):
    features = []
    labels = []
    classes = ['cats', 'dogs']
    for label in classes:
        class_dir = os.path.join(DATA_DIR, set_name, label)
        for img_name in os.listdir(class_dir):
            img_path_ = os.path.join(class_dir, img_name)
            hog_feature = get_hog_features(img_path_)
            if hog_feature is not None:
                features.append(hog_feature)
                labels.append(0 if label == 'cats' else 1)  # 0 cho mèo, 1 cho chó
    return np.array(features), np.array(labels)



def run_hog():
    print(" ĐANG KHỞI CHẠY HOG (hog)...\n")
    
    # 1. Xử lý tập Train
    feature_train_hog, label_train_hog = extract_features("train")
    # 2. Xử lý tập Test
    feature_test_hog, label_test_hog = extract_features("test")

    print("\nKẾT QUẢ SAU KHI CHẠY HOG:")
    print(f"- Số lượng ảnh Train: {feature_train_hog.shape[0]} tấm")
    print(f"- Chiều dài vector HOG của MỖI tấm ảnh: {feature_train_hog.shape[1]} con số")

    # --- LƯU LẠI DỮ LIỆU THÔ ĐỂ DÀNH ---
    np.save('feature_train_hog.npy', feature_train_hog)
    np.save('feature_test_hog.npy', feature_test_hog)
    np.save('label_train.npy', label_train_hog)
    np.save('label_test.npy', label_test_hog)

    print("\n💾 Đã lưu xong các file .npy gốc (Chưa qua PCA)!")

if __name__ == "__main__":
    run_hog()