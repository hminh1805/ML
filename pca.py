import os
from sklearn.decomposition import PCA
import joblib # 
import numpy as np

def run_pca(n_components=0.95):
    print(" ĐANG KHỞI CHẠY PCA ...\n")
    
    try:
        feature_train_hog = np.load('feature_train_hog.npy')
        feature_valid_hog = np.load('feature_valid_hog.npy')
        feature_test_hog = np.load('feature_test_hog.npy')
    except FileNotFoundError:
        print("Không tìm thấy file .npy! Hãy chạy extract_feature.py trước để tạo các file này.")
        return

    print(f"Dữ liệu gốc: {feature_train_hog.shape[1]} chiều.")

    #khởi tạo PCA với n thông tin
    pca = PCA(n_components, random_state= 42)

    #nén dữ liệu Train
    feature_train_pca = pca.fit_transform(feature_train_hog)
    #nén dữ liệu Valid và Test
    feature_valid_pca = pca.transform(feature_valid_hog)
    feature_test_pca = pca.transform(feature_test_hog)

    print(f"Dữ liệu sau khi PCA: {feature_train_pca.shape[1]} chiều.")

    np.save('feature_train_pca.npy', feature_train_pca)
    np.save('feature_valid_pca.npy', feature_valid_pca)
    np.save('feature_test_pca.npy', feature_test_pca)

    print("\nĐã lưu xong các file .npy sau khi PCA!")
    
    
    #LƯU LẠI MODEL PCA ĐỂ DÙNG LÚC INFERENCE/DEMO
    os.makedirs('models', exist_ok=True) # Tạo folder models nếu chưa có
    joblib.dump(pca, 'models/pca_model.pkl')
    print("Đã lưu model PCA tại: models/pca_model.pkl")