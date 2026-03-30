import os
from sklearn.decomposition import PCA
import numpy as np

def run_pca():
    print(" ĐANG KHỞI CHẠY PCA (pca)...\n")
    #load dữ liệu HOG đã lưu
    feature_train_hog = np.load('feature_train_hog.npy')
    
    feature_test_hog = np.load('feature_test_hog.npy')
    

    print(f"Dữ liệu gốc: {feature_train_hog.shape[1]} chiều.")

    #khởi tạo PCA với 95% thông tin
    pca = PCA(n_components=0.95, random_state= 42)

    #nén dữ liệu Train
    feature_train_pca = pca.fit_transform(feature_train_hog)
    #nén dữ liệu Test
    feature_test_pca = pca.transform(feature_test_hog)

    print(f"Dữ liệu sau khi PCA: {feature_train_pca.shape[1]} chiều.")

    np.save('feature_train_pca.npy', feature_train_pca)
    np.save('feature_test_pca.npy', feature_test_pca)

    print("\nĐã lưu xong các file .npy sau khi PCA!")