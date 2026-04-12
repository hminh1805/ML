import numpy as np
import os
from myModel.randomforest import MyRandomForest  # Đổi tên file import cho đúng với file của ông nha
from myModel.logisticRegression import LogisticRegression  # Đổi tên file import cho đúng với file của ông nha
from myModel.SVM import SVM  # Đổi tên file import cho đúng với file của ông nha
from make_minidataset import generate_dataset_indices
from extract_feature import run_hog
from pca import run_pca
from evaluate import evaluate_and_compare


import time
from eda import EDA
import joblib





def run_sklearn_benchmark():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print("="*60)
    print(" 🚀 BẮT ĐẦU ĐỐI CHUẨN VỚI THƯ VIỆN SCIKIT-LEARN")
    print("="*60)
    X_train, y_train,X_valid,y_valid, X_test, y_test = load_data('full')
    # Khởi tạo 3 mô hình Sklearn (Cấu hình ép cho giống với mô hình sếp tự code nhất)
    models = {
        "Sklearn Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        
        # Dùng kernel='linear' để công bằng với con SVM thủ công của sếp
        "Sklearn SVM (Linear)": SVC(kernel='linear', random_state=42), 
        
        # Nạp đúng thông số Trùm Cuối mà sếp đã tìm ra
        "Sklearn Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            min_samples_split=4, 
            n_jobs=-1, # Dùng full lõi CPU
            random_state=42
        )
    }

    # In tiêu đề bảng
    print(f"{'Mô hình (Sklearn)':<28} | {'Accuracy':<8} | {'F1-Score':<8} | {'Train (s)':<9} | {'Đoán (s)':<8}")
    print("-" * 70)

    for name, model in models.items():
        # --- ĐO THỜI GIAN TRAIN ---
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # --- ĐO THỜI GIAN TEST ---
        start_test = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - start_test
        
        # --- TÍNH ĐIỂM ---
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # In kết quả
        print(f"{name:<28} | {acc:.4f}   | {f1:.4f}   | {train_time:<9.4f} | {test_time:<8.4f}")

    print("="*70)

def load_data(mode='mini'):
    """Hàm phụ trợ để tải dữ liệu Numpy"""
    print("⏳ Đang tải dữ liệu đã qua xử lý (HOG + PCA)...")
    # generate_dataset_indices(source_dir='PetImages', mode=mode,seed=43)
    # run_hog(mode) 
    # run_pca(0.95)

    try:
        X_train = np.load('feature_train_pca.npy')
        y_train = np.load('label_train.npy')
        X_test = np.load('feature_test_pca.npy')
        y_test = np.load('label_test.npy')    
        
        X_valid = np.load('feature_valid_pca.npy')
        y_valid = np.load('label_valid.npy')
        
        
        print(f"✅ Tải dữ liệu thành công!")
        print(f"   - Tập Train: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng.")
        print(f"   - Tập Valid: {X_valid.shape[0]} mẫu, {X_valid.shape[1]} đặc trưng.")
        print(f"   - Tập Test:  {X_test.shape[0]} mẫu, {X_test.shape[1]} đặc trưng.\n")
        return X_train, y_train,X_valid, y_valid, X_test, y_test
        
    except FileNotFoundError as e:
        print(f" LỖI: Không tìm thấy file dữ liệu ({e.filename})")
        print(" HƯỚNG DẪN: Hãy chắc chắn bạn đã chạy tuần tự các file:")
        print("   1. python extract_features.py")
        print("   2. python pca.py")
        return None, None, None, None,None,None


def eda(y_train, y_valid, y_test):
    # import matplotlib.pyplot as plt

    # plt.hist(X_train[:, 1], bins=50)
    # plt.show()
    e = EDA()
    e.plot_class_balance(y_train, y_valid, y_test)
    e.plot_image_dimensions('PetImages', sample_size=24000)
    e.plot_pixel_intensity('PetImages', sample_size=12000)
    # e.plot_mean_image('PetImages', img_size=(100, 100), sample_size=1000)
    # e.plot_pca_correlation(X_train, num_components=15)
    # e.plot_pca_pairplot(X_train, y_train, num_components=4)
    
def main():
    print("="*60)
    print("CHƯƠNG TRÌNH PHÂN LOẠI CHÓ MÈO BẰNG MACHINE LEARNING".center(60))
    print("="*60 + "\n")

    # 1. Tải dữ liệu
    X_train, y_train,X_valid,y_valid, X_test, y_test = load_data('full')
    
    # Nếu tải dữ liệu thất bại thì dừng chương trình
    if X_train is None:
        return

    # 2. Khởi tạo danh sách (Từ điển) các mô hình cần đánh giá
    print(" Đang khởi tạo các mô hình thuật toán...")
    n_features = int(np.sqrt(X_train.shape[1]))  # Số lượng đặc trưng sau PCA
    models_dict = {
        "Logistic Regression": LogisticRegression( learning_rate=0.1, epochs=1000),
        "Support Vector Machine": SVM(learning_rate=0.1, epochs=1000),
        "Random Forest": MyRandomForest(n_trees=100, max_depth=10, min_samples_split= 5, n_features=n_features)

    }

    # 3. Chạy hàm đánh giá tổng hợp từ evaluate_models.py
    # Hàm này sẽ tự in ra bảng Pandas trên Terminal
    df_report = evaluate_and_compare(models_dict, X_train, y_train, X_test, y_test)

    # 4. Xuất kết quả ra file để copy vào báo cáo
    os.makedirs('reports', exist_ok=True) # Tạo folder reports nếu chưa có
    report_path = "reports/model_comparison_report.csv"
    
    df_report.to_csv(report_path, index=False)
    
    print(f"\n Đã tự động xuất file báo cáo chi tiết tại: {report_path}")
    
    print("\n ĐANG ĐÓNG GÓI VÀ LƯU MÔ HÌNH...")
    os.makedirs('models', exist_ok=True) # Tạo thư mục chứa model cho gọn
    
    for name, model in models_dict.items():
        # Sửa tên xíu cho nó không bị khoảng trắng (VD: Random Forest -> Random_Forest)
        safe_name = name.replace(" ", "_")
        file_path = f"models/{safe_name}.pkl"
        
        joblib.dump(model, file_path)
        print(f"  + Đã lưu {name} tại: {file_path}")
        
        
    print("\n" + "="*60)
    print(" HOÀN TẤT CHƯƠNG TRÌNH ".center(60))
    print("="*60)




    
    
    
if __name__ == "__main__":
    #main_old()
    #main()
    run_sklearn_benchmark()