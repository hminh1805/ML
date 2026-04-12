import numpy as np
import os
from myModel.randomforest import MyRandomForest  # Đổi tên file import cho đúng với file của ông nha
from myModel.logisticRegression import LogisticRegression  # Đổi tên file import cho đúng với file của ông nha
from myModel.SVM import SVM  # Đổi tên file import cho đúng với file của ông nha
from make_minidataset import generate_dataset_indices
from extract_feature import run_hog
from pca import run_pca
# Import hàm đánh giá từ file bạn vừa tạo
from evaluate import evaluate_and_compare

def load_data():
    """Hàm phụ trợ để tải dữ liệu Numpy"""
    print("⏳ Đang tải dữ liệu đã qua xử lý (HOG + PCA)...")
    generate_dataset_indices(source_dir='PetImages', mode='mini')
    run_hog()
    run_pca()

    try:
        X_train = np.load('feature_train_pca.npy')
        y_train = np.load('label_train.npy')
        X_test = np.load('feature_test_pca.npy')
        y_test = np.load('label_test.npy')
        
        print(f"✅ Tải dữ liệu thành công!")
        print(f"   - Tập Train: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng.")
        print(f"   - Tập Test:  {X_test.shape[0]} mẫu, {X_test.shape[1]} đặc trưng.\n")
        return X_train, y_train, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy file dữ liệu ({e.filename})")
        print("👉 HƯỚNG DẪN: Hãy chắc chắn bạn đã chạy tuần tự các file:")
        print("   1. python extract_features.py")
        print("   2. python pca.py")
        return None, None, None, None

def main():
    print("="*60)
    print("CHƯƠNG TRÌNH PHÂN LOẠI CHÓ MÈO BẰNG MACHINE LEARNING".center(60))
    print("="*60 + "\n")

    # 1. Tải dữ liệu
    X_train, y_train, X_test, y_test = load_data()
    
    # Nếu tải dữ liệu thất bại thì dừng chương trình
    if X_train is None:
        return

    # 2. Khởi tạo danh sách (Từ điển) các mô hình cần đánh giá
    print("⚙️ Đang khởi tạo các mô hình thuật toán...")
    models_dict = {
        "Logistic Regression": LogisticRegression( learning_rate=0.1, epochs=1000),
        "Support Vector Machine": SVM(learning_rate=0.1, epochs=1000),
        # "Random Forest": # MyRandomForest(n_trees=20, max_depth=7, min_samples_split= 5, n_features=n_features)

    }

    # 3. Chạy hàm đánh giá tổng hợp từ evaluate_models.py
    # Hàm này sẽ tự in ra bảng Pandas trên Terminal
    df_report = evaluate_and_compare(models_dict, X_train, y_train, X_test, y_test)

    # 4. Xuất kết quả ra file để copy vào báo cáo
    # os.makedirs('reports', exist_ok=True) # Tạo folder reports nếu chưa có
    # report_path = "reports/model_comparison_report.csv"
    
    # df_report.to_csv(report_path, index=False)
    
    # print(f"\n💾 Đã tự động xuất file báo cáo chi tiết tại: {report_path}")
    print("\n" + "="*60)
    print("🎉 HOÀN TẤT CHƯƠNG TRÌNH 🎉".center(60))
    print("="*60)

if __name__ == "__main__":
    main()
