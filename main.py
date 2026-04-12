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

import time
import numpy as np
from myModel.randomforest import MyRandomForest  # Đổi tên file import cho đúng với file của ông nha
from myModel.logisticRegression import LogisticRegression  # Đổi tên file import cho đúng với file của ông nha
from myModel.SVM import SVM  # Đổi tên file import cho đúng với file của ông nha
from make_minidataset import generate_dataset_indices
from extract_feature import run_hog
from pca import run_pca
from eda import EDA
from multiprocessing import shared_memory
import joblib


if __name__ == "__main__":

    #generate_dataset_indices(source_dir='PetImages', mode='mini',seed=42)

    #run_hog()
    #run_pca(0.95)

    print("🚀 ĐANG LOAD DỮ LIỆU HOG-PCA THẬT...")
    try:
        X_train = np.load('feature_train_pca.npy')
        X_train_hog = np.load('feature_train_hog.npy')
        y_train = np.load('label_train.npy')
        X_valid = np.load('feature_valid_pca.npy')
        X_valid_hog = np.load('feature_valid_hog.npy')
        y_valid = np.load('label_valid.npy')
        X_test = np.load('feature_test_pca.npy')
        y_test = np.load('label_test.npy')
        print(f"Đã load xong! Kích thước Train gốc: {X_train.shape}")
    except FileNotFoundError:
        print("Không tìm thấy file .npy!")
        exit()

    # # # =========================================================
    # # # RANDOM FOREST
    # # # =========================================================
    # print(f"\nĐang lấy {len(y_train)} ảnh để dạy rừng...")

    # # Tính số lượng cột cho mỗi cây (Căn bậc 2 của tổng số cột PCA)
    # n_features = int(np.sqrt(X_train.shape[1]))

    # # Khởi tạo rừng: 20 cây, mỗi cây cao tối đa 10 tầng
    # forest = MyRandomForest(n_trees=100, max_depth=10, min_samples_split=5, n_features=n_features)

    # start_time = time.time()
    # forest.fit(X_train, y_train)
    # print(f"⏳ Thời gian trồng rừng: {time.time() - start_time:.2f} giây")
    
    # print("\n🔍 Đang đem Khu rừng đi dự đoán tập test...")
    # f_predictions = forest.predict(X_test)

    # # Đánh giá
    # f_accuracy = np.sum(f_predictions == y_test) / len(y_test) * 100
    # print(f"🎉 Độ chính xác trên tập thực tế thu nhỏ: {f_accuracy:.2f}%")
    
    # print("💾 Đang xuất file mô hình Random Forest...")
    # joblib.dump(forest, 'rf_trum_cuoi.pkl')
    # print("✅ Đã lưu thành công file 'rf_trum_cuoi.pkl'!")

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    # print("\n Huấn luyện mô hình LOGISTIC REGRESSION...")

    # logistic = LogisticRegression( learning_rate=0.1, epochs=1000)

    # start_time = time.time()

    # logistic.fit(X_train, y_train)
    # print(f"⏳ Thời gian train Logistic Regression: {time.time() - start_time:.2f} giây")

    # lr_predictions = logistic.predict(X_valid)
    # lr_accucary = np.mean(lr_predictions == y_valid) * 100
    # print(f"🎯 Logistic Regression Accuracy: {lr_accucary:.2f}%")

    # =========================================================
    # SVM
    # =========================================================
    # print("\n Huấn luyện mô hình SVM...")

    # svm = SVM( learning_rate=0.001, epochs=1000)

    # start_time = time.time()
    # svm.fit(X_train, y_train)
    # print(f"⏳ Thời gian train SVM: {time.time() - start_time:.2f} giây")

    # svm_predictions = svm.predict(X_valid)
    # svm_accuracy = svm.score(X_valid, y_valid) * 100
    # # svm_accucary = np.mean(svm_predictions == y_valid) * 100
    # print(f"🎯 SVM Accuracy: {svm_accuracy:.2f}%")

    # =========================================================
    # So sánh
    # =========================================================
    # print("\n📊 SO SÁNH KẾT QUẢ")
    # print(f"Random Forest      : {f_accuracy:.2f}%")
    # print(f"Logistic Regression: {lr_accucary:.2f}%")
    # print(f"SVM                : {svm_accuracy:.2f}%")



    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from sklearn.metrics import confusion_matrix

    # def plot_confusion_matrices(y_valid, preds_list, names):
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #     for i, (preds, name) in enumerate(zip(preds_list, names)):
    #         cm = confusion_matrix(y_valid, preds)
    #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
    #                     xticklabels=['Mèo (0)', 'Chó (1)'], 
    #                     yticklabels=['Mèo (0)', 'Chó (1)'])
    #         axes[i].set_title(f'Confusion Matrix: {name}')
    #         axes[i].set_xlabel('Dự đoán')
    #         axes[i].set_ylabel('Thực tế')
    #     plt.tight_layout()
    #     plt.show()

    # # Gọi hàm sau khi đã có đủ 3 kết quả dự đoán
    # plot_confusion_matrices(y_valid, 
    #                         [svm_predictions], 
    #                         ['SVM'])

    # def plot_decision_boundaries(X_train, y_train, names):
    #     # Chỉ lấy 2 chiều đầu tiên để vẽ 2D
    #     X_reduced = X_train[:, :2]
        
    #     # Tạo lưới (grid) để dự đoán màu nền
    #     models_mini = [
    #         MyRandomForest(n_trees=10, max_depth=5, n_features=2), # n_features phải <= 2
    #         LogisticRegression(learning_rate=0.1, epochs=500),
    #         SVM(learning_rate=0.01, epochs=500)
    #     ]
    #     h = .02  # độ phân giải của lưới
    #     x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    #     y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
    #     fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    #     for i, (model, name) in enumerate(zip(models_mini, names)):
    #         # Train lại model với dữ liệu 2D
    #         # Lưu ý: fit lại trên bản sao để không hỏng model chính
    #         model.fit(X_reduced, y_train)
            
    #         # Dự đoán trên toàn bộ lưới
    #         Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #         Z = Z.reshape(xx.shape)
            
    #         # Vẽ màu nền (vùng quyết định)
    #         axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            
    #         # Vẽ các điểm dữ liệu thực tế (chỉ vẽ 200 điểm cho đỡ rối mắt)
    #         scatter = axes[i].scatter(X_reduced[:200, 0], X_reduced[:200, 1], c=y_train[:200], 
    #                                   edgecolors='k', cmap='coolwarm', s=30)
    #         axes[i].set_title(f'Decision Boundary: {name}')
            
    #     plt.legend(*scatter.legend_elements(), title="Classes")
    #     plt.show()

    # # Gọi hàm vẽ
    # plot_decision_boundaries(X_train, y_train, ['SVM'])
    
    
    
    # # # =========================================================
    # # # RANDOM FOREST
    # # # =========================================================
    # for t in [20,50,70,100,150]:
    #         print(f"\nĐang lấy {len(y_train)} ảnh để dạy rừng với trees={t} ...")
    #         # Tính số lượng cột cho mỗi cây (Căn bậc 2 của tổng số cột PCA)
    #         n_features = int(np.sqrt(X_train.shape[1]))
    #         # Khởi tạo rừng: 20 cây, mỗi cây cao tối đa 10 tầng
    #         forest = MyRandomForest(n_trees=t, max_depth=10, min_samples_split=6 ,n_features=n_features)
    #         start_time = time.time()
    #         forest.fit(X_train, y_train)
    #         print(f"⏳ Thời gian trồng rừng: {time.time() - start_time:.2f} giây")
    #         f_predictions = forest.predict(X_valid)
    #         f_accuracy = np.sum(f_predictions == y_valid) / len(y_valid) * 100
    #         print(f"🎉 Độ chính xác trên tập thực tế thu nhỏ: {f_accuracy:.2f}%")
    # from sklearn.model_selection import KFold
    # X_cv = np.vstack((X_train, X_valid))
    # y_cv = np.concatenate((y_train, y_valid))
    # print(f"Tổng số ảnh đem đi thi K-Fold: {len(y_cv)} ảnh")

    # # Tính số cột để truyền vào hàm
    # n_features_cv = int(np.sqrt(X_cv.shape[1]))

    # # 3. SETUP K-FOLD (5 phần, CÓ XÁO TRỘN DỮ LIỆU)
    # kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # # 4. TRẬN CHIẾN BẮT ĐẦU
    # print("\n" + "="*50)
    # print("⚔️ KHỞI ĐỘNG VÒNG LẶP K-FOLD (5 LẦN THỬ LỬA)")
    # print("="*50)

    # for depth in [10,15,16,17,18,20]:
    #     for min_split in [5,4,3,2]:
    #         print(f"\n--- Đang test Ứng cử viên: Depth {depth}, Min {min_split} ---")
    #         fold_scores = []
    #         t_scores = []
    #         # Chạy 5 vòng K-Fold
    #         for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
    #             # Tách đề thi và sách giáo khoa theo chỉ số của K-Fold
    #             X_kf_train, X_kf_val = X_cv[train_idx], X_cv[val_idx]
    #             y_kf_train, y_kf_val = y_cv[train_idx], y_cv[val_idx]
    #             start_time = time.time()

                
    #             model = MyRandomForest(n_trees=100, max_depth=depth, min_samples_split=min_split, n_features=n_features_cv)
    #             model.fit(X_kf_train, y_kf_train)
    #             # Bắt đầu thi
    #             preds = model.predict(X_kf_val)
    #             acc = np.mean(preds == y_kf_val) * 100
    #             tim = time.time() - start_time
    #             fold_scores.append(acc)
    #             t_scores.append(tim)
    #             print(f"  + Lần {fold+1}: {acc:.2f}% với {tim:.2f} giây")
                
    #         # Tính điểm trung bình môn
    #         mean_score = np.mean(fold_scores)
    #         time_score = np.mean(t_scores)
    #         print(f"👉 ĐIỂM TRUNG BÌNH K-FOLD : {mean_score:.2f}% và {time_score:.2f}s")

    # print("\n🎉 HOÀN TẤT K-FOLD! SẾP HÃY CHỌN THẰNG CÓ ĐIỂM TRUNG BÌNH CAO NHẤT LÀM TRÙM CUỐI!")
    # import matplotlib.pyplot as plt

    # plt.hist(X_train[:, 1], bins=50)
    # plt.show()
    e = EDA()
    e.plot_class_balance(y_train, y_valid, y_test)
    e.plot_image_dimensions('PetImages', sample_size=1000)
    e.plot_pixel_intensity('PetImages', sample_size=1000)
    e.plot_mean_image('PetImages', img_size=(100, 100), sample_size=1000)
    e.plot_pca_correlation(X_train, num_components=15)
    e.plot_pca_pairplot(X_train, y_train, num_components=4)
    