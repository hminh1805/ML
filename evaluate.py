import time
import tracemalloc
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def evaluate_and_compare(models_dict, X_train, y_train, X_test, y_test):
    """
    Hàm đánh giá và so sánh nhiều mô hình Machine Learning.
    
    Tham số:
    - models_dict: Dictionary chứa tên mô hình và object mô hình tương ứng.
    - X_train, y_train: Dữ liệu huấn luyện.
    - X_test, y_test: Dữ liệu kiểm thử.
    """
    results = []

    print(f"{'='*50}")
    print(" BẮT ĐẦU QUÁ TRÌNH SO SÁNH CÁC MÔ HÌNH")
    print(f"{'='*50}\n")

    for model_name, model in models_dict.items():
        print(f"⏳ Đang xử lý: {model_name}...")
        
        # 1. BẮT ĐẦU ĐO LƯỜNG BỘ NHỚ VÀ THỜI GIAN
        tracemalloc.start()
        start_train_time = time.time()
        
        # 2. HUẤN LUYỆN MÔ HÌNH (TRAIN)
        model.fit(X_train, y_train)
        end_train_time = time.time()
        
        # 3. DỰ ĐOÁN (PREDICT)
        start_pred_time = time.time()
        y_pred = model.predict(X_test)
        end_pred_time = time.time()
        
        # 4. KẾT THÚC ĐO LƯỜNG BỘ NHỚ
        # current: bộ nhớ hiện tại đang dùng, peak: mức bộ nhớ tiêu tốn cao nhất
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Chuyển đổi bộ nhớ từ Bytes sang Megabytes (MB)
        peak_mem_mb = peak_mem / (1024 * 1024)
        
        # Tính toán thời gian
        train_time = end_train_time - start_train_time
        pred_time = end_pred_time - start_pred_time
        
        # 5. TÍNH TOÁN CÁC CHỈ SỐ KỸ THUẬT
        # Lưu ý: average='binary' vì đây là bài toán phân loại nhị phân (chó/mèo)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # 6. LƯU KẾT QUẢ
        results.append({
            "Mô hình": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "T.gian Train (s)": round(train_time, 4),
            "T.gian Đoán (s)": round(pred_time, 4),
            "RAM Tối đa (MB)": round(peak_mem_mb, 4)
        })
        
        print(f"✅ Hoàn thành: {model_name}\n")

    # Chuyển đổi list dictionary thành Pandas DataFrame để hiển thị dạng bảng đẹp mắt
    df_results = pd.DataFrame(results)
    
    print(f"{'='*50}")
    print(" BẢNG TỔNG HỢP KẾT QUẢ")
    print(f"{'='*50}")
    print(df_results.to_string(index=False))
    
    return df_results

# ==========================================
# CÁCH SỬ DỤNG HÀM TRONG THỰC TẾ
# ==========================================
if __name__ == "__main__":
    import numpy as np
    
    # 1. Tải dữ liệu đã qua xử lý HOG và PCA từ các bước trước
    try:
        print("Đang tải dữ liệu...")
        X_train = np.load('feature_train_pca.npy')
        y_train = np.load('label_train.npy')
        X_test = np.load('feature_test_pca.npy')
        y_test = np.load('label_test.npy')
        print("Tải dữ liệu thành công!\n")
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file dữ liệu. Hãy đảm bảo bạn đã chạy file pca.py!")
        exit()

    # 2. Khởi tạo danh sách các mô hình cần so sánh
    models_to_compare = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel='rbf', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 3. Gọi hàm so sánh
    df_report = evaluate_and_compare(models_to_compare, X_train, y_train, X_test, y_test)
    
    # (Tùy chọn) Lưu kết quả ra file Excel hoặc CSV để chép vào báo cáo
    # df_report.to_csv("model_comparison_report.csv", index=False)
    # print("\n💾 Đã lưu báo cáo ra file model_comparison_report.csv")