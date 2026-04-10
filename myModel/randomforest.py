import numpy as np
from collections import Counter


def calculate_GINI(arr):
    if len(arr) ==0:
        return 0
    count = np.bincount(arr)
    probabilities = count / len(arr)
    gini = 1 - np.sum(probabilities ** 2)
    
    return gini



# ==========================================
# CẤU TRÚC NODE
# ==========================================
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        # Nếu là NODE CÀNH (Chứa nhát chém):
        self.feature_idx = feature_idx  # Cắt ở cột PCA thứ mấy?
        self.threshold = threshold      # Giá trị ngưỡng cắt là bao nhiêu?
        self.left = left                # Cành bên trái (cũng là 1 Node)
        self.right = right              # Cành bên phải (cũng là 1 Node)
        
        # Nếu là NODE LÁ (Chứa kết quả phân loại cuối cùng):
        self.value = value              

    def is_leaf_node(self):
        if self.value is not None:
            return True
        return False



# ==========================================
# CẤU TRÚC CÂY (Decision Tree)
# ==========================================
class MyDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split # Số data tối thiểu để được phép chém tiếp
        self.max_depth = max_depth                 # Chiều cao tối đa của cây (chống Overfitting)
        self.n_features = n_features               # Cây sẽ được phép nhìn bao nhiêu cột ngẫu nhiên?
        self.root = None                           # Cái rễ cây ban đầu

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        """HÀM ĐỆ QUY: Tự động mọc cành và lá"""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # --- BƯỚC 1: ĐIỀU KIỆN DỪNG (Rụng thành lá) ---
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split):
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)
        
        # --- BƯỚC 2: ẢO THUẬT RANDOM FOREST ---
        # Chọn NGẪU NHIÊN một số cột từ tổng số cột n_feats
        if self.n_features is None:
            n_feats_to_use = int(np.sqrt(n_feats))
        else:
            n_feats_to_use = self.n_features
        feat_idxs = np.random.choice(n_feats, n_feats_to_use, replace=False)

        # --- BƯỚC 3: TÌM NHÁT CHÉM TỐT NHẤT ---
        # Gọi hàm _best_split để tìm ra Cột tốt nhất và Ngưỡng tốt nhất
        best_feat, best_thresh = self.best_split(X, y, feat_idxs)
        # best_feat, best_thresh = self.best_split_V2(X, y, feat_idxs)
        
        if best_feat is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)
        
        # --- BƯỚC 4: CHÉM VÀ ĐỆ QUY ---
        # Dùng best_feat và best_thresh để chia X và y thành 2 tập: (Left) và (Right)
        left_idxs = np.where(X[:, best_feat] <= best_thresh)[0]
        right_idxs = np.where(X[:, best_feat] > best_thresh)[0]
        
        left_child = self.grow_tree(X[left_idxs,:], y[left_idxs], depth + 1)
        right_child = self.grow_tree(X[right_idxs,:], y[right_idxs], depth + 1)

        # --- BƯỚC 5: TRẢ VỀ NODE CÀNH ---
        return Node(best_feat, best_thresh, left_child, right_child)


    def best_split(self, X, y, feat_idxs):
        """Hàm cầm kiếm đi chém thử để tìm Gini Gain lớn nhất"""
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_gini = calculate_GINI(y)
        # Vòng lặp 1: Duyệt qua từng cột trong feat_idxs
            # Lấy toàn bộ con số của cột đó ra (X_column)
            # Tìm tất cả các giá trị duy nhất (unique) trong cột đó để làm Ngưỡng cắt thử nghiệm (thresholds)
            
            # Vòng lặp 2: Duyệt qua từng threshold
                # Cắt thử X_column bằng threshold này
                # Tính Gini_Gain (như công thức tui ghi ở trên)
                
                # Nếu Gini_Gain > best_gain hiện tại:
                    # Cập nhật lại best_gain, split_idx, split_thresh
        
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for thresh in thresholds:
                left_idxs = np.where(X_column <= thresh)[0]
                right_idxs = np.where(X_column > thresh)[0]
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                left_arr = y[left_idxs]
                right_arr = y[right_idxs]
                
                gini = calculate_GINI(left_arr) * len(left_arr) / len(y) + calculate_GINI(right_arr) * len(right_arr) / len(y)
                gain = parent_gini - gini
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
                

        # Trả về split_idx và split_thresh tốt nhất
        return split_idx, split_thresh

    def best_split_V2(self, X, y, feat_idxs):
        """Hàm cầm kiếm đi chém thử để tìm Gini Gain lớn nhất bằng Histogram"""
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_gini = calculate_GINI(y)
        
        n_bins = 50 
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            # --- BƯỚC 1: TẠO BINS ---
            min_val, max_val = X_column.min(), X_column.max()
            if min_val == max_val:
                continue
            bins = np.linspace(min_val,max_val,n_bins)
            
            # --- BƯỚC 2: GÁN BINS ---  
            bin_ids = np.digitize(X_column, bins)
            
            # --- BƯỚC 3: TẠO HISTOGRAM ---
            hist = np.zeros((n_bins + 1, 2)) 
        
            for c in range(2):
                hist[:, c] = np.bincount(bin_ids[y == c], minlength=n_bins+1)

            # --- BƯỚC 4: CUMULATIVE SUM ---
            left_hist = np.cumsum(hist, axis=0)
            total_hist = left_hist[-1]
            
            
            for i in range(1, n_bins):
                left = left_hist[i]
                right = total_hist - left

                left_count = np.sum(left)
                right_count = np.sum(right)

                if left_count == 0 or right_count == 0:
                    continue

                # --- TÍNH GINI ---
                left_gini = 1 - np.sum((left / left_count) ** 2)
                right_gini = 1 - np.sum((right / right_count) ** 2)

                gini = (left_gini * left_count + right_gini * right_count ) / len(y)

                gain = parent_gini - gini
                
                if gain > best_gain:

                    left_idxs = np.where(bin_ids <= i)[0]
                    right_idxs = np.where(bin_ids > i)[0]

                    if len(left_idxs) == 0 or len(right_idxs) == 0:
                        continue  # bỏ luôn threshold này
                
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = (bins[i] + bins[i-1]) / 2
                

        # Trả về split_idx và split_thresh tốt nhất
        return split_idx, split_thresh
    
    
    def predict(self, X):
        return np.array([self.traverse_tree(x_single_img, self.root) for x_single_img in X])

    def traverse_tree(self, x_single_img, node):
        if node.is_leaf_node():
            return node.value
        if x_single_img[node.feature_idx] <= node.threshold:
            return self.traverse_tree(x_single_img, node.left)
        return self.traverse_tree(x_single_img, node.right)
    
    
    
    
    
# ==========================================
# CẤU TRÚC RỪNG (Random Forest)
# ==========================================
class MyRandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees                       # Số lượng cây trong rừng
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []                              # Cái mảng này để chứa N cái cây sau khi trồng

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for t in range(self.n_trees):
            # --- BƯỚC 1: TẠO DATA ẢO BẰNG BOOTSTRAP ---

            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, label_sample = X[idxs], y[idxs]

            # --- BƯỚC 2: KHỞI TẠO VÀ TRỒNG 1 CÁI CÂY ---
            tree = MyDecisionTree(max_depth=self.max_depth,
                                  min_samples_split=self.min_samples_split,
                                  n_features=self.n_features)
            

            tree.fit(X_sample, label_sample)
            
            self.trees.append(tree)

    def predict(self, X):
        """Cho cả khu rừng dự đoán và bầu cử"""
        # Bước 1: Lấy dự đoán của TẤT CẢ các cây
        # Kết quả sẽ là 1 ma trận (Số lượng cây x Số lượng ảnh)
        tree_preds = np.array([ t.predict(X) for t in self.trees ])

        # Bước 2: Xoay ma trận lại thành (Số lượng ảnh x Số lượng cây)
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # Bước 3: Đếm phiếu bầu (Majority Vote)
        predictions = []
        for votes in tree_preds:
            # votes là mảng các phiếu bầu cho 1 bức ảnh (ví dụ: [0, 0, 1, 0, 1])
            most_common = Counter(votes).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # ==========================================
# # TEST THỬ CODE (UNIT TEST VỚI 5 CỘT)
# # ==========================================
# if __name__ == "__main__":
#     print("🚀 ĐANG TEST THỬ CÂY QUYẾT ĐỊNH VỚI 5 CỘT (n_features=3)...")

#     # 1. Tự chế ra tập "Data đồ chơi" (10 ảnh, mỗi ảnh 5 đặc trưng PCA)
#     # Các cột: PC0, PC1, PC2, PC3, PC4
#     # Mèo (0): PC0, PC1 thấp; PC3 rất cao (cột đánh lừa)
#     # Chó (1): PC0, PC1 cao; PC3 rất thấp (cột đánh lừa)
#     np.random.seed(42) # Khóa seed để kết quả random luôn giống nhau khi test
    
#     X_test_mau = np.array([
#         [1.5, 2.0, 0.5, 9.1, 1.2], # Mèo
#         [1.0, 2.5, 0.8, 8.5, 1.5], # Mèo
#         [2.0, 1.5, 0.3, 7.2, 1.8], # Mèo
#         [1.8, 2.2, 0.9, 8.8, 1.1], # Mèo
#         [1.2, 1.8, 0.6, 9.5, 1.3], # Mèo
        
#         [5.0, 4.5, 8.5, 1.1, 8.2], # Chó
#         [4.5, 5.0, 7.8, 2.5, 7.5], # Chó
#         [6.0, 4.0, 9.1, 1.8, 8.8], # Chó
#         [5.5, 5.5, 8.8, 0.5, 9.2], # Chó
#         [4.8, 4.8, 7.5, 2.1, 7.1]  # Chó
#     ])

#     # Nhãn: 5 Mèo (0), 5 Chó (1)
#     y_test_mau = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1])

#     # 2. Đem Cây ra trồng
#     print("🌲 Đang trồng cây (Cây chỉ được chọn 3/5 cột ngẫu nhiên để chém)...")
#     # Set n_features = 3 để test tính năng ngẫu nhiên của Random Forest
#     tree = MyRandomForest(n_trees=4,max_depth=3,min_samples_split=2, n_features=3)
#     tree.fit(X_test_mau, y_test_mau)

#     # 3. Dự đoán lại chính tập vừa học
#     print("🔍 Đang dự đoán...")
#     predictions = tree.predict(X_test_mau)

#     # 4. Chấm điểm
#     print("\n--- KẾT QUẢ ---")
#     print(f"Nhãn thực tế: {y_test_mau}")
#     print(f"Máy dự đoán:  {predictions}")
    
#     accuracy = np.sum(predictions == y_test_mau) / len(y_test_mau) * 100
#     print(f"Độ chính xác: {accuracy:.2f}%")
    
#     if accuracy == 100.0:
#         print("🎉 XUẤT SẮC! Cây phân loại 100%. Hàm random chọn 3 cột hoạt động hoàn hảo!")
#     else:
#         print("🐛 Cây chưa đạt 100%, có thể do random trúng toàn cột nhiễu hoặc có lỗi logic.")



if __name__ == "__main__":
    print("🚀 TEST CÂY VỚI 7 CỘT - 8 HÀNG (n_features=5)...")

    np.random.seed(42)

    X_test = np.array([
    [2.1, 1.9, 2.0, 5.5, 5.2, 5.1, 3.0],  # 0
    [2.3, 2.2, 1.8, 5.3, 5.4, 5.0, 3.2],  # 0
    [1.8, 2.1, 2.2, 5.6, 5.1, 5.3, 3.1],  # 0
    [2.0, 2.0, 2.1, 5.4, 5.3, 5.2, 3.3],  # 0
    [2.2, 1.7, 2.3, 5.2, 5.5, 5.4, 3.0],  # 0

    [2.1, 1.9, 2.0, 5.5, 5.2, 5.1, 6.0],  # 1
    [2.3, 2.2, 1.8, 5.3, 5.4, 5.0, 6.2],  # 1
    [1.8, 2.1, 2.2, 5.6, 5.1, 5.3, 6.1],  # 1
    [2.0, 2.0, 2.1, 5.4, 5.3, 5.2, 6.3],  # 1
    [2.2, 1.7, 2.3, 5.2, 5.5, 5.4, 6.0],  # 1
    ])

    y_test = np.array([0,0,0,0,0, 1,1,1,1,1])

    # Khởi tạo cây
    tree = MyDecisionTree(
        max_depth=2,
        min_samples_split=2,
        n_features=4  # mỗi node chọn 5/7 cột
    )

    print("🌲 Đang train...")
    tree.fit(X_test, y_test)

    print("🔍 Đang predict...")
    preds = tree.predict(X_test)

    print("\n--- KẾT QUẢ ---")
    print("Nhãn thật: ", y_test)
    print("Dự đoán:   ", preds)

    acc = np.sum(preds == y_test) / len(y_test) * 100
    print(f"Accuracy: {acc:.2f}%")

    if acc == 100:
        print("🎉 Perfect luôn!")
    else:
        print("⚠️ Có thể do random feature chọn trúng cột nhiễu")