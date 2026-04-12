import numpy as np
from collections import Counter
import concurrent.futures
import os
from joblib import Parallel, delayed
from multiprocessing import shared_memory
from numba import njit

@njit
def calculate_GINI(arr):
    if len(arr) ==0:
        return 0
    count = np.bincount(arr)
    probabilities = count / len(arr)
    gini = 1 - np.dot(probabilities, probabilities)
    
    return gini


@njit
def find_best_split(X, y, feat_idxs, n_bins, parent_gini):
    """Quét toàn bộ ma trận để tìm nhát chém tốt nhất"""
    n_samples = len(y)
    best_gain = -1.0
    best_idx = -1     # Dùng -1 thay vì None vì Numba bắt buộc phải trả về số
    best_thresh = 0.0

    if n_samples == 0:
        return best_idx, best_thresh

    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx]
        min_val = np.min(X_column)
        max_val = np.max(X_column)
        
        if min_val == max_val:
            continue
            
        bin_width = (max_val - min_val) / n_bins
        
        # Đếm Histogram thủ công cho 2 class (Mèo=0, Chó=1)
        hist_0 = np.zeros(n_bins)
        hist_1 = np.zeros(n_bins)
        
        for i in range(n_samples):
            val = X_column[i]
            label = y[i]
            
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= n_bins: 
                bin_idx = n_bins - 1
                
            if label == 0:
                hist_0[bin_idx] += 1
            else:
                hist_1[bin_idx] += 1

        left_count_0 = 0.0
        left_count_1 = 0.0
        total_0 = np.sum(hist_0)
        total_1 = np.sum(hist_1)
        
        for i in range(n_bins - 1):
            left_count_0 += hist_0[i]
            left_count_1 += hist_1[i]
            
            left_total = left_count_0 + left_count_1
            right_total = n_samples - left_total
            
            if left_total == 0 or right_total == 0:
                continue
                
            p0_left = left_count_0 / left_total
            p1_left = left_count_1 / left_total
            gini_left = 1.0 - (p0_left**2 + p1_left**2)
            
            right_count_0 = total_0 - left_count_0
            right_count_1 = total_1 - left_count_1
            p0_right = right_count_0 / right_total
            p1_right = right_count_1 / right_total
            gini_right = 1.0 - (p0_right**2 + p1_right**2)
            
            gini = (gini_left * left_total + gini_right * right_total) / n_samples
            gain = parent_gini - gini
            
            if gain > best_gain:
                best_gain = gain
                best_thresh = min_val + (i + 1) * bin_width
                best_idx = feat_idx

    return best_idx, best_thresh

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
        
        # --- BƯỚC 2: RANDOM ---
        # Chọn NGẪU NHIÊN một số cột từ tổng số cột n_feats
        if self.n_features is None:
            n_feats_to_use = int(np.sqrt(n_feats))
        else:
            n_feats_to_use = self.n_features
        feat_idxs = np.random.choice(n_feats, n_feats_to_use, replace=False)

        # --- BƯỚC 3: TÌM NHÁT CHÉM TỐT NHẤT ---
        # Gọi hàm best_split để tìm ra Cột tốt nhất và Ngưỡng tốt nhất
        gini_parent = calculate_GINI(y)
        best_feat, best_thresh = find_best_split(X, y, feat_idxs, n_bins=50, parent_gini=gini_parent)
        
        if best_feat == -1:
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

    
    def train_single_tree(self, X, y,seed):
        np.random.seed(seed)

        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, label_sample = X[idxs], y[idxs]
        tree = MyDecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
        
        tree.fit(X_sample, label_sample)
        return tree
    
    
    def fit(self, X, y):
    
        self.trees = Parallel(n_jobs=os.cpu_count()-1, prefer="processes")(
            delayed(self.train_single_tree)(X, y, seed) for seed in range(self.n_trees) 
        )
        

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    