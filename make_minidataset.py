import os
import random
import numpy as np
import json

def generate_dataset_indices(source_dir, save_dir='data_splits', mode='mini', 
                             num_train=4000, num_valid=1000, num_test=1000, seed=42,cat_ratio=0.5):
    """
    Hàm chia dataset chung cho cả mini và full.
    mode: Tiền tố để đặt tên file (ví dụ: 'mini' hoặc 'full')
    """
    if mode == 'full':
        num_train, num_valid, num_test = 16000, 4000, 4000
    print(f"Đang tạo bản đồ index cho {mode.upper()} dataset...")
    os.makedirs(save_dir, exist_ok=True)
    
    cat_source = os.path.join(source_dir, 'Cat')
    dog_source = os.path.join(source_dir, 'Dog')
    
    cats = [(os.path.join('Cat', f), 0) for f in os.listdir(cat_source) if f.endswith(('.jpg', '.png'))]
    dogs = [(os.path.join('Dog', f), 1) for f in os.listdir(dog_source) if f.endswith(('.jpg', '.png'))]
    
    random.seed(42) 
    random.shuffle(cats)
    random.shuffle(dogs)
    
    c_train = int(num_train * cat_ratio)
    d_train = num_train - c_train
    
    # Chia đều số lượng yêu cầu cho 2 class (Mèo / Chó)
    c_val, d_val = int(num_valid // 2), int(num_valid // 2)
    c_test, d_test = int(num_test // 2), int(num_test // 2)
    
    train_data = cats[:c_train] + dogs[:d_train]
    
    valid_data = (cats[c_train : c_train + c_val] + 
                  dogs[d_train : d_train + d_val])
                  
    test_data = (cats[c_train + c_val : c_train + c_val + c_test] + 
                 dogs[d_train + d_val : d_train + d_val + d_test])
                 
    dataset_splits = {'train': train_data, 'valid': valid_data, 'test': test_data}
    
    print(f"[{mode.upper()}] Đã lấy: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")

    # Xáo trộn lại từng tập
    for split_name in dataset_splits.keys():
        random.shuffle(dataset_splits[split_name])

    # Lưu ra file JSON
    json_path = os.path.join(save_dir, f'{mode}_dataset.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_splits, f, indent=4)
        
    print(f"-> Đã lưu xong tại: {json_path}\n")

# ==========================================
# CÁCH CHẠY KHI DEBUG (Tạo bản mini 400-100-100)
# ==========================================
# generate_dataset_indices(
#     source_dir='PetImages', 
#     mode='mini', 
#     num_train=4000, num_valid=1000, num_test=1000
# )

# ==========================================
# CÁCH CHẠY KHI ĐEM LÊN COLAB TRAIN THẬT (16k-4k-4k)
# ==========================================
# generate_dataset_indices(
#     source_dir='PetImages', 
#     mode='full', 
#     num_train=16000, num_valid=4000, num_test=4000
# )
