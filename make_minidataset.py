import os 
import random
import shutil

def create_minidataset(source_dir,target_dir, num_train = 1000 , num_test = 200):
    print('đang tạo mini dataset')
    # xóa thư mục mini cũ nếu có
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    #tạo thư mục mini mới
    dirs_to_make = [
        f"{target_dir}/train/cats", f"{target_dir}/train/dogs",
        f"{target_dir}/test/cats", f"{target_dir}/test/dogs"
    ]
    for d in dirs_to_make:
        os.makedirs(d,exist_ok= True)
        
    #lấy đường dẫn folder gốc (2 folder)
    cat_source = os.path.join(source_dir, 'Cat')
    dog_source = os.path.join(source_dir, 'Dog')
    
    #đọc dữ liệu
    cats = [f for f in os.listdir(cat_source) ]
    dogs = [f for f in os.listdir(dog_source) ]
    
    #shuffle
    random.seed(42)
    random.shuffle(cats)
    random.shuffle(dogs)
    
    #lấy train và test
    train_cats = cats[:num_train]
    train_dogs = dogs[:num_train]
    test_cats = cats[num_train:num_test + num_train]
    test_dogs = dogs[num_train:num_test + num_train]
    
    print(f"Đang copy {num_train} ảnh Train và {num_test} ảnh Test cho mỗi loài...")
    
    def copy_files(file_list , src_folder, dest_folder):
        for file_name in file_list:
            src_path = os.path.join(src_folder, file_name)
            dst_path = os.path.join(dest_folder, file_name)
            try:
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                pass # Bỏ qua nếu có ảnh bị lỗi không copy được

    #copy từ thư mục gốc sang minidataset
    copy_files(train_cats, cat_source, f"{target_dir}/train/cats")
    copy_files(test_cats, cat_source, f"{target_dir}/test/cats")
    
    copy_files(train_dogs, dog_source, f"{target_dir}/train/dogs")
    copy_files(test_dogs, dog_source, f"{target_dir}/test/dogs")

    print(f"Đã tạo xong minidataset tại thư mục: {target_dir}!")
    

create_minidataset(
    source_dir= 'PetImages',
    target_dir= 'minidata',
    num_train= 2000,
    num_test= 200,
)