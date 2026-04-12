import streamlit as st
import numpy as np
from PIL import Image
import time
import joblib
import os
from extract_feature import get_hog_features

# --- CẤU HÌNH TRANG GIAO DIỆN ---
st.set_page_config(page_title="Phân Loại Chó Mèo", page_icon="🐶", layout="centered")
st.title(" Hệ Thống Phân Loại Chó Mèo")
st.write("Tải một bức ảnh lên và chọn model !")

# --- HÀM LOAD MODEL (Tối ưu: Load cả PCA ở đây luôn) ---
@st.cache_resource
def load_all_models():
    models = {}
    model_dir = "models"
    if not os.path.exists(model_dir):
        st.warning("Chưa có thư mục models. Vui lòng train và lưu model trước!")
        return models
        
    try:
        models["Random Forest"] = joblib.load(f"{model_dir}/Random_Forest.pkl")
        models["Logistic Regression"] = joblib.load(f"{model_dir}/Logistic_Regression.pkl")
        models["Support Vector Machine"] = joblib.load(f"{model_dir}/Support_Vector_Machine.pkl")
        models["PCA"] = joblib.load(f"{model_dir}/pca_model.pkl")
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        
    return models

models_dict = load_all_models()

# --- HÀM TIỀN XỬ LÝ (Nhận thêm tham số pca_model) ---
def preprocess_image_for_model(img_path, pca_model):
    # 1. Trích xuất HOG từ file tạm
    feature_hog = get_hog_features(img_path)  
    
    # 2. Dùng khuôn PCA đã load sẵn để ép chiều
    feature_pca = pca_model.transform([feature_hog])
    
    return feature_pca

# --- GIAO DIỆN CHÍNH ---
if models_dict:
    selected_model_name = st.selectbox("Chọn mô hình AI:", [k for k in models_dict.keys() if k != "PCA"])
    
    uploaded_file = st.file_uploader("Chọn một bức ảnh (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Mở ảnh một lần duy nhất
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh bạn vừa tải lên", width=300)

        if st.button("Bắt đầu dự đoán", use_container_width=True):
            with st.spinner('Đang nhẩm tính...'):
                temp_img_path = "temp_upload.jpg"
                
                # Xử lý hệ màu và lưu file tạm
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(temp_img_path)

                try:
                    # 🚀 Tính thời gian bắt đầu (Bao gồm cả preprocessing cho nó thực tế)
                    start_total_time = time.time()

                    # Tiền xử lý (Truyền thêm khuôn PCA đã cache)
                    X_input = preprocess_image_for_model(temp_img_path, models_dict["PCA"])
                    
                    # Dự đoán
                    model = models_dict[selected_model_name]
                    prediction = model.predict(X_input)
                    
                    # ⏱️ Chốt thời gian
                    total_time = time.time() - start_total_time
                    
                    st.markdown("---")
                    result_class = prediction[0]
                    if result_class == 1:
                        st.success(f"## KẾT QUẢ: ĐÂY LÀ CON CHÓ!")
                    else:
                        st.info(f"## KẾT QUẢ: ĐÂY LÀ CON MÈO!")
                        
                    st.caption(f" Tổng thời gian xử lý và dự đoán: {total_time:.5f} giây")
                    
                except Exception as e:
                    st.error(f"Gặp lỗi trong lúc dự đoán: {e}")
                    
                finally:
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)