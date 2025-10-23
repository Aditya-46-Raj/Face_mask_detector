import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Face Mask Detector 😷",
    page_icon="😷",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .main-title {
            text-align: center;
            color: #2e7d32;
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-text {
            text-align: center;
            color: #333;
            font-size: 1rem;
            margin-bottom: 25px;
        }
        .prediction-box {
            text-align: center;
            border-radius: 15px;
            padding: 20px;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-title'>😷 Face Mask Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Upload an image to check if the person is wearing a mask or not.</div>", unsafe_allow_html=True)

# Cache model to avoid reloading every time
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/saved_models/face_mask_detection_improved_model.h5")

model = load_model()
input_shape = model.input_shape[1:3]

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "JPG"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image... Please wait ⏳"):
        img = img.resize(input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0][0]

    # Display result
    if prediction > 0.5:
        st.markdown(
            "<div class='prediction-box' style='background-color: #2e7d32;'>✅ Wearing Mask</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='prediction-box' style='background-color: #c62828;'>❌ Not Wearing Mask</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("<div class='footer'>Developed by <b>Aditya Raj</b> | NIT Patna</div>", unsafe_allow_html=True)
