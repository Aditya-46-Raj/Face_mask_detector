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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main-title {
            text-align: center;
            color: #1e3a8a;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-text {
            text-align: center;
            color: #475569;
            font-size: 1.1rem;
            margin-bottom: 15px;
        }
        .info-box {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 5px solid #f59e0b;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .info-box-text {
            color: #78350f;
            font-size: 1rem;
            font-weight: 500;
            margin: 0;
        }
        .prediction-box {
            text-align: center;
            border-radius: 15px;
            padding: 25px;
            color: white;
            font-size: 1.8rem;
            font-weight: bold;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin: 20px 0;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .confidence-text {
            text-align: center;
            font-size: 1.2rem;
            color: #475569;
            margin-top: 10px;
            font-weight: 500;
        }
        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 40px;
            padding: 20px;
            background-color: #f1f5f9;
            border-radius: 10px;
        }
        
        /* File Uploader Styling for Dark Mode */
        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 2px dashed #8b5cf6;
            border-radius: 15px;
            padding: 30px 20px;
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stFileUploader"] > div {
            background-color: transparent !important;
        }
        
        [data-testid="stFileUploader"] label {
            color: #a78bfa !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        
        [data-testid="stFileUploader"] section {
            border: none !important;
            background-color: transparent !important;
        }
        
        [data-testid="stFileUploader"] section > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: transform 0.2s, box-shadow 0.2s !important;
        }
        
        [data-testid="stFileUploader"] section > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4) !important;
        }
        
        [data-testid="stFileUploader"] small {
            color: #c4b5fd !important;
        }
        
        .divider {
            height: 2px;
            background: linear-gradient(to right, transparent, #667eea, transparent);
            margin: 30px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-title'>😷 Face Mask Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Upload an image to check if the person is wearing a mask or not.</div>", unsafe_allow_html=True)

# Important Note Box
st.markdown("""
    <div class='info-box'>
        <p class='info-box-text'>
            ⚠️ <b>Important:</b> For better accuracy, please upload a <b>closely cropped image</b> 
            of the face. The model works best when the face occupies most of the image area.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Cache model to avoid reloading every time
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/saved_models/face_mask_detection_improved_model.h5")

model = load_model()
input_shape = model.input_shape[1:3]

# File uploader
uploaded_file = st.file_uploader("📁 Choose an image file", type=["jpg", "jpeg", "png", "JPG"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Display image in a column for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption="📸 Uploaded Image", use_container_width=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    with st.spinner("🔍 Analyzing image... Please wait"):
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0][0]
        confidence = prediction if prediction > 0.5 else (1 - prediction)

    # Display result with confidence
    if prediction > 0.5:
        st.markdown(
            f"<div class='prediction-box' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%);'>✅ Wearing Mask</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='prediction-box' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);'>❌ Not Wearing Mask</div>",
            unsafe_allow_html=True
        )
    
    st.markdown(f"<div class='confidence-text'>🎯 Confidence: {confidence*100:.2f}%</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class='footer'>
        <b>Developed by Aditya Raj</b> | NIT Patna<br>
        🚀 Powered by TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)
