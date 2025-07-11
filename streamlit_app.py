import streamlit as st
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo

# Page config
st.set_page_config(page_title="Helmet Compliance Detector", page_icon="⛑️", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.savedmodel", compile=False)

model = load_model()
class_names = ["ON Helmet", "NO Helmet"]

# Preprocessing
def preprocess(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Prediction
def predict(image):
    processed = preprocess(image)
    prediction = model.predict(processed)
    label = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return label, confidence

# Session history
if 'helmet_log' not in st.session_state:
    st.session_state.helmet_log = []

# Title
st.markdown("<h1 style='text-align: center; color: #3ABEFF;'>⛑️ Helmet Compliance Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Capture or upload an image to detect helmet compliance</p>", unsafe_allow_html=True)
st.markdown("---")

# Input method
col1, col2 = st.columns(2)
with col1:
    input_method = st.selectbox("Choose Input Method", ["Upload Image", "Camera Input"])
with col2:
    threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.01)

image = None
filename = ""

# Input handling
if input_method == "Upload Image":
    uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        filename = uploaded.name
else:
    camera_image = st.camera_input("📷 Take a picture")
    if camera_image:
        image = Image.open(camera_image)
        filename = "Camera Snapshot"

# Prediction block
if image:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(image, caption=filename, use_column_width=True)

    with col2:
        with st.spinner("🧠 Analyzing..."):
            label, confidence = predict(image)
            confidence_percent = confidence * 100

        if label == "ON Helmet":
            st.markdown(f"""
                <div style='background-color: #e6ffed; padding: 1.5rem; border-radius: 12px; border: 2px solid #22c55e;'>
                    <h2 style='color: #22c55e;'>✅ COMPLIANT</h2>
                    <p style='font-size: 1.1rem;'><strong>Confidence:</strong> {confidence_percent:.2f}%</p>
                    <p>No safety violation detected.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color: #ffeaea; padding: 1.5rem; border-radius: 12px; border: 2px solid #f87171;'>
                    <h2 style='color: #f87171;'>⚠️ INCOMPLIANT</h2>
                    <p style='font-size: 1.1rem;'><strong>Confidence:</strong> {confidence_percent:.2f}%</p>
                    <p>Helmet not detected. Safety violation possible.</p>
                </div>
            """, unsafe_allow_html=True)

        # Log the result
        if confidence >= threshold:
            st.session_state.helmet_log.insert(0, {
                "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
                "image": filename,
                "result": label,
                "confidence": f"{confidence_percent:.2f}%"
            })

# Detection log
if st.session_state.helmet_log:
    st.markdown("---")
    st.markdown("### 🧾 Detection History")
    for log in st.session_state.helmet_log[:5]:
        with st.expander(f"{log['timestamp']} - {log['result']} ({log['confidence']})"):
            st.write(log)
else:
    st.info("No detections logged yet.")
