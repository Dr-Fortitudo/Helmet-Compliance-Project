import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo

# Page config
st.set_page_config(page_title="Helmet Compliance Detector", page_icon="⛑️", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.savedmodel", compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

class_names = ["ON Helmet", "NO Helmet"]

# Preprocessing
def preprocess(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Predict
def predict(image):
    img_array = preprocess(image)
    prediction = model.predict(img_array)
    label = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return label, confidence

# Init log
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.markdown("<h1 style='text-align:center;'>⛑️ Helmet Compliance Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Capture or upload an image to check for helmet compliance</p>", unsafe_allow_html=True)
st.markdown("---")

# Input selector
col1, col2 = st.columns(2)
with col1:
    input_type = st.selectbox("Choose Input Method", ["Upload Image", "Camera Input"])
with col2:
    threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.01)

image = None
filename = ""

# File or camera input
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        filename = uploaded_file.name
elif input_type == "Camera Input":
    camera_file = st.camera_input("📷 Capture from Camera")
    if camera_file:
        image = Image.open(camera_file)
        filename = "camera_snapshot.png"

# Prediction & Output
if image:
    st.image(image, caption=filename, use_column_width=True)

    with st.spinner("🧠 Analyzing..."):
        label, confidence = predict(image)
        confidence_percent = confidence * 100

    st.markdown("---")

    if label == "ON Helmet":
        st.success(f"✅ Worker is in compliance")
        st.markdown(f"<p style='color:green; font-size:18px;'>Confidence: {confidence_percent:.2f}%</p>", unsafe_allow_html=True)
    else:
        st.error(f"❌ Worker is not wearing a helmet")
        st.markdown(f"<p style='color:red; font-size:18px;'>Confidence: {confidence_percent:.2f}%</p>", unsafe_allow_html=True)

    # Store log if confidence is high
    if confidence >= threshold:
        st.session_state.history.insert(0, {
            "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
            "result": label,
            "confidence": f"{confidence_percent:.2f}%",
            "image": filename
        })

# History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🧾 Detection Log")
    for entry in st.session_state.history[:5]:
        st.markdown(f"- **{entry['timestamp']}** | `{entry['result']}` ({entry['confidence']}) — *{entry['image']}*")
else:
    st.info("No predictions logged yet.")
