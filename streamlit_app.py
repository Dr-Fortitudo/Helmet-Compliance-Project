import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import zipfile
from PIL import Image
import io

# Streamlit App Config
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_PATH = "best.onnx"
MODEL_ZIP_PATH = "best.zip"
MODEL_EXTRACTED_PATH = "best.onnx"
LOGO_PATH = "logo.png"
LABELS = ["NO Helmet", "ON. Helmet"]

# Unzip ONNX model if needed
if not os.path.exists(MODEL_EXTRACTED_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# Image Preprocessing
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Model Postprocessing
def postprocess(outputs, threshold=0.3):
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# Alarm
def play_alarm():
    st.warning("🚨 Violation detected! (Alarm sound not supported in browser)")

# Sidebar
st.sidebar.image(LOGO_PATH, use_column_width=True)
st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow; font-size: 36px;'>CapSure</h1>
    <h2 style='text-align:center; color:yellow; font-size: 20px;'>Real-time Helmet Compliance Detection</h2>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False, key="cam_toggle")
reset_trigger = st.sidebar.button("🔁 RESET")

# Init session state
if 'captured_image_bytes' not in st.session_state:
    st.session_state.captured_image_bytes = None

if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

# Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

frame_placeholder = st.empty()

# STEP 1: Capture image if camera is on
if start_camera and not st.session_state.violation:
    uploaded_image = st.camera_input("📸 Capture Image for Detection")

    if uploaded_image is not None:
        # Store image bytes in session state
        st.session_state.captured_image_bytes = uploaded_image.getvalue()
        st.success("✅ Image captured. Click below to run detection.")

# STEP 2: Run detection only when button is clicked
if st.session_state.captured_image_bytes and not st.session_state.violation:
    run_detection = st.button("🧠 Run Detection")

    if run_detection:
        image = Image.open(io.BytesIO(st.session_state.captured_image_bytes)).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img_input = preprocess(frame)
        outputs = session.run(None, {input_name: img_input})
        detections = postprocess(outputs)

        alert_triggered = False

        for cls_id, conf, (x1, y1, x2, y2) in detections:
            label = LABELS[cls_id]
            color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if label == "NO Helmet":
                alert_triggered = True

        if alert_triggered:
            play_alarm()
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            formatted_time = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"

            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            # Save to defect log
            st.session_state.history.insert(0, {
                "timestamp": formatted_time,
                "class": "NO Helmet",
                "filename": filename,
                "image_bytes": img_bytes
            })

            st.session_state.violation = True

            st.warning("🚨 Violation Detected! Please RESET to continue.")
            st.download_button(
                label="⬇️ Download Violation Snapshot",
                data=img_bytes,
                file_name=filename,
                mime="image/jpeg"
            )

        # Show image with detection
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

# RESET logic
if reset_trigger:
    st.session_state.violation = False
    st.session_state.captured_image_bytes = None
    st.rerun()

# Defect Log
st.markdown("---")
st.markdown("## 📋 Defect Log (Recent Violations)")

if st.session_state.history:
    for idx, entry in enumerate(st.session_state.history):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{idx+1}.** `{entry['timestamp']}` — `{entry['class']}`")
        with col2:
            st.download_button(
                label="⬇️ Download Image",
                data=entry['image_bytes'],
                file_name=entry['filename'],
                mime="image/jpeg",
                key=f"download_{idx}"
            )
else:
    st.info("No helmet violations recorded yet.")
