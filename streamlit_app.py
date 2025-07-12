import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from PIL import Image
import zipfile

# --- CONFIGURATION ---
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")
MODEL_ZIP_PATH = "best.zip"
MODEL_PATH = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]
LOGO_PATH = "logo.png"

# --- EXTRACT MODEL ---
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# --- IMAGE PREPROCESSING ---
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)  # HWC → CHW
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0), img_resized

# --- POSTPROCESSING ---
def postprocess(outputs, threshold=0.3):
    preds = outputs[0][0]
    results = []
    for pred in preds:
        if len(pred) >= 6:
            x1, y1, x2, y2, conf, cls = pred[:6]
            if conf > threshold:
                results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# --- SIDEBAR ---
st.sidebar.image(LOGO_PATH, use_column_width=True)
st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow;'>CapSure</h1>
    <h2 style='text-align:center; color:yellow;'>Helmet Detection</h2>
    """, unsafe_allow_html=True
)
st.sidebar.markdown("---")
start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False)

# --- MAIN UI ---
st.title("🪖 CapSure - Helmet Detection System")
st.markdown("---")

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

if "violation" not in st.session_state:
    st.session_state.violation = False

# --- CAMERA INPUT ---
if start_camera and not st.session_state.violation:
    img_file = st.camera_input("📸 Take a photo")
    if img_file:
        img_pil = Image.open(img_file)
        frame_rgb = np.array(img_pil)
        input_tensor, display_frame = preprocess_image(frame_rgb)
        outputs = session.run(None, {input_name: input_tensor})
        detections = postprocess(outputs)

        alert = False
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            label = LABELS[cls_id]
            color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if label == "NO Helmet":
                alert = True

        st.image(display_frame, channels="BGR", caption="Processed Image")

        if alert:
            st.warning("🚨 Violation detected!")
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            timestamp = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            _, img_bytes = cv2.imencode(".jpg", display_frame)

            st.session_state.history.insert(0, {
                "timestamp": timestamp,
                "class": "NO Helmet",
                "filename": filename,
                "image_bytes": img_bytes.tobytes()
            })
            st.session_state.violation = True

            st.download_button("⬇️ Download Violation Snapshot", img_bytes.tobytes(), filename, "image/jpeg")

# --- RESET ---
if st.sidebar.button("🔁 RESET"):
    st.session_state.violation = False
    st.rerun()

# --- DEFECT LOG ---
st.markdown("---")
st.subheader("📋 Defect Log (Recent Violations)")

if st.session_state.history:
    for idx, entry in enumerate(st.session_state.history):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"🕒 **{entry['timestamp']}**")
        with col2:
            st.markdown(f"🚧 **{entry['class']}**")
        with col3:
            st.download_button("Download", entry["image_bytes"], entry["filename"], "image/jpeg", key=f"dl_{idx}")
else:
    st.info("No violations recorded.")
