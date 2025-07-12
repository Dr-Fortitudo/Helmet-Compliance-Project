import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
import zipfile

# Config
st.set_page_config(page_title="CapSure", page_icon="🪖", layout="wide")
MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]

# Extract ONNX model if not already extracted
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

@st.cache_resource
def load_model():
    s = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name

session, input_name = load_model()

# Preprocess image
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)  # HWC -> CHW
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Postprocess predictions
def postprocess(outputs, threshold=0.3):
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((
                int(cls),
                float(conf),
                (int(x1), int(y1), int(x2), int(y2))
            ))
    return results

# Init session state
if "history" not in st.session_state:
    st.session_state.history = []
if "violated" not in st.session_state:
    st.session_state.violated = False

# Sidebar
st.sidebar.title("CapSure")
start = st.sidebar.checkbox("Camera ON/OFF")
if st.sidebar.button("RESET"):
    st.session_state.violated = False

st.title("🪖 Helmet Detection")

if start and not st.session_state.violated:
    img_file = st.camera_input("Capture Image")
    if img_file:
        img_pil = Image.open(img_file)
        frame = np.array(img_pil)
        inp = preprocess(frame)

        # Run detection
        outs = session.run(None, {input_name: inp})
        detections = postprocess(outs)

        display = frame.copy()  # <-- FIXED: this was missing
        alert = False

        for clsid, conf, (x1, y1, x2, y2) in detections:
            label = LABELS[clsid]
            color = (0, 255, 0) if clsid else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if label == "NO Helmet":
                alert = True

        st.image(display, channels="BGR")

        if alert:
            st.warning("🚨 Violation Detected")
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            ts = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            fn = f"viol_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            _, b = cv2.imencode('.jpg', display)
            st.session_state.history.insert(0, {
                "ts": ts,
                "class": "NO Helmet",
                "bytes": b.tobytes(),
                "fn": fn
            })
            st.session_state.violated = True
            st.download_button("Download Snapshot", b.tobytes(), fn, mime="image/jpeg")

# Defect log
st.markdown("---\n## 📋 Defect Log")
for idx, h in enumerate(st.session_state.history):
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.write(h["ts"])
    with c2:
        st.write(h["class"])
    with c3:
        st.download_button("Download", data=h["bytes"], file_name=h["fn"],
                           mime="image/jpeg", key=f"dl_{idx}")
