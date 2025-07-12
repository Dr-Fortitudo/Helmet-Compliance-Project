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

# Unzip if needed
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

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

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("CapSure")
start = st.sidebar.checkbox("📷 Camera ON/OFF")

st.title("🪖 Helmet Detection System")

# Camera input
if start:
    img_file = st.camera_input("📸 Capture Image")

    if img_file:
        img_pil = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        inp = preprocess(frame)
        outs = session.run(None, {input_name: inp})
        det = postprocess(outs)

        display = frame.copy()
        alert = False

        for clsid, conf, (x1, y1, x2, y2) in det:
            label = LABELS[clsid]
            color = (0, 255, 0) if clsid else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if clsid == 0:
                alert = True

        st.image(display, channels="BGR")

        if alert:
            st.warning("🚨 Violation Detected!")
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            ts = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            fn = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            _, img_bytes = cv2.imencode('.jpg', display)

            st.session_state.history.insert(0, {
                "ts": ts, "class": "NO Helmet", "bytes": img_bytes.tobytes(), "fn": fn
            })

            st.download_button("⬇️ Download Violation Snapshot", data=img_bytes.tobytes(), file_name=fn, mime="image/jpeg")

# Log
st.markdown("---")
st.subheader("📋 Defect Log")

if st.session_state.history:
    for idx, h in enumerate(st.session_state.history):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.write("🕒", h["ts"])
        with c2:
            st.write("🪖", h["class"])
        with c3:
            st.download_button("Download", data=h["bytes"], file_name=h["fn"], mime="image/jpeg", key=idx)
else:
    st.info("No helmet violations recorded yet.")
