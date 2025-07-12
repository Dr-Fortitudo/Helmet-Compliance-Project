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
import torch

# Config
st.set_page_config(page_title="CapSure", page_icon="🪖", layout="wide")
MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]

if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

@st.cache_resource
def load_model():
    s = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name

session, input_name = load_model()

def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    t = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(t, axis=0), img

def postprocess(output, conf_thres=0.3, iou_thres=0.4):
    import torchvision.ops as ops  # This works in Streamlit Cloud too

    preds = output[0]  # (num_boxes, 6+num_classes)
    boxes = []
    confidences = []
    class_ids = []

    for det in preds:
        x_center, y_center, width, height = det[:4]
        objectness = det[4]
        class_probs = det[5:]
        conf = objectness * np.max(class_probs)
        class_id = np.argmax(class_probs)

        if conf > conf_thres:
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(class_id)

    if len(boxes) == 0:
        return []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(confidences, dtype=torch.float32)
    keep = ops.nms(boxes, scores, iou_thres)

    results = []
    for i in keep:
        i = int(i)
        box = boxes[i].numpy().astype(int)
        results.append((class_ids[i], confidences[i], tuple(box)))
    return results


if "history" not in st.session_state:
    st.session_state.history = []
if "violated" not in st.session_state:
    st.session_state.violated = False

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
        inp, display = preprocess(frame)
        st.write("Tensor shape:", inp.shape)
        outs = session.run(None, {input_name: inp})
        st.write("Model output sample:", outs[0][:2])
        det = postprocess(outs)
        alert = False
        for clsid, conf, (x1, y1, x2, y2) in det:
            label = LABELS[clsid]
            color = (0,255,0) if clsid else (0,0,255)
            cv2.rectangle(display, (x1,y1),(x2,y2), color, 2)
            cv2.putText(display, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if label == "NO Helmet":
                alert = True
        st.image(display, channels="BGR")
        if alert:
            st.warning("🚨 Violation")
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            ts = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            fn = f"viol_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            _, b = cv2.imencode('.jpg', display)
            st.session_state.history.insert(0, {"ts": ts, "class": "NO Helmet", "bytes": b.tobytes(), "fn": fn})
            st.session_state.violated = True
            st.download_button("Download shot", data=b.tobytes(), file_name=fn, mime="image/jpeg")

st.markdown("---\n## Defect log")
for idx, h in enumerate(st.session_state.history):
    c1, c2, c3 = st.columns([2,2,1])
    with c1: st.write(h["ts"])
    with c2: st.write(h["class"])
    with c3: st.download_button("Download", data=h["bytes"], file_name=h["fn"], mime="image/jpeg", key=idx)
