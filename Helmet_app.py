import cv2
import numpy as np
import onnxruntime as ort
import time
import subprocess
import os
from datetime import datetime

# Constants
MODEL_PATH = "best.onnx"
ALARM_PATH = "alarm.mp3"
LABELS = ["NO Helmet", "ON Helmet"]
CONFIDENCE_THRESHOLD = 0.7
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
SAVE_DIR = "folder path"

# Make sure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs):
    predictions = outputs[0][0]  # Output shape is (1, N, 6)
    results = []
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > CONFIDENCE_THRESHOLD:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

def play_alarm():
    subprocess.Popen(["mpg123", ALARM_PATH],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def capture_and_infer():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("🎥 Starting Helmet Detection. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            original_frame = frame.copy()  # Keep original for saving
            img_input = preprocess(frame)
            outputs = session.run(None, {input_name: img_input})
            detections = postprocess(outputs)

            alert_triggered = False

            for cls_id, conf, (x1, y1, x2, y2) in detections:
                label = LABELS[cls_id]
                color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if label == "NO Helmet":
                    alert_triggered = True

            # Show the frame with boxes
            cv2.imshow("Helmet Detection", frame)

            if alert_triggered:
                print("🚨 Incompliant Worker Detected")
                play_alarm()

                # Save violation image with drawn boxes
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"violation_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📸 Saved violation image to {save_path}")

                print("Press any key to stop the alarm...")
                while True:
                    if cv2.waitKey(1) != -1:
                        break

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 'q' pressed. Exiting detection.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_infer()
