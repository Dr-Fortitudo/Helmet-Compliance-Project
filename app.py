import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import pandas as pd
from playsound import playsound
import threading
from pathlib import Path

# ---------------------- Configuration ----------------------
MODEL_PATH = "best.onnx"
ALARM_PATH = "alarm.mp3"
LOGO_PATH = "logo.png"
SAVE_DIR = Path.home() / "Documents" / "CapSureViolations"
LABELS = ["NO Helmet", "ON. Helmet"]

# Ensure violation folder exists
try:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[ERROR] Could not create folder: {e}")
    SAVE_DIR = Path.cwd() / "violations"
    SAVE_DIR.mkdir(exist_ok=True)

# ---------------------- Load ONNX Model ----------------------
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

def postprocess(outputs, threshold=0.3):
    preds = outputs[0][0]
    results = []
    for pred in preds:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

def play_alarm():
    threading.Thread(target=lambda: playsound(ALARM_PATH), daemon=True).start()

# ---------------------- Main GUI Class ----------------------
class CapSureApp:
    def _init_(self, root):
        self.root = root
        self.root.title("CapSure - Helmet Detection")
        self.root.geometry("1000x750")

        self.style = ttk.Style()
        self.dark_mode = False
        self.cap = None
        self.violated = False
        self.violation_log = []
        self.helmet_count = 0

        # Title
        tk.Label(root, text="CapSure - Helmet Detection System", font=("Arial", 22, "bold"), fg="#3ABEFF").pack(pady=10)

        # Video Preview
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Button Panel
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(btn_frame, text="▶ Start Camera", font=("Arial", 12), width=15,
                                   command=self.start_camera, bg="green", fg="white")
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop Camera", font=("Arial", 12), width=15,
                                  command=self.stop_camera, bg="red", fg="white")
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.export_btn = tk.Button(btn_frame, text="📤 Export Log to CSV", font=("Arial", 12), width=20,
                                    command=self.export_log, bg="#3ABEFF")
        self.export_btn.grid(row=0, column=2, padx=5)

        self.theme_btn = tk.Button(btn_frame, text="🌗 Toggle Theme", font=("Arial", 12),
                                   command=self.toggle_theme, width=18)
        self.theme_btn.grid(row=0, column=3, padx=5)

        self.reset_counter_btn = tk.Button(btn_frame, text="🧮 Reset Counters", font=("Arial", 12),
                                           command=self.reset_counters, width=18, bg="#d4af37")
        self.reset_counter_btn.grid(row=0, column=4, padx=5)

        # Status & Counter
        self.count_label = tk.Label(root, text="🟢 Helmet: 0 | 🔴 No Helmet: 0", font=("Arial", 14, "bold"))
        self.count_label.pack(pady=5)

        self.status_label = tk.Label(root, text="System Ready ✅", font=("Arial", 14), fg="green")
        self.status_label.pack(pady=5)

        # Violation Table
        log_frame = tk.LabelFrame(root, text="📋 Violations Log", font=("Arial", 13, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(log_frame, columns=("Time", "Class", "Path"), show="headings", height=8)
        self.tree.heading("Time", text="Timestamp")
        self.tree.heading("Class", text="Class")
        self.tree.heading("Path", text="Saved Image")
        self.tree.column("Time", width=250)
        self.tree.column("Class", width=100)
        self.tree.column("Path", width=450)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.update_frame()

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.video_label.config(image="")

    def reset_counters(self):
        self.helmet_count = 0
        self.count_label.config(text="🟢 Helmet: 0 | 🔴 No Helmet: 0")

    def export_log(self):
        if not self.violation_log:
            messagebox.showinfo("No Data", "No violations to export.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if save_path:
            df = pd.DataFrame(self.violation_log)
            df.to_csv(save_path, index=False)
            messagebox.showinfo("Exported", f"Log exported to:\n{save_path}")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.enable_dark_mode()
        else:
            self.enable_light_mode()

    def enable_light_mode(self):
        self.style.theme_use("default")
        self.style.configure("Treeview", background="white", foreground="black", rowheight=25)
        self.style.configure("Treeview.Heading", background="#3ABEFF", foreground="white", font=('Arial', 12, 'bold'))

    def enable_dark_mode(self):
        self.style.theme_use("default")
        self.style.configure("Treeview", background="#2e2e2e", foreground="white", rowheight=25)
        self.style.configure("Treeview.Heading", background="#555555", foreground="white", font=('Arial', 12, 'bold'))

    def reset_violation(self):
        self.violated = False
        self.status_label.config(text="System Ready ✅", fg="green")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return

            input_tensor = preprocess(frame)
            outputs = session.run(None, {input_name: input_tensor})
            detections = postprocess(outputs)

            alert = False

            for cls_id, conf, (x1, y1, x2, y2) in detections:
                label = LABELS[cls_id]
                color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)

                if cls_id == 0:
                    alert = True
                else:
                    self.helmet_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.count_label.config(text=f"🟢 Helmet: {self.helmet_count} | 🔴 No Helmet: {len(self.violation_log)}")

            if alert and not self.violated:
                self.violated = True
                now = datetime.now(ZoneInfo("Asia/Kolkata"))
                timestamp = now.strftime("%I:%M:%S %p @ %d %B, %Y")
                filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                violation_path = SAVE_DIR / filename

                try:
                    cv2.imwrite(str(violation_path), frame)
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
                    return

                self.violation_log.insert(0, {
                    "Time": timestamp,
                    "Class": "NO Helmet",
                    "Path": str(violation_path)
                })
                self.tree.insert("", 0, values=(timestamp, "NO Helmet", str(violation_path)))

                play_alarm()
                self.status_label.config(text="🚨 Violation Detected! Resetting...", fg="red")
                self.root.after(5000, self.reset_violation)
                messagebox.showwarning("Helmet Violation", f"🚨 NO Helmet Detected!\nSaved at:\n{violation_path}")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

            self.root.after(10, self.update_frame)

# ---------------------- App Launch ----------------------
if _name_ == "_main_":
    root = tk.Tk()
    root.state('zoomed')
    try:
        root.iconbitmap("logo.ico")
    except Exception as e:
        print(f"[INFO] Icon load skipped: {e}")
    app = CapSureApp(root)
    root.mainloop()
