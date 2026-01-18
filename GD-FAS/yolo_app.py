import os
import sys

# --- CRITICAL FIX: PREVENT DLL CONFLICTS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT ORDER IS IMPORTANT ---
import torch
import cv2
import numpy as np

from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame, QMessageBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
from torchvision import transforms
from PIL import Image

# --- PROJECT IMPORTS ---
try:
    from models import build_model
except ImportError:
    try:
        from models.networks import build_model
    except ImportError:
        build_model = None


# --- CONFIGURATION ---
class Config:
    def __init__(self):
        self.backbone = 'clip'
        self.gs = True
        self.num_classes = 2
        self.temperature = 0.1
        self.lambda_1 = 0.1
        self.lambda_2 = 0.1
        self.lambda_3 = 0.1
        self.num_domain = 1
        self.beta = 1.5
        self.params = [1.0, 0.8, 0.1, 1.0]
        self.protocol = "Custom_to_Custom"


# --- PREPROCESSING ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])


# --- WORKER THREAD ---
class InferenceWorker(QThread):
    frame_processed = Signal(object, str, float, str)

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.transform = get_transform()
        self.current_frame = None
        self.is_running = True
        self.yolo = None

    def run(self):
        if self.yolo is None:
            print("Loading YOLOv8n on CPU (Safe Mode)...")
            try:
                # Load YOLO
                self.yolo = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"❌ YOLO Load Error: {e}")
                return

        while self.is_running:
            if self.current_frame is not None:
                try:
                    frame = self.current_frame.copy()

                    # --- FIX: FORCE YOLO TO RUN ON CPU ---
                    # This avoids the 'torchvision::nms' CUDA crash
                    results = self.yolo(frame, verbose=False, classes=[0], device='cpu')

                    detected = False
                    label_text = "NO FACE"
                    conf_value = 0.0
                    box_color = "#808080"  # Grey

                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        largest_area = 0
                        best_box = None

                        for box in boxes:
                            # Use .cpu() to ensure coordinates are safe to use
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            area = (x2 - x1) * (y2 - y1)
                            if area > largest_area:
                                largest_area = area
                                best_box = (x1, y1, x2, y2)

                        if best_box:
                            x1, y1, x2, y2 = best_box

                            h, w, _ = frame.shape
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            if x2 > x1 and y2 > y1:
                                detected = True

                                # Crop Face
                                face_crop = frame[y1:y2, x1:x2]

                                # --- RUN FAS MODEL ON GPU (This works fine!) ---
                                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(rgb_crop)

                                # Transform on CPU, then move to GPU
                                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                                with torch.no_grad():
                                    outputs = self.model(input_tensor)
                                    if isinstance(outputs, tuple):
                                        cls_score = outputs[0]
                                    else:
                                        cls_score = outputs

                                    probs = torch.softmax(cls_score, dim=1)
                                    conf, preds = torch.max(probs, 1)
                                    conf_value = conf.item()

                                    if preds.item() == 1:
                                        label_text = "LIVE"
                                        box_color = "#00FF00"
                                        cv_color = (0, 255, 0)
                                    else:
                                        label_text = "ATTACK"
                                        box_color = "#FF0000"
                                        cv_color = (0, 0, 255)

                                cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)
                                cv2.putText(frame, f"{label_text}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, cv_color, 2)

                    self.frame_processed.emit(frame, label_text, conf_value, box_color)
                    self.current_frame = None

                except Exception as e:
                    print(f"Processing Error: {e}")

            self.msleep(30)

    def update_frame(self, frame):
        self.current_frame = frame

    def stop(self):
        self.is_running = False
        self.wait()


# --- MAIN WINDOW ---
class FASWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FAS System (Hybrid CPU/GPU)")
        self.setGeometry(100, 100, 1000, 700)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FAS Model will run on: {self.device}")

        self.model = self.load_model()
        if not self.model:
            sys.exit(1)

        self.worker = InferenceWorker(self.model, self.device)
        self.worker.frame_processed.connect(self.update_ui)
        self.worker.start()

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_stream)

        self.init_ui()

    def load_model(self):
        args = Config()
        if build_model is None:
            QMessageBox.critical(self, "Error", "Could not import build_model.")
            return None

        try:
            model = build_model(args)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model build failed: {e}")
            return None

        model_path = "results/Run_Fold1/Custom_to_Custom_best.pth"
        if os.path.exists(model_path):
            try:
                print(f"Loading weights from {model_path}...")
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    elif 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    model.load_state_dict(checkpoint.state_dict(), strict=False)
                print("✅ FAS Model loaded successfully!")
            except Exception as e:
                print(f"❌ Load Error: {e}")
        else:
            print(f"⚠️ Warning: {model_path} not found.")

        model.to(self.device)
        model.eval()
        return model

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.video_label = QLabel("Click Webcam to Start")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; border-radius: 10px; color: #555;")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        self.result_frame = QFrame()
        self.result_frame.setStyleSheet("background-color: #333; border-radius: 8px; padding: 10px;")
        result_layout = QHBoxLayout(self.result_frame)

        self.status_label = QLabel("READY")
        self.status_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white;")
        result_layout.addWidget(self.status_label)

        self.conf_label = QLabel("--")
        self.conf_label.setFont(QFont("Arial", 16))
        self.conf_label.setStyleSheet("color: #ccc;")
        result_layout.addWidget(self.conf_label)

        layout.addWidget(self.result_frame)

        btn_layout = QHBoxLayout()
        self.btn_webcam = QPushButton("📷 Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_file = QPushButton("📂 File")
        self.btn_file.clicked.connect(self.open_file)
        self.btn_stop = QPushButton("🛑 Stop")
        self.btn_stop.clicked.connect(self.stop_stream)

        for btn in [self.btn_webcam, self.btn_file, self.btn_stop]:
            btn.setStyleSheet("padding: 12px; font-weight: bold; font-size: 14px;")
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

    def start_webcam(self):
        self.stop_stream()
        # TRY INDEX 0 FIRST, IF FAILS, TRY 1 (External Camera)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("⚠️ Camera 0 failed. Trying Camera 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open any webcam.\nCheck if Zoom/Teams is using it.")
                return
        self.timer.start(30)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Video (*.mp4 *.avi)")
        if file_path:
            self.stop_stream()
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def stop_stream(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Stopped")
        self.status_label.setText("WAITING")
        self.status_label.setStyleSheet("color: white;")
        self.result_frame.setStyleSheet("background-color: #333;")

    def process_video_stream(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.worker.update_frame(frame)
            else:
                self.stop_stream()

    @Slot(object, str, float, str)
    def update_ui(self, frame, label, conf, color_hex):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.status_label.setText(label)
        self.conf_label.setText(f"{conf * 100:.1f}%" if conf > 0 else "")

        if label == "NO FACE":
            bg, txt = "#555", "#FFF"
        elif label == "LIVE":
            bg, txt = "#d4edda", "#155724"
        else:
            bg, txt = "#f8d7da", "#721c24"

        self.status_label.setStyleSheet(f"color: {txt};")
        self.result_frame.setStyleSheet(f"background-color: {bg}; border: 2px solid {color_hex}; border-radius: 8px;")

    def closeEvent(self, event):
        self.stop_stream()
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FASWindow()
    window.show()
    sys.exit(app.exec())