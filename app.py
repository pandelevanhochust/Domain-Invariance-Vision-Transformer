import sys
import os
import cv2
import torch
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
from torchvision import transforms
from PIL import Image

# --- PROJECT IMPORTS ---
# We assume this script is running from the GD-FAS root folder
from models import build_model


# --- CONFIGURATION CLASS ---
# We need to mimic the 'args' parser from training.py to build the model correctly
class Config:
    def __init__(self):
        self.backbone = 'clip'
        self.gs = True  # Gradient Scaling (used in training)
        self.num_classes = 2  # Live vs Attack
        # Add other defaults if your build_model requires them
        self.temperature = 0.1
        self.lambda_1 = 0.1
        self.lambda_2 = 0.1
        self.lambda_3 = 0.1


# --- PREPROCESSING ---
# Standard CLIP transformation
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])


# --- WORKER THREAD FOR INFERENCE ---
class InferenceWorker(QThread):
    frame_processed = Signal(object, str, float)  # Image, Label, Confidence

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.transform = get_transform()
        self.current_frame = None
        self.is_running = True

    def update_frame(self, frame):
        self.current_frame = frame

    def run(self):
        while self.is_running:
            if self.current_frame is not None:
                try:
                    # 1. Prepare Frame
                    # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
                    rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)

                    # 2. Transform
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                    # 3. Inference
                    with torch.no_grad():
                        # The model likely returns (cls_score, domain_score) or similar
                        # We only care about the first output (class score)
                        outputs = self.model(input_tensor)

                        # Handle different output formats from your specific model structure
                        if isinstance(outputs, tuple):
                            cls_score = outputs[0]
                        else:
                            cls_score = outputs

                        # Apply Softmax to get probabilities
                        probs = torch.softmax(cls_score, dim=1)
                        conf, preds = torch.max(probs, 1)

                        # 0 = Attack, 1 = Live (Check your dataset mapping! Usually 1 is Live)
                        # Adjust this mapping based on your specific training labels
                        label = "LIVE" if preds.item() == 1 else "ATTACK"
                        confidence = conf.item()

                    # 4. Emit Result
                    self.frame_processed.emit(self.current_frame, label, confidence)
                    self.current_frame = None  # Reset to wait for next frame

                except Exception as e:
                    print(f"Inference Error: {e}")

            self.msleep(30)  # Prevent CPU hogging

    def stop(self):
        self.is_running = False
        self.wait()


# --- MAIN GUI WINDOW ---
class FASWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FAS Detection System - PySide6")
        self.setGeometry(100, 100, 1000, 700)

        # Initialize Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Model on {self.device}...")
        self.model = self.load_model()
        self.worker = InferenceWorker(self.model, self.device)
        self.worker.frame_processed.connect(self.update_ui)
        self.worker.start()

        # Camera Setup
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_stream)

        self.init_ui()

    def load_model(self):
        args = Config()
        model = build_model(args)

        # --- PATH TO YOUR TRAINED MODEL ---
        # Update this path if your model is elsewhere
        model_path = "results/Run_Fold1/Custom_to_Custom_best.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle if state_dict is nested under 'state_dict' or 'model' key
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print("Model weights loaded successfully.")
        else:
            print(f"WARNING: Model file not found at {model_path}. Running with random weights.")

        model.to(self.device)
        model.eval()
        return model

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. Video Display
        self.video_label = QLabel("No Input")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #2b2b2b; color: white; border-radius: 10px;")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # 2. Result Display
        self.result_frame = QFrame()
        self.result_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 8px; padding: 10px;")
        result_layout = QHBoxLayout(self.result_frame)

        self.status_label = QLabel("WAITING")
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.status_label)

        self.conf_label = QLabel("Conf: 0.00%")
        self.conf_label.setFont(QFont("Arial", 16))
        result_layout.addWidget(self.conf_label)

        layout.addWidget(self.result_frame)

        # 3. Controls
        btn_layout = QHBoxLayout()

        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_webcam.setStyleSheet("padding: 10px; font-size: 14px;")

        self.btn_file = QPushButton("Open File (Img/Video)")
        self.btn_file.clicked.connect(self.open_file)
        self.btn_file.setStyleSheet("padding: 10px; font-size: 14px;")

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_stop.setStyleSheet("padding: 10px; font-size: 14px; background-color: #ffcccc;")

        btn_layout.addWidget(self.btn_webcam)
        btn_layout.addWidget(self.btn_file)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

    def start_webcam(self):
        self.stop_stream()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # 30ms ~ 33fps

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image or Video", "",
                                                   "Files (*.mp4 *.avi *.jpg *.png *.jpeg)")
        if file_path:
            self.stop_stream()
            if file_path.endswith(('.jpg', '.png', '.jpeg')):
                # Static Image
                frame = cv2.imread(file_path)
                if frame is not None:
                    self.worker.update_frame(frame)
            else:
                # Video File
                self.cap = cv2.VideoCapture(file_path)
                self.timer.start(30)

    def stop_stream(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Stopped")
        self.status_label.setText("WAITING")
        self.status_label.setStyleSheet("color: black;")

    def process_video_stream(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Send frame to worker for inference
                self.worker.update_frame(frame)
            else:
                self.stop_stream()

    @Slot(object, str, float)
    def update_ui(self, frame, label, conf):
        # 1. Update Video Feed (Convert OpenCV BGR to Qt Pixmap)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        # Scale to fit label maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

        # 2. Update Result Labels
        self.status_label.setText(label)
        self.conf_label.setText(f"Conf: {conf * 100:.2f}%")

        if label == "LIVE":
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.result_frame.setStyleSheet("background-color: #d4edda; border: 2px solid green; border-radius: 8px;")
        else:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.result_frame.setStyleSheet("background-color: #f8d7da; border: 2px solid red; border-radius: 8px;")

    def closeEvent(self, event):
        self.stop_stream()
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FASWindow()
    window.show()
    sys.exit(app.exec())