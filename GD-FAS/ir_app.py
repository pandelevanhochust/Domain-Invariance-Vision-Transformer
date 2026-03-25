import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame, QMessageBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

# --- IMPORT MODEL BUILDER ---
try:
    from models import build_model
except ImportError:
    try:
        from models.networks import build_model
    except ImportError:
        print("❌ Error: Could not import build_model. Run this from the project root.")
        build_model = None


# --- CONFIGURATION (MATCHING YOUR IR-ONLY TRAINING) ---
class Config:
    def __init__(self):
        self.backbone = 'clip'  # <--- Single Stream
        self.num_classes = 2
        self.num_domain = 1  # <--- Single Domain
        # Dummy args required by build_model
        self.gs = True
        self.temperature = 0.1
        self.lambda_1 = 0.1
        self.lambda_2 = 0.1
        self.lambda_3 = 0.1
        self.beta = 1.5
        self.params = [1.0, 0.8, 0.1, 1.0]
        self.protocol = "CASIA_IR"


# --- PREPROCESSING ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # CLIP Standard Normalization
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])


# --- INFERENCE WORKER ---
class IRInferenceWorker(QThread):
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
        # Load Face Detector
        if self.yolo is None:
            w = "yolov8n-face.pt" if os.path.exists("yolov8n-face.pt") else "yolov8n.pt"
            print(f"Loading Face Detector: {w}")
            try:
                self.yolo = YOLO(w)
            except Exception as e:
                print(f"YOLO Error: {e}")
                return

        while self.is_running:
            if self.current_frame is not None:
                try:
                    frame = self.current_frame.copy()

                    # 1. Detect Face
                    results = self.yolo(frame, verbose=False, conf=0.5, device='cpu')

                    label_text = "NO FACE"
                    conf_value = 0.0
                    box_color = "#555"

                    if len(results) > 0 and len(results[0].boxes) > 0:
                        # Find Largest Face
                        boxes = results[0].boxes
                        best_box = max(boxes,
                                       key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

                        # Clamp Coordinates
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # 2. Crop Face
                        face_crop = frame[y1:y2, x1:x2]

                        # 3. Preprocess for IR Model
                        # Even though it's IR, CLIP expects 3 channels.
                        # If input is B&W, we convert Gray->RGB (replicates channels)
                        if len(face_crop.shape) == 2:
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
                        else:
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                        pil_img = Image.fromarray(face_rgb)
                        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                        # 4. Inference
                        with torch.no_grad():
                            # Single Stream returns tuple (logits, features)
                            logits, _ = self.model(input_tensor)
                            probs = torch.softmax(logits, dim=1)
                            conf, preds = torch.max(probs, 1)
                            conf_value = conf.item()

                            # Label 0 = Live, 1 = Spoof (Standard CASIA Protocol)
                            if preds.item() == 0:
                                label_text = "LIVE"
                                box_color = "#00FF00"  # Green
                                cv_color = (0, 255, 0)
                            else:
                                label_text = "SPOOF"
                                box_color = "#FF0000"  # Red
                                cv_color = (0, 0, 255)

                        # Draw Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)
                        cv2.putText(frame, f"{label_text} ({conf_value:.2f})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cv_color, 2)

                    self.frame_processed.emit(frame, label_text, conf_value, box_color)
                    self.current_frame = None

                except Exception as e:
                    print(f"Inference Error: {e}")

            self.msleep(30)

    def update_frame(self, frame):
        self.current_frame = frame

    def stop(self):
        self.is_running = False
        self.wait()


# --- MAIN GUI ---
class FASApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR-Only FAS System")
        self.setGeometry(100, 100, 900, 700)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on: {self.device}")

        self.model = self.load_model()
        if not self.model: sys.exit(1)

        self.worker = IRInferenceWorker(self.model, self.device)
        self.worker.frame_processed.connect(self.update_ui)
        self.worker.start()

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_stream)

        self.init_ui()

    def load_model(self):
        args = Config()
        try:
            print("Building IR-Only Model (CLIP)...")
            model = build_model(args)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Build failed: {e}")
            return None

        # POINT THIS TO YOUR DOWNLOADED WEIGHTS
        # Assuming you copied it to results/Run_IR_Only/
        model_path = "results/Run_IR_Only/CASIA_IR_best.pth"

        if os.path.exists(model_path):
            try:
                print(f"Loading weights: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)

                # Flexible loading (handles 'state_dict' wrapper or direct dict)
                if isinstance(checkpoint, dict):
                    sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                else:
                    sd = checkpoint.state_dict()

                model.load_state_dict(sd, strict=False)
                print("✅ Weights Loaded Successfully")
            except Exception as e:
                print(f"❌ Weight Load Error: {e}")
                return None
        else:
            QMessageBox.warning(self, "Warning",
                                f"Weights not found at:\n{model_path}\n\nPlease download CASIA_IR_best.pth from Drive.")
            # For testing without weights, we return the random model (User will get random results)

        model.to(self.device)
        model.eval()
        return model

    def init_ui(self):
        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)

        # Video Display
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #000; border: 2px solid #333;")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # Result Panel
        res_frame = QFrame()
        res_frame.setStyleSheet("background: #222; border-radius: 8px; padding: 10px;")
        h_layout = QHBoxLayout(res_frame)

        self.lbl_status = QLabel("READY")
        self.lbl_status.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.lbl_status.setStyleSheet("color: white;")

        self.lbl_conf = QLabel("--")
        self.lbl_conf.setFont(QFont("Segoe UI", 18))
        self.lbl_conf.setStyleSheet("color: #aaa;")

        h_layout.addWidget(self.lbl_status)
        h_layout.addStretch()
        h_layout.addWidget(self.lbl_conf)
        layout.addWidget(res_frame)

        # Controls
        controls = QHBoxLayout()
        self.btn_cam = QPushButton("Start Camera")
        self.btn_cam.clicked.connect(self.start_cam)
        self.btn_file = QPushButton("Load Video")
        self.btn_file.clicked.connect(self.open_file)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_stream)

        for b in [self.btn_cam, self.btn_file, self.btn_stop]:
            b.setStyleSheet("padding: 10px; font-weight: bold;")
            controls.addWidget(b)

        layout.addLayout(controls)

    def start_cam(self):
        self.stop_stream()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try external cam if 0 fails
            self.cap = cv2.VideoCapture(1)
        self.timer.start(30)

    def open_file(self):
        self.stop_stream()
        f, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video (*.mp4 *.avi)")
        if f:
            self.cap = cv2.VideoCapture(f)
            self.timer.start(30)

    def stop_stream(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.lbl_status.setText("STOPPED")
        self.lbl_status.setStyleSheet("color: white;")

    def process_stream(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.worker.update_frame(frame)
            else:
                self.stop_stream()

    @Slot(object, str, float, str)
    def update_ui(self, frame, label, conf, color):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.lbl_status.setText(label)
        self.lbl_conf.setText(f"{conf * 100:.1f}%")

        if label == "LIVE":
            bg = "#155724"  # Green
        elif label == "SPOOF":
            bg = "#721c24"  # Red
        else:
            bg = "#222"

        self.lbl_status.setStyleSheet("color: white;")
        self.video_label.parentWidget().findChild(QFrame).setStyleSheet(
            f"background: {bg}; border: 2px solid {color}; border-radius: 8px;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FASApp()
    window.show()
    sys.exit(app.exec())