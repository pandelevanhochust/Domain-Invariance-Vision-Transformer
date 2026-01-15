import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget)
from torchvision import transforms
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "results/CelebA_Spoof_clip/S_to_S_best.pth"  # Path to your trained .pth file
YOLO_MODEL = "yolov8n-face.pt"  # YOLO face detection model (or yolov8n.pt)
# Force CPU since CUDA is not properly configured
DEVICE = "cpu"
SKIP_FRAMES = 3  # Process every Nth frame for video/webcam
FACE_PADDING = 0.3  # Padding ratio for face crop (30%)
INPUT_SIZE = 224  # CLIP expects 224x224


# ============================================================================
# YOLO FACE DETECTION PREPROCESSING NODE
# ============================================================================
class YoloFaceDetector:
    """Detects faces using YOLO and crops them for the model."""

    def __init__(self, yolo_weights="yolov8n.pt", device="cpu"):
        """
        Args:
            yolo_weights: Path to YOLO model weights
            device: 'cuda' or 'cpu' (forced to 'cpu' to avoid CUDA errors)
        """
        # Force CPU to avoid CUDA/NMS errors
        self.device = "cpu"
        self.padding = FACE_PADDING

        try:
            # Initialize YOLO with explicit CPU device
            self.yolo = YOLO(yolo_weights)
            self.yolo.to('cpu')
            print(f"✓ YOLO loaded: {yolo_weights} (CPU mode)")
        except Exception as e:
            # Fallback to standard YOLOv8n
            print(f"⚠ Could not load {yolo_weights}, using yolov8n.pt. Error: {e}")
            self.yolo = YOLO("yolov8n.pt")
            self.yolo.to('cpu')

        # CLIP-style transforms (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def detect_and_crop_face(self, img_rgb):
        """
        Detect the largest face in the image and return cropped face.

        Args:
            img_rgb: numpy array (H, W, 3) in RGB format

        Returns:
            cropped_face: numpy array of face region, or None if no face detected
        """
        # Run YOLO detection - force device to CPU to avoid CUDA errors
        results = self.yolo(img_rgb, verbose=False, device='cpu')

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # Get bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return None

        # Find largest face (by area)
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[largest_idx].astype(int)

        # Apply padding
        h, w = img_rgb.shape[:2]
        face_w = x2 - x1
        face_h = y2 - y1

        pad_w = int(face_w * self.padding)
        pad_h = int(face_h * self.padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        # Crop face
        face_crop = img_rgb[y1:y2, x1:x2]

        return face_crop

    def preprocess(self, img_rgb):
        """
        Full preprocessing pipeline: detect face → crop → transform → tensor

        Args:
            img_rgb: numpy array (H, W, 3) in RGB format

        Returns:
            face_tensor: torch.Tensor (1, 3, 224, 224) ready for model, or None
        """
        face_crop = self.detect_and_crop_face(img_rgb)

        if face_crop is None or face_crop.size == 0:
            return None

        # Transform to tensor
        face_tensor = self.transform(face_crop)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        return face_tensor


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================
class AntiSpoofApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GD-FAS Face Anti-Spoofing Detector (ICCV 2025)")
        self.setGeometry(100, 100, 1100, 800)

        # State variables
        self.model = None
        self.face_detector = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.frame_counter = 0
        self.is_playing = False
        self.is_webcam = False
        self.is_video = False
        self.current_frame = None
        self.current_rotation = 0
        self.optimal_threshold = 0.5  # Default threshold

        # Build UI
        self.setup_ui()

        # Initialize model
        self.init_system()

    def setup_ui(self):
        """Build the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ===== LEFT PANEL =====
        left_panel = QVBoxLayout()

        # Status label
        self.lbl_status = QLabel("Status: Initializing...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: gray; font-size: 14px;")

        # Load file button
        self.btn_load_file = QPushButton("📂 Load Image/Video")
        self.btn_load_file.clicked.connect(self.load_media_file)
        self.btn_load_file.setEnabled(False)
        self.btn_load_file.setStyleSheet("padding: 10px; font-size: 14px;")

        # Webcam button
        self.btn_live_cam = QPushButton("📹 Switch to Live Camera")
        self.btn_live_cam.clicked.connect(self.start_webcam)
        self.btn_live_cam.setEnabled(False)
        self.btn_live_cam.setStyleSheet(
            "background-color: #6c757d; color: white; padding: 10px; font-size: 14px;"
        )

        # Rotation buttons
        rotate_layout = QHBoxLayout()
        self.btn_rot_left = QPushButton("↺ Rotate Left")
        self.btn_rot_left.clicked.connect(lambda: self.rotate_media(-90))
        self.btn_rot_left.setStyleSheet("padding: 8px;")

        self.btn_rot_right = QPushButton("↻ Rotate Right")
        self.btn_rot_right.clicked.connect(lambda: self.rotate_media(90))
        self.btn_rot_right.setStyleSheet("padding: 8px;")

        rotate_layout.addWidget(self.btn_rot_left)
        rotate_layout.addWidget(self.btn_rot_right)

        # Result display
        result_label = QLabel("<b>Detection Result:</b>")
        result_label.setStyleSheet("font-size: 14px; margin-top: 10px;")

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setStyleSheet(
            "font-size: 14px; padding: 15px; border: 2px solid #ddd; border-radius: 5px;"
        )
        self.result_box.setHtml(
            "<p style='color: gray; text-align: center;'>No detection yet</p>"
        )

        # Add to left panel
        left_panel.addWidget(self.lbl_status)
        left_panel.addWidget(self.btn_load_file)
        left_panel.addWidget(self.btn_live_cam)
        left_panel.addLayout(rotate_layout)
        left_panel.addWidget(result_label)
        left_panel.addWidget(self.result_box)
        left_panel.addStretch()

        # ===== RIGHT PANEL =====
        right_panel = QVBoxLayout()

        # Image display
        self.image_display = QLabel("Load Image/Video or Start Camera")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet(
            "border: 2px dashed #aaa; background-color: #000; "
            "color: #fff; font-size: 16px;"
        )
        self.image_display.setMinimumSize(600, 500)

        # Process button
        self.btn_process = QPushButton("▶ Play / Detect")
        self.btn_process.setStyleSheet(
            "background-color: #007bff; color: white; font-weight: bold; "
            "padding: 15px; font-size: 16px; border-radius: 5px;"
        )
        self.btn_process.clicked.connect(self.toggle_playback)
        self.btn_process.setEnabled(False)

        right_panel.addWidget(self.image_display)
        right_panel.addWidget(self.btn_process)

        # Combine panels
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 7)

    def init_system(self):
        """Initialize the face detection and anti-spoofing models."""
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            self.lbl_status.setText(f"Status: Model file not found!")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.result_box.setHtml(
                f"<p style='color: red;'><b>ERROR:</b> '{MODEL_PATH}' not found!<br>"
                f"Please train the model first or update MODEL_PATH in the code.</p>"
            )
            return

        try:
            self.lbl_status.setText("Status: Loading Models...")
            QApplication.processEvents()

            # 1. Load YOLO Face Detector
            print(f"Loading YOLO face detector...")
            self.face_detector = YoloFaceDetector(YOLO_MODEL, DEVICE)

            # 2. Load GD-FAS Model
            print(f"Loading GD-FAS model from {MODEL_PATH}...")
            # Force load to CPU and move all parameters to CPU
            self.model = torch.load(MODEL_PATH, map_location='cpu')
            self.model.to('cpu')  # Ensure all parameters are on CPU
            self.model.eval()
            
            # Additional safety: ensure CLIP model inside is also on CPU
            if hasattr(self.model, 'model'):
                self.model.model.to('cpu')
            
            print(f"✓ Model loaded successfully (CPU mode)")

            # Update UI
            self.lbl_status.setText(f"Status: Ready ({DEVICE.upper()})")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            self.btn_load_file.setEnabled(True)
            self.btn_live_cam.setEnabled(True)

            self.result_box.setHtml(
                f"<p style='color: green;'><b>✓ System Ready</b></p>"
                f"<p>Device: <b>{DEVICE.upper()}</b><br>"
                f"Model: <b>GD-FAS (CLIP-based)</b><br>"
                f"Face Detector: <b>YOLOv8</b></p>"
            )

        except Exception as e:
            self.lbl_status.setText("Status: Error Loading Models")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.result_box.setHtml(f"<p style='color: red;'><b>ERROR:</b><br>{str(e)}</p>")
            print(f"Initialization Error: {e}")
            import traceback
            traceback.print_exc()

    def rotate_media(self, angle):
        """Rotate the current frame by the specified angle."""
        self.current_rotation = (self.current_rotation + angle) % 360
        if self.current_frame is not None and not self.is_playing:
            rotated_frame = self.apply_rotation(self.current_frame)
            self.display_image(rotated_frame)

    def apply_rotation(self, img_array):
        """Apply the current rotation to an image array."""
        if self.current_rotation == 0:
            return img_array
        elif self.current_rotation == 90:
            return cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        elif self.current_rotation == 180:
            return cv2.rotate(img_array, cv2.ROTATE_180)
        elif self.current_rotation == 270:
            return cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img_array

    def load_media_file(self):
        """Load an image or video file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Media", "",
            "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv)"
        )

        if not file_name:
            return

        self.stop_playback()
        self.result_box.clear()
        self.is_webcam = False
        self.current_rotation = 0

        # Check if video or image
        if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.is_video = True
            self.cap = cv2.VideoCapture(file_name)

            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(self.current_frame)
                self.result_box.setHtml(
                    f"<p>✓ Loaded Video: <b>{os.path.basename(file_name)}</b></p>"
                )
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            else:
                self.result_box.setHtml("<p style='color: red;'>Failed to load video</p>")
                return
        else:
            # Load image
            self.is_video = False
            self.cap = None
            try:
                img = Image.open(file_name).convert('RGB')
                self.current_frame = np.array(img)
                self.display_image(self.current_frame)
                self.result_box.setHtml(
                    f"<p>✓ Loaded Image: <b>{os.path.basename(file_name)}</b></p>"
                )
            except Exception as e:
                self.result_box.setHtml(f"<p style='color: red;'>Error loading image: {e}</p>")
                return

        self.btn_process.setEnabled(True)
        self.btn_process.setText("▶ Play / Detect" if self.is_video else "🔍 Run Detection")

    def start_webcam(self):
        """Start webcam capture."""
        self.stop_playback()
        self.result_box.clear()
        self.is_webcam = True
        self.is_video = True
        self.current_rotation = 0
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.result_box.setHtml(
                "<p style='color: red;'>❌ Error: Could not open webcam.</p>"
            )
            return

        self.result_box.setHtml("<p style='color: green;'>✓ Webcam started</p>")
        self.start_playback()

    def toggle_playback(self):
        """Toggle video playback or run single image detection."""
        if self.is_video:
            if self.is_playing:
                self.stop_playback()
            else:
                self.start_playback()
        else:
            # Single image inference
            rotated = self.apply_rotation(self.current_frame)
            self.run_inference(rotated)

    def start_playback(self):
        """Start video/webcam playback."""
        if not self.cap:
            return
        self.is_playing = True
        self.btn_process.setText("⏸ Pause")
        self.timer.start(30)  # ~33 FPS

    def stop_playback(self):
        """Stop video/webcam playback."""
        self.is_playing = False
        self.timer.stop()
        if self.is_webcam:
            self.btn_process.setText("▶ Resume Camera")
        else:
            self.btn_process.setText("▶ Play / Detect")

    def next_frame(self):
        """Process the next frame from video/webcam."""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.is_webcam:
                print("Webcam disconnected.")
            self.stop_playback()
            if not self.is_webcam:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            return

        # Flip webcam for mirror effect
        if self.is_webcam:
            frame = cv2.flip(frame, 1)

        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rotated_frame = self.apply_rotation(self.current_frame)
        self.display_image(rotated_frame)

        # Run inference every N frames
        self.frame_counter += 1
        if self.frame_counter % SKIP_FRAMES == 0:
            self.run_inference(rotated_frame)

    def run_inference(self, img_rgb):
        """Run face anti-spoofing inference on the image."""
        if self.model is None or self.face_detector is None:
            return

        try:
            # 1. Detect and preprocess face
            face_tensor = self.face_detector.preprocess(img_rgb)

            if face_tensor is None:
                self.result_box.setHtml(
                    "<h3 style='color: orange; text-align: center;'>⚠️ No Face Detected</h3>"
                    "<p style='text-align: center;'>Please ensure face is visible and well-lit</p>"
                )
                return

            # 2. Run GD-FAS model inference
            with torch.no_grad():
                # Forward pass - returns (logits_per_image, classifier_output)
                logits_per_image, classifier_output = self.model(face_tensor)

                # Use CLIP-based logits (Image-Text similarity)
                probs = torch.nn.functional.softmax(logits_per_image, dim=1)

                spoof_score = probs[0][0].item()  # Spoof probability
                real_score = probs[0][1].item()  # Real probability

                # Determine prediction
                is_real = real_score > self.optimal_threshold
                confidence = real_score if is_real else spoof_score

                # 3. Display results
                if is_real:
                    status = "✅ REAL FACE"
                    color = "green"
                    advice = "This appears to be a genuine face."
                else:
                    status = "❌ SPOOF DETECTED"
                    color = "red"
                    advice = "Warning: This may be a presentation attack!"

                html = f"""
                <div style='text-align: center; padding: 10px;'>
                    <h2 style='color: {color}; margin: 10px 0;'>{status}</h2>
                    <div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                        <p style='font-size: 16px; margin: 5px 0;'>
                            <b>Confidence:</b> <span style='color: {color};'>{confidence * 100:.1f}%</span>
                        </p>
                        <p style='font-size: 14px; margin: 5px 0;'>
                            <b>Real Score:</b> {real_score * 100:.1f}%
                        </p>
                        <p style='font-size: 14px; margin: 5px 0;'>
                            <b>Spoof Score:</b> {spoof_score * 100:.1f}%
                        </p>
                    </div>
                    <p style='color: #666; font-size: 13px; margin-top: 10px;'>{advice}</p>
                </div>
                """

                self.result_box.setHtml(html)

        except Exception as e:
            print(f"Inference Error: {e}")
            import traceback
            traceback.print_exc()
            self.result_box.setHtml(
                f"<p style='color: red;'><b>Inference Error:</b><br>{str(e)}</p>"
            )

    def display_image(self, img_array):
        """Display an image array in the QLabel."""
        if img_array is None or img_array.size == 0:
            return

        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_display.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """Clean up resources on window close."""
        self.stop_playback()
        if self.cap:
            self.cap.release()
        event.accept()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = AntiSpoofApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
