import os
import cv2
import torch
import kagglehub
import numpy as np
import torch.nn as nn
import torch.optim as optim
import timm
import requests
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from plotter import save_training_graphs

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
MODEL_SAVE_PATH = "../trained_model/yolo-divt_combined_final.pth"
YOLO_WEIGHTS_PATH = "yolov8n-face.pt"

# Dataset Paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

MANUAL_PATH_95K = os.path.join(project_root, "datasets", "archive")
REAL_98K_PATH = os.path.join(project_root, "datasets", "98k_real")
FAS_30K_PATH = os.path.join(project_root, "datasets", "30k_fas")
UNIDATA_REAL_PATH = os.path.join(project_root, "datasets", "unidata_real")

# Class Mapping
CLASS_MAP = {
    'Selfies': 0, 'real': 0, 'live': 0, 'true': 0,
    '3D_paper_mask': 1, 'print': 1, 'photo': 1, 'paper': 1,
    'Cutout_attacks': 2, 'cutout': 2,
    'Latex_mask': 3, 'latex': 3,
    'Replay_display_attacks': 4, 'replay': 4, 'screen': 4, 'video': 4,
    'Replay_mobile_attacks': 5, 'mobile': 5,
    'Silicone_mask': 6, 'silicone': 6,
    'Textile': 7, 'cloth': 7,
    'Wrapped': 8
}


def ensure_yolo_weights(path):
    if not os.path.exists(path):
        print(f"[INFO] {path} not found. Downloading generic yolov8n-face model...")
        url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
        try:
            r = requests.get(url, allow_redirects=True)
            with open(path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"[ERROR] Could not download weights: {e}")


# --- 2. THE PREPROCESS NODE (UPDATED) ---
class YoloPreprocessNode:

    def __init__(self, weights_path):
        ensure_yolo_weights(weights_path)
        self.detector = YOLO(weights_path)

        # Standard Transforms for DiVT
        self.transform_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))

    # [CHANGE 1] Default expansion increased to 2.5 (Maximum Background)
    def get_crop_coords(self, box, img_w, img_h, expansion=2.5):
        # Box is [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Expand box to include background context
        max_side = max(w, h)
        new_size = int(max_side * expansion)

        x1_new = max(0, cx - new_size // 2)
        y1_new = max(0, cy - new_size // 2)
        x2_new = min(img_w, cx + new_size // 2)
        y2_new = min(img_h, cy + new_size // 2)

        return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

    def run(self, image_bgr):
        """
        Main pipeline function.
        1. Detect Faces
        2. Select Largest
        3. Check Background Margin (Skip if too small)
        4. Crop, Resize & Normalize
        """
        if image_bgr is None: return None

        try:
            # 1. Detect
            results = self.detector(image_bgr, verbose=False, conf=0.5)

            if not results or len(results[0].boxes) == 0:
                return None

            # 2. Select Largest Face
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = np.argmax(areas)
            best_box = boxes[best_idx]

            # 3. Calculate Crop with High Expansion (2.5x)
            h_img, w_img = image_bgr.shape[:2]
            nx1, ny1, nx2, ny2 = self.get_crop_coords(best_box, w_img, h_img, expansion=2.5)

            # [CHANGE 2] Validate Background Margin
            # Calculate raw face size
            face_w = best_box[2] - best_box[0]
            face_h = best_box[3] - best_box[1]
            max_face_dim = max(face_w, face_h)

            # Calculate actual cropped area size (it might be smaller due to image edges)
            actual_crop_dim = min(nx2 - nx1, ny2 - ny1)

            # CRITICAL CHECK:
            # If the final crop is not at least 1.5x larger than the face,
            # it means the face is too close to the edge (Marginal Background).
            # We return None to SKIP this frame.
            if actual_crop_dim < (max_face_dim * 1.5):
                # print("[DEBUG] Skipped frame: Insufficient background context")
                return None

            # 4. Crop & Process
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            crop = pil_img.crop((nx1, ny1, nx2, ny2))

            crop = self.resize(crop)
            tensor = self.to_tensor(crop)
            tensor = self.transform_norm(tensor)

            return tensor

        except Exception as e:
            return None


# --- 3. Dataset Class ---
class UnifiedSpoofDataset(Dataset):
    def __init__(self, root_dir, source_name="Generic", force_label=None):
        self.root_dir = root_dir
        self.source_name = source_name
        self.force_label = force_label
        self.samples = []

        # Initialize the Node
        self.preprocess_node = YoloPreprocessNode(YOLO_WEIGHTS_PATH)

        self._crawl()

    def _crawl(self):
        print(f"[INFO] Scanning {self.source_name} at: {self.root_dir}")
        valid_exts = ('.jpg', '.png', '.jpeg', '.mp4', '.avi')
        count = 0

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    if self.force_label is not None:
                        label_attack = self.force_label
                    else:
                        folder_name = os.path.basename(root).lower()
                        label_attack = -1

                        for key, val in CLASS_MAP.items():
                            if key.lower() in folder_name:
                                label_attack = val
                                break
                        if label_attack == -1:
                            if 'real' in folder_name or 'live' in folder_name:
                                label_attack = 0
                            else:
                                continue

                    is_real = 1.0 if label_attack == 0 else 0.0
                    self.samples.append((os.path.join(root, file), label_attack, is_real))
                    count += 1
        print(f"[INFO] -> Found {count} valid labeled samples in {self.source_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_attack, label_real = self.samples[idx]
        final_tensor = None

        if path.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0:
                attempts = 5
                for _ in range(attempts):
                    random_frame_idx = np.random.randint(0, total_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
                    ret, frame = cap.read()

                    if ret:
                        final_tensor = self.preprocess_node.run(frame)
                        # If run() returns None (due to marginal background), loop continues to try next frame
                        if final_tensor is not None:
                            break
            cap.release()
        else:
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                final_tensor = self.preprocess_node.run(img_bgr)

        # If None (YOLO failed OR Background was too small), skip file
        if final_tensor is None:
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        return final_tensor, torch.tensor(label_real, dtype=torch.float32), torch.tensor(label_attack, dtype=torch.long)


# --- 4. Model (DiVT) ---
class MultiTaskDiVT(nn.Module):
    def __init__(self, num_attack_classes=9, pretrained=True):
        super(MultiTaskDiVT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.liveness_head = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1),
                                           nn.Sigmoid())
        self.attack_head = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.5),
                                         nn.Linear(256, num_attack_classes))

    def forward(self, x):
        feat = self.backbone(x)
        return self.liveness_head(feat), self.attack_head(feat)


# --- 5. Main Execution ---
if __name__ == "__main__":
    datasets = []

    # 1. Load Datasets
    if os.path.exists(REAL_98K_PATH):
        datasets.append(UnifiedSpoofDataset(REAL_98K_PATH, "98K Dataset"))
    else:
        print(f"[WARN] Path not found: {REAL_98K_PATH}")

    if os.path.exists(MANUAL_PATH_95K):
        datasets.append(UnifiedSpoofDataset(MANUAL_PATH_95K, "95k Dataset"))
    else:
        print(f"[WARN] Path not found: {MANUAL_PATH_95K}")

    if os.path.exists(FAS_30K_PATH):
        datasets.append(UnifiedSpoofDataset(FAS_30K_PATH, "30K FAS Dataset"))
    else:
        print(f"[WARN] Path not found: {FAS_30K_PATH}")

    if os.path.exists(UNIDATA_REAL_PATH):
        datasets.append(UnifiedSpoofDataset(UNIDATA_REAL_PATH, "Unidata Dataset"))
    else:
        print(f"[WARN] Path not found: {UNIDATA_REAL_PATH}")

    # 2. Combine & Split
    if len(datasets) > 0:
        full_dataset = ConcatDataset(datasets)
        print(f"\n[INFO] TRAINING ON {len(full_dataset)} TOTAL SAMPLES")

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])

        # num_workers=0 is safest for YOLO preprocessing
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 3. Model Setup
        model = MultiTaskDiVT(num_attack_classes=9).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion_live = nn.BCELoss()
        criterion_attack = nn.CrossEntropyLoss()

        # History Tracking
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        print("[INFO] Starting Training Loop...")

        for epoch in range(NUM_EPOCHS):
            # --- TRAINING ---
            model.train()
            running_train_loss = 0.0

            for i, (imgs, lbl_real, lbl_attack) in enumerate(train_loader):
                imgs, lbl_real, lbl_attack = imgs.to(DEVICE), lbl_real.to(DEVICE).unsqueeze(1), lbl_attack.to(DEVICE)

                optimizer.zero_grad()
                pred_real, pred_attack = model(imgs)

                loss = (0.6 * criterion_live(pred_real, lbl_real)) + (0.4 * criterion_attack(pred_attack, lbl_attack))
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                if i % 10 == 0:
                    print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss.item():.4f}")

            avg_train_loss = running_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # --- VALIDATION ---
            model.eval()
            running_val_loss = 0.0
            correct_live = 0
            total_samples = 0

            with torch.no_grad():
                for imgs, lbl_real, lbl_attack in val_loader:
                    imgs, lbl_real, lbl_attack = imgs.to(DEVICE), lbl_real.to(DEVICE).unsqueeze(1), lbl_attack.to(
                        DEVICE)

                    pred_real, pred_attack = model(imgs)

                    loss = (0.6 * criterion_live(pred_real, lbl_real)) + (
                            0.4 * criterion_attack(pred_attack, lbl_attack))
                    running_val_loss += loss.item()

                    # Calculate Accuracy
                    preds = (pred_real > 0.5).float()
                    correct_live += (preds == lbl_real).sum().item()
                    total_samples += lbl_real.size(0)

            avg_val_loss = running_val_loss / len(val_loader)
            val_acc = (correct_live / total_samples) * 100

            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save Checkpoint & Update Graph
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_training_graphs(history)

        print("[INFO] Training Complete.")
    else:
        print("[ERROR] No datasets available.")