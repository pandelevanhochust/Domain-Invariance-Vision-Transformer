import os
import cv2
import torch
import kagglehub
import numpy as np
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN  # Import Face Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
MODEL_SAVE_PATH = "../trained_model/mtcnn-divt_combined_model.pth"

CLASS_MAP = {
    'Selfies': 0,  # REAL
    '3D_paper_mask': 1,
    'Cutout_attacks': 2,
    'Latex_mask': 3,
    'Replay_display_attacks': 4,
    'Replay_mobile_attacks': 5,
    'Silicone_mask': 6,
    'Textile 3D Face Mask Atta': 7,
    'Wrapped_3D_paper_mask': 8
}

ID_TO_NAME = {v: k for k, v in CLASS_MAP.items()}


# --- 2. Dataset Class (With MTCNN Face Detection) ---
class FullSpoofDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        # --- Initialize MTCNN ---
        # image_size=224: Crops and resizes face to 224x224 automatically
        # margin=40: Adds background context (chin/ears) which is crucial for seeing mask edges
        # post_process=False: Returns raw RGB tensor (0-255) so we can normalize it ourselves
        self.mtcnn = MTCNN(
            image_size=224,
            margin=40,
            keep_all=False,
            select_largest=True,
            post_process=False,
            device=DEVICE
        )

        # Fallback transform if NO face is detected
        self.transform_fallback = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Normalization for detected faces (since MTCNN returns tensor)
        self.transform_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self._crawl()

    def _crawl(self):
        valid_exts = ('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')
        print(f"[INFO] Scanning dataset at: {self.root_dir}")

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    folder_name = os.path.basename(root)
                    label_attack = -1

                    if folder_name in CLASS_MAP:
                        label_attack = CLASS_MAP[folder_name]
                    else:
                        for key, val in CLASS_MAP.items():
                            if key in folder_name:
                                label_attack = val
                                break

                    if label_attack != -1:
                        is_real = 1.0 if label_attack == 0 else 0.0
                        self.samples.append((os.path.join(root, file), label_attack, is_real))

        print(f"[INFO] Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_attack, label_real = self.samples[idx]
        img = None

        # A. Load Image / Video Frame
        if path.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                # Random frame sampling
                cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total_frames))
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
            cap.release()
        else:
            try:
                img = Image.open(path).convert('RGB')
            except:
                pass

        # Handle corrupt files
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        # B. Detect and Crop Face
        # MTCNN returns a tensor if face found, None if not
        try:
            face_tensor = self.mtcnn(img)
        except RuntimeError:
            # Catch sporadic MTCNN errors
            face_tensor = None

        if face_tensor is not None:
            # MTCNN (post_process=False) returns [3, 224, 224] in range 0-255
            # We convert to float 0-1 and normalize
            face_tensor = face_tensor.float() / 255.0
            face_tensor = self.transform_norm(face_tensor)
        else:
            # Fallback: Use whole image if no face detected
            face_tensor = self.transform_fallback(img)

        # Ensure tensor is on CPU for DataLoader collation (moved to GPU in loop)
        return face_tensor.detach(), torch.tensor(label_real, dtype=torch.float32), torch.tensor(label_attack,
                                                                                                 dtype=torch.long)


# --- 3. Model Architecture (DiVT) ---
class MultiTaskDiVT(nn.Module):
    def __init__(self, num_attack_classes=9, pretrained=True):
        super(MultiTaskDiVT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features

        self.liveness_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 1), nn.Sigmoid()
        )

        self.attack_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_attack_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        liveness_prob = self.liveness_head(features)
        attack_logits = self.attack_head(features)
        return liveness_prob, attack_logits


# --- 4. Training Function ---
def train_model(model, dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Important: num_workers=0 is SAFER when using MTCNN inside the dataset on Windows
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_live = nn.BCELoss()
    criterion_attack = nn.CrossEntropyLoss()

    print(f"\n[INFO] Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (imgs, lbl_real, lbl_attack) in enumerate(train_loader):
            imgs, lbl_real, lbl_attack = imgs.to(DEVICE), lbl_real.to(DEVICE).unsqueeze(1), lbl_attack.to(DEVICE)

            optimizer.zero_grad()
            pred_real, pred_attack = model(imgs)

            loss = (0.6 * criterion_live(pred_real, lbl_real)) + (0.4 * criterion_attack(pred_attack, lbl_attack))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss.item():.4f}")

        model.eval()
        correct_live = 0
        total = 0
        with torch.no_grad():
            for imgs, lbl_real, _ in val_loader:
                imgs = imgs.to(DEVICE)
                lbl_real = lbl_real.to(DEVICE).unsqueeze(1)
                pred_real, _ = model(imgs)
                predicted = (pred_real > 0.5).float()
                correct_live += (predicted == lbl_real).sum().item()
                total += lbl_real.size(0)

        print(f"Epoch {epoch + 1} Complete. Val Accuracy (Real/Fake): {100 * correct_live / total:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")


# --- 5. Inference / Testing Function (Updated for MTCNN) ---
def predict_file(model, file_path):
    model.eval()

    # Initialize MTCNN for inference
    mtcnn = MTCNN(image_size=224, margin=40, keep_all=False, post_process=False, device=DEVICE)

    # Transforms
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    fallback_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Load Media
    img = None
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        try:
            img = Image.open(file_path).convert('RGB')
        except Exception as e:
            print(f"Error opening file: {e}")
            return

    if img is None:
        print("Could not load image data.")
        return

    # Detect & Crop
    print(f"[INFO] Processing {os.path.basename(file_path)}...")
    face_tensor = mtcnn(img)

    if face_tensor is not None:
        print("[INFO] Face detected. Cropping...")
        # Normalize: MTCNN (0-255) -> Float (0-1) -> Normalize
        face_tensor = face_tensor.float() / 255.0
        input_tensor = normalize(face_tensor).unsqueeze(0).to(DEVICE)
    else:
        print("[WARN] No face detected! Using full image (Confidence may be lower).")
        input_tensor = fallback_transform(img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        liveness_prob, attack_logits = model(input_tensor)

        is_real = liveness_prob.item() > 0.5
        confidence = liveness_prob.item() if is_real else 1 - liveness_prob.item()
        attack_idx = torch.argmax(attack_logits, dim=1).item()
        attack_name = ID_TO_NAME.get(attack_idx, "Unknown")

    print("\n" + "=" * 30)
    if is_real:
        print(f"✅ RESULT: REAL FACE (Selfie)")
        print(f"   Confidence: {confidence * 100:.2f}%")
    else:
        print(f"⚠️ RESULT: FAKE DETECTED")
        print(f"   Attack Type: {attack_name}")
        print(f"   Confidence: {confidence * 100:.2f}%")
    print("=" * 30 + "\n")


# --- 6. Main Function ---
if __name__ == "__main__":
    # A. Download Dataset
    print("[INFO] Checking Dataset...")
    try:
        path = kagglehub.dataset_download("axondata/face-anti-spoofing-dataset")
    except:
        path = "dataset"  # Fallback

    # B. Initialize Model
    model = MultiTaskDiVT(num_attack_classes=9).to(DEVICE)

    # C. Train or Load
    # If you want to FORCE retraining to use the new cropped faces,
    # delete your old .pth file or rename it!
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Found saved model at {MODEL_SAVE_PATH}. Loading weights...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

        # --- TRAINING MODE TRIGGER ---
        # Uncomment the next two lines if you want to CONTINUE training the loaded model
        # dataset = FullSpoofDataset(path)
        # train_model(model, dataset)
    else:
        print("[INFO] No saved model found. Starting training with MTCNN...")
        dataset = FullSpoofDataset(path)
        train_model(model, dataset)

    # D. Test
    predict_file(model, "../test/screen.mov")