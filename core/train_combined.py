import os
import cv2
import torch
import kagglehub
import numpy as np
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
MODEL_SAVE_PATH = "../trained_model/mtcnn-divt_combined_model.pth"

# !!! PASTE YOUR MANUAL PATH HERE !!!
# Example: r"C:\Users\user\Downloads\face-anti-spoofing-dataset-95000-sets"
MANUAL_PATH_95K = r"/FAS/archive"

# Unified Class Mapping
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


# --- 2. Dataset Class ---
class UnifiedSpoofDataset(Dataset):
    def __init__(self, root_dir, source_name="Generic"):
        self.root_dir = root_dir
        self.source_name = source_name
        self.samples = []

        # MTCNN setup
        self.mtcnn = MTCNN(
            image_size=224, margin=40, keep_all=False,
            select_largest=True, post_process=False, device=DEVICE
        )

        self.transform_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
        self.transform_fallback = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), self.transform_norm
        ])

        self._crawl()

    def _crawl(self):
        print(f"[INFO] Scanning {self.source_name} at: {self.root_dir}")
        valid_exts = ('.jpg', '.png', '.jpeg', '.mp4', '.avi')
        count = 0

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
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
                            label_attack = 4  # Default to generic replay if unknown

                    is_real = 1.0 if label_attack == 0 else 0.0
                    self.samples.append((os.path.join(root, file), label_attack, is_real))
                    count += 1

        print(f"[INFO] -> Found {count} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_attack, label_real = self.samples[idx]
        img = None
        if path.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total))
                ret, frame = cap.read()
                if ret: img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            try:
                img = Image.open(path).convert('RGB')
            except:
                pass

        if img is None: return self.__getitem__((idx + 1) % len(self))

        try:
            face_tensor = self.mtcnn(img)
        except:
            face_tensor = None

        if face_tensor is not None:
            face_tensor = face_tensor.float() / 255.0
            face_tensor = self.transform_norm(face_tensor)
        else:
            face_tensor = self.transform_fallback(img)

        return face_tensor.detach(), torch.tensor(label_real, dtype=torch.float32), torch.tensor(label_attack,
                                                                                                 dtype=torch.long)


# --- 3. Model ---
class MultiTaskDiVT(nn.Module):
    def __init__(self, num_attack_classes=9, pretrained=True):
        super(MultiTaskDiVT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.liveness_head = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1),
                                           nn.Sigmoid())
        self.attack_head = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.4),
                                         nn.Linear(256, num_attack_classes))

    def forward(self, x):
        feat = self.backbone(x)
        return self.liveness_head(feat), self.attack_head(feat)


# --- 4. Main Execution ---
if __name__ == "__main__":
    datasets = []

    # --- DATASET 1: Axondata (Auto-download) ---
    try:
        print("[INFO] Checking Dataset 1 (Axondata)...")
        p1 = kagglehub.dataset_download("axondata/face-anti-spoofing-dataset")
        datasets.append(UnifiedSpoofDataset(p1, "Axondata"))
    except Exception as e:
        print(f"[WARN] Failed to load Dataset 1: {e}")

    # --- DATASET 2: 95k Sets (MANUAL PATH) ---
    if os.path.exists(MANUAL_PATH_95K):
        print(f"[INFO] Loading Dataset 2 (95k Sets) from manual path...")
        datasets.append(UnifiedSpoofDataset(MANUAL_PATH_95K, "95k Dataset"))
    else:
        print(f"[WARN] Manual path for 95k dataset not found: {MANUAL_PATH_95K}")
        print("Please edit the 'MANUAL_PATH_95K' variable at the top of the script!")

    # --- DATASET 3: 30k Sets (Auto-download) ---
    try:
        print("[INFO] Checking Dataset 3 (30k Sets)...")
        p3 = kagglehub.dataset_download("tapakah68/face-anti-spoofing-data")
        datasets.append(UnifiedSpoofDataset(p3, "30k Dataset"))
    except Exception as e:
        print(f"[WARN] Failed to load Dataset 3: {e}")

    # Combine & Train
    if len(datasets) > 0:
        full_dataset = ConcatDataset(datasets)
        print(f"\n[INFO] TRAINING ON {len(full_dataset)} TOTAL SAMPLES")

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = MultiTaskDiVT(num_attack_classes=9).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion_live = nn.BCELoss()
        criterion_attack = nn.CrossEntropyLoss()

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
                if i % 50 == 0: print(f"Epoch {epoch + 1} Batch {i} Loss: {loss.item():.4f}")

            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Epoch {epoch + 1} Saved.")
    else:
        print("[ERROR] No datasets available. Check paths/internet.")