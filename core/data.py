import os
import cv2
import torch
import kagglehub
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- 1. Download Dataset (as per your snippet) ---
print("Downloading/Verifying Dataset...")
dataset_path = kagglehub.dataset_download("axondata/face-anti-spoofing-dataset")
print(f"Dataset is located at: {dataset_path}")


# --- 2. Custom Dataset Loader ---
class AntiSpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=1):
        """
        Crawls root_dir for images/videos. Assumes structure like:
        root_dir/
          real/
          spoof_print/
          spoof_replay/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.frames_per_video = frames_per_video

        # 1. Traverse directory to find classes
        # We try to detect 'real' vs 'spoof' based on folder keywords
        self.class_to_idx = {}
        self._crawl_directory()

    def _crawl_directory(self):
        valid_exts = ('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')
        idx_counter = 0

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    # Infer label from parent folder name
                    folder_name = os.path.basename(root).lower()

                    # Logic to assign Class ID (Modify based on actual folder names)
                    if folder_name not in self.class_to_idx:
                        self.class_to_idx[folder_name] = idx_counter
                        idx_counter += 1

                    label = self.class_to_idx[folder_name]
                    is_fake = 0 if 'real' in folder_name or 'live' in folder_name else 1

                    self.samples.append((os.path.join(root, file), label, is_fake))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Found {len(self.samples)} samples across classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_type, is_fake = self.samples[idx]

        img = None
        # Handle Video: Extract one random frame
        if path.endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total_frames))
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
            cap.release()

        # Handle Image
        else:
            try:
                img = Image.open(path).convert('RGB')
            except:
                pass  # Handle corrupt files gracefully

        # Fallback for corrupt/empty video
        if img is None:
            # Return a black image or recursively get another sample
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label_type), torch.tensor(is_fake).float()


# --- Transforms ---
# ViT typically requires 224x224 and ImageNet normalization
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize Dataset
full_dataset = AntiSpoofDataset(dataset_path, transform=transform_pipeline)