import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Define the Multi-Class Mapping Globally so both classes use it
LABEL_MAP = {
    "live": 0,
    "cutout": 1,
    "replay": 2,
}


class FaceDataset(Dataset):
    def __init__(self, root_dir, data_names, phase='test', transform=None, verbose=False):
        self.transform = transform
        self.video_list = []
        self.cefa_mapping = {'1': 0, '2': 1, '3': 2}

        # --- SPLIT CONFIGURATION ---
        # Test on Subjects 41 to 50
        self.train_max_id = 40

        if verbose: print(f'Scanning TEST dataset (Subjects > {self.train_max_id})...')

        for data_name in data_names:
            phase_path = os.path.join(root_dir, data_name)
            if not os.path.exists(phase_path): continue

            for root, dirs, files in os.walk(phase_path):
                if os.path.basename(root) == 'profile':
                    parent_name = os.path.basename(os.path.dirname(root))

                    # Check Subject ID
                    if not self.is_valid_subject(parent_name, phase='test'):
                        continue

                    spoof_label = self.parse_cefa_label(parent_name)
                    if spoof_label is not None:
                        self.video_list.append((root, spoof_label))

        if verbose: print(f'Found {len(self.video_list)} test videos.')

    def is_valid_subject(self, filename, phase):
        try:
            parts = filename.split('_')
            subject_id = int(parts[1])

            if phase == 'test':
                # Accept only ID > 40
                return subject_id > self.train_max_id
            return False
        except:
            return False

    def parse_cefa_label(self, filename):
        try:
            return self.cefa_mapping.get(filename.split('_')[-1], None)
        except:
            return None

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name, label = self.video_list[idx]
        image_x = self.sample_image(video_name)
        return {"image_x": self.transform(image_x), "label": label, "domain": 0, "name": video_name}

    def sample_image(self, image_dir):
        frames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]
        if not frames: return Image.new('RGB', (224, 224))
        return Image.open(os.path.join(image_dir, random.choice(frames))).convert('RGB')

class BalanceFaceDataset(Dataset):
    def __init__(self, root_dir, data_names, phase='train', transform=None, max_iter=4000, verbose=False):
        self.transform = transform
        self.max_iter = max_iter
        self.video_list = {'live': [], 'cutout': [], 'replay': []}
        self.cefa_mapping = {'1': 'live', '2': 'cutout', '3': 'replay'}

        # --- SPLIT CONFIGURATION ---
        # Train on Subjects 1 to 40 (since you have 50 total)
        self.train_max_id = 40

        if verbose: print(f'Scanning TRAIN dataset (Subjects 000-{self.train_max_id})...')

        for data_name in data_names:
            phase_path = os.path.join(root_dir, data_name)
            if not os.path.exists(phase_path): continue

            for root, dirs, files in os.walk(phase_path):
                if os.path.basename(root) == 'profile':
                    parent_name = os.path.basename(os.path.dirname(root))

                    # Check Subject ID
                    if not self.is_valid_subject(parent_name, phase='train'):
                        continue

                    label_key = self.parse_cefa_filename(parent_name)
                    if label_key in self.video_list:
                        self.video_list[label_key].append(root)

    def is_valid_subject(self, filename, phase):
        try:
            parts = filename.split('_')
            subject_id = int(parts[1])

            if phase == 'train':
                # Accept only ID <= 40
                return subject_id <= self.train_max_id
            return False
        except:
            return False

    def parse_cefa_filename(self, filename):
        try:
            return self.cefa_mapping.get(filename.split('_')[-1], None)
        except:
            return None

    def __len__(self):
        return self.max_iter

    def __getitem__(self, idx):
        sample = {}
        for class_name in self.video_list:
            video_paths = self.video_list[class_name]
            if len(video_paths) == 0: continue
            video_name = random.choice(video_paths)

            spoofing_label = 0
            if class_name == 'cutout':
                spoofing_label = 1
            elif class_name == 'replay':
                spoofing_label = 2

            image_x = self.sample_image(video_name)
            sample[class_name] = {
                "image_x_v1": self.transform(image_x),
                "image_x_v2": self.transform(image_x),
                "label": spoofing_label,
                "domain": 0,
                "name": video_name,
            }
        return sample

    def sample_image(self, image_dir):
        frames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]
        if not frames: return Image.new('RGB', (224, 224))
        return Image.open(os.path.join(image_dir, np.random.choice(frames))).convert('RGB')