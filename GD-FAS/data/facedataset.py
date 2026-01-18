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
        self.data_names = {}
        self.video_list = []

        if verbose:
            print('--------------------------------------------------')
            print(f'{"build test dataset":20} - number of video')
            print('--------------------------------------------------')

        for num, data_name in enumerate(data_names):
            self.data_names[data_name] = num

            # 1. Determine the path to the phase folder (train/test)
            # Try: dataset/Unified_FAS/CustomFAS/test
            phase_path = os.path.join(root_dir, data_name, phase)

            # If that doesn't exist, Try: dataset/Unified_FAS/test (Direct structure)
            if not os.path.exists(phase_path):
                phase_path = os.path.join(root_dir, phase)

            if not os.path.exists(phase_path):
                print(f"Warning: Could not find path: {phase_path}")
                continue

            # 2. Walk through ALL subfolders (live, cutout, replay, etc.)
            # This avoids hardcoding "attack" or "spoof"
            files_found = 0
            for root, dirs, files in os.walk(phase_path):
                # If this folder contains images (frames), treat it as a video sample
                # Your logic implies a "video" is a folder of images.
                # We check if there are image files inside.
                image_files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

                if len(image_files) > 0:
                    # It's a valid sample folder
                    self.video_list.append(root)
                    files_found += 1

            if verbose:
                print(f'{phase_path}: Found {files_found} folders/videos')

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]

        # --- MULTI-CLASS LABELING ---
        spoofing_label = 0  # Default to Live

        # Check path for keywords
        path_parts = video_name.replace('\\', '/').lower().split('/')

        # Find which label matches the folder name
        for part in path_parts:
            if part in LABEL_MAP:
                spoofing_label = LABEL_MAP[part]
                # If we found a specific attack type, stick with it.
                if part != 'live':
                    break

                    # Domain Logic
        domain_label = -1
        for part in path_parts:
            if part in self.data_names:
                domain_label = self.data_names[part]
                break

        if domain_label == -1:
            domain_label = 0

        image_x = self.sample_image(video_name)
        image_x = self.transform(image_x)

        sample = {
            "image_x": image_x,
            "label": spoofing_label,
            "domain": domain_label,
            "name": video_name,
        }
        return sample

    def sample_image(self, image_dir):
        # Your original logic: assume image_dir is a folder of frames
        frames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(frames) == 0:
            # Fallback if folder is empty (prevent crash)
            return Image.new('RGB', (224, 224))

        frames_total = len(frames)
        image_id = np.random.randint(0, frames_total)
        image_path = os.path.join(image_dir, frames[image_id])

        return Image.open(image_path).convert('RGB')


class BalanceFaceDataset(Dataset):
    def __init__(self, root_dir, data_names, phase='train', transform=None, max_iter=4000, verbose=False):
        self.transform = transform
        self.max_iter = max_iter
        self.video_list = {}  # This will store iterators for each class folder
        self.data_names = {}

        for num, data_name in enumerate(data_names):
            self.data_names[data_name] = num

            # 1. Determine Path (Robust check)
            phase_path = os.path.join(root_dir, data_name, phase)
            if not os.path.exists(phase_path):
                phase_path = os.path.join(root_dir, phase)

            if not os.path.exists(phase_path):
                print(f"Warning: Training path not found: {phase_path}")
                continue

            # 2. Find ALL sub-folders (live, cutout, replay) in the directory
            # We treat every immediate subfolder as a distinct class source
            subfolders = [f.path for f in os.scandir(phase_path) if f.is_dir()]

            for folder_path in subfolders:
                # Find all "video" directories inside this class folder
                # Assuming structure: train/live/subject1_folder/frames...
                # OR structure: train/live/image1.jpg (if flat)

                # Let's support your folder-based structure:
                valid_videos = []
                for root, dirs, files in os.walk(folder_path):
                    # Check if this root contains images
                    imgs = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if len(imgs) > 0:
                        valid_videos.append(root)

                if len(valid_videos) > 0:
                    random.shuffle(valid_videos)
                    # Use the folder path as the key
                    self.video_list[folder_path] = valid_videos

        if verbose:
            print('--------------------------------------------------')
            print(f'{"build train dataset":20} - number of video sources')
            print('--------------------------------------------------')
            for key in self.video_list:
                print(f'{key}: {len(self.video_list[key])}')

    def __len__(self):
        return self.max_iter

    def __getitem__(self, idx):
        sample = {}

        # Iterate over every class folder we found (e.g., live, cutout, replay)
        # This ensures the batch has 1 sample from EACH class type -> Balanced!
        for video_key in self.video_list:

            # Get next video from the list (or reshuffle if empty)
            if len(self.video_list[video_key]) == 0:
                continue  # Skip if empty

            # Simple circular iterator logic manually
            # We pick a random one for simplicity and robustness
            video_name = random.choice(self.video_list[video_key])

            # --- MULTI-CLASS LABELING ---
            spoofing_label = 0
            path_parts = video_name.replace('\\', '/').lower().split('/')

            for part in path_parts:
                if part in LABEL_MAP:
                    spoofing_label = LABEL_MAP[part]
                    if part != 'live': break
            # -----------------------------

            # Domain Logic
            domain_label = -1
            for part in path_parts:
                if part in self.data_names:
                    domain_label = self.data_names[part]
                    break
            if domain_label == -1:
                domain_label = 0

            image_x = self.sample_image(video_name)
            image_x_view1 = self.transform(image_x)
            image_x_view2 = self.transform(image_x)

            # Generate a unique key for the batch dictionary
            key_parts = video_key.replace('\\', '/').split('/')
            key_name = f"{key_parts[-1]}"  # e.g. "live", "cutout"

            sample[key_name] = {
                "image_x_v1": image_x_view1,
                "image_x_v2": image_x_view2,
                "label": spoofing_label,
                "domain": domain_label,
                "name": video_name,
            }

        return sample

    def sample_image(self, image_dir):
        frames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(frames) == 0:
            return Image.new('RGB', (224, 224))

        frames_total = len(frames)
        image_id = np.random.randint(0, frames_total)
        image_path = os.path.join(image_dir, frames[image_id])

        return Image.open(image_path).convert('RGB')