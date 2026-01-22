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
        self.data_names = {}

        # We Initialize specific buckets for your 3 target classes
        # This ensures the 'Balanced' logic works by forcing every batch to have
        # one of each type.
        self.video_list = {
            'live': [],
            'cutout': [],
            'replay': []
        }

        # Mapping CeFA numeric codes (last digit) to your class buckets
        # 1=Live, 2=Print(Cutout), 3=Replay, 4=3DMask(Ignored)
        self.cefa_mapping = {
            '1': 'live',
            '2': 'cutout',
            '3': 'replay'
        }

        print('--------------------------------------------------')
        print(f' Scanning CeFA Dataset at: {root_dir}')
        print('--------------------------------------------------')

        for num, data_name in enumerate(data_names):
            self.data_names[data_name] = num

            # Construct path: e.g. dataset/CeFA/phase (if phase folders exist)
            # Or just search the whole root if structure is flat
            phase_path = os.path.join(root_dir, data_name)
            if not os.path.exists(phase_path):
                # Fallback for simpler structure
                phase_path = os.path.join(root_dir)

            # --- NEW TRAVERSAL LOGIC ---
            # We walk through the entire directory tree looking for "profile" folders.
            # In CeFA, "profile" contains the RGB images.
            for root, dirs, files in os.walk(phase_path):

                # We only care about the folders that contain the actual images
                # In CeFA, the RGB images are inside a folder named "profile"
                if os.path.basename(root) == 'profile':

                    # The parent folder name holds the label info (e.g., "1_001_1_1_1")
                    video_folder_path = root
                    parent_folder_name = os.path.basename(os.path.dirname(root))

                    # Parse the Label from the parent folder name
                    label_key = self.parse_cefa_filename(parent_folder_name)

                    # If it's a valid class (Live, Cutout, Replay), add it to the bucket
                    if label_key in self.video_list:
                        self.video_list[label_key].append(video_folder_path)

        # Verbose output to verify data was found
        if verbose:
            print(f'{"Data Summary":20}')
            print('--------------------------------------------------')
            for key in self.video_list:
                print(f'{key.capitalize()}: {len(self.video_list[key])} videos found')

    def parse_cefa_filename(self, filename):
        """
        Parses strings like '1_001_1_1_1' or '1_001_3_1_4'
        Splits by '_' and checks the last digit.
        """
        try:
            parts = filename.split('_')
            # The type is usually the last digit in CeFA filenames
            # Format: Race_Subject_Session_Light_TYPE
            type_code = parts[-1]

            return self.cefa_mapping.get(type_code, None)
        except:
            return None

    def __len__(self):
        return self.max_iter

    def __getitem__(self, idx):
        sample = {}

        # Iterate over 'live', 'cutout', 'replay' buckets
        # This guarantees every batch has 1 Live, 1 Cutout, 1 Replay
        for class_name in self.video_list:

            video_paths = self.video_list[class_name]

            # Safety check if a class is missing from dataset
            if len(video_paths) == 0:
                continue

            # Randomly select one video folder from this class
            video_name = random.choice(video_paths)

            # Determine Label ID (0, 1, 2)
            # We map class_name string back to integer for the model
            if class_name == 'live':
                spoofing_label = 0
            elif class_name == 'cutout':
                spoofing_label = 1
            elif class_name == 'replay':
                spoofing_label = 2
            else:
                spoofing_label = 0  # Default

            # Domain Logic (simplified)
            domain_label = 0

            # Sample image from the folder
            image_x = self.sample_image(video_name)

            # Create two augmented views (Contrastive Learning)
            image_x_view1 = self.transform(image_x)
            image_x_view2 = self.transform(image_x)

            sample[class_name] = {
                "image_x_v1": image_x_view1,
                "image_x_v2": image_x_view2,
                "label": spoofing_label,
                "domain": domain_label,
                "name": video_name,
            }

        return sample

    def sample_image(self, image_dir):
        # Standard sampling from the folder
        frames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(frames) == 0:
            return Image.new('RGB', (224, 224))

        frames_total = len(frames)
        image_id = np.random.randint(0, frames_total)
        image_path = os.path.join(image_dir, frames[image_id])

        return Image.open(image_path).convert('RGB')