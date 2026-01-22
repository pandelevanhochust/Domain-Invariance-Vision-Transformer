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

        # CeFA Mapping: 1=Live(0), 2=Cutout(1), 3=Replay(2)
        self.cefa_mapping = {'1': 0, '2': 1, '3': 2}

        if verbose:
            print('--------------------------------------------------')
            print(f'Scanning TEST dataset for: {data_names}')
            print('--------------------------------------------------')

        for data_name in data_names:
            # 1. Determine Path
            # Try specific phase path (e.g., dataset/CeFA/test)
            phase_path = os.path.join(root_dir, data_name, phase)

            # If 'test' folder doesn't exist (like in your screenshot),
            # fall back to the main folder (dataset/CeFA)
            if not os.path.exists(phase_path):
                phase_path = os.path.join(root_dir, data_name)

            if not os.path.exists(phase_path):
                print(f"Warning: Could not find path: {phase_path}")
                continue

            # 2. Walk and Find "profile" folders
            for root, dirs, files in os.walk(phase_path):
                if os.path.basename(root) == 'profile':
                    video_folder_path = root
                    parent_folder_name = os.path.basename(os.path.dirname(root))

                    # 3. Parse Label from Filename
                    # Try to parse CeFA format: 1_001_1_1_1
                    spoof_label = self.parse_cefa_label(parent_folder_name)

                    # If parsing failed, fallback to Folder Name check (for Unified_FAS support)
                    if spoof_label is None:
                        path_parts = root.lower().split(os.sep)
                        if 'live' in path_parts:
                            spoof_label = 0
                        elif 'cutout' in path_parts:
                            spoof_label = 1
                        elif 'replay' in path_parts:
                            spoof_label = 2

                    # Only add if we found a valid label
                    if spoof_label is not None:
                        # We store the path and the label directly
                        self.video_list.append((video_folder_path, spoof_label))

        if verbose:
            print(f'Found {len(self.video_list)} test videos.')

    def parse_cefa_label(self, filename):
        try:
            # Check if filename ends with _1, _2, _3
            parts = filename.split('_')
            type_code = parts[-1]
            return self.cefa_mapping.get(type_code, None)
        except:
            return None

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        # Retrieve path and label
        video_name, spoofing_label = self.video_list[idx]

        # Domain Logic (Default to 0 for single domain)
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
        frames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(frames) == 0:
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