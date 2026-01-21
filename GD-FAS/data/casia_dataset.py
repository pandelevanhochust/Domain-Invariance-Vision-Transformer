import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class CasiaSurfDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.data_list = []

        # 1. Define paths to Real and Fake parts
        # Adjust 'Colab_Casia_Simple/CASIA-SURF-CROP' to your actual root
        real_root = os.path.join(root_dir, 'real_part', f'real_{phase}_part')
        fake_root = os.path.join(root_dir, 'fake_part', f'fake_{phase}_part')

        print(f"Scanning CASIA-SURF ({phase})...")
        
        # 2. Scan Real Data (Label 0)
        self._scan_folder(real_root, label=0)
        
        # 3. Scan Fake Data (Label 1)
        self._scan_folder(fake_root, label=1)
        
        print(f"Found {len(self.data_list)} paired samples.")

    def _scan_folder(self, phase_root, label):
        if not os.path.exists(phase_root):
            print(f"Warning: Folder not found {phase_root}")
            return

        # Loop through Subjects (e.g., CLKJ_AS0005)
        for subject in os.listdir(phase_root):
            subj_path = os.path.join(phase_root, subject)
            if not os.path.isdir(subj_path): continue

            # Loop through Trials (e.g., 01_e_s.rssd)
            for trial in os.listdir(subj_path):
                trial_path = os.path.join(subj_path, trial)
                
                # Check for Color and IR folders
                rgb_dir = os.path.join(trial_path, 'color')
                ir_dir = os.path.join(trial_path, 'ir')

                if os.path.exists(rgb_dir) and os.path.exists(ir_dir):
                    # Match files by name (assuming 001.jpg in color matches 001.jpg in ir)
                    for img_name in os.listdir(rgb_dir):
                        if img_name.lower().endswith('.jpg'):
                            rgb_path = os.path.join(rgb_dir, img_name)
                            ir_path = os.path.join(ir_dir, img_name)
                            
                            # Ensure both exist
                            if os.path.exists(ir_path):
                                self.data_list.append({
                                    'rgb': rgb_path,
                                    'ir': ir_path,
                                    'label': label
                                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_info = self.data_list[idx]
        
        # Load RGB
        img_rgb = Image.open(sample_info['rgb']).convert('RGB')
        
        # Load IR (Load as grayscale, then convert to RGB format for model compatibility)
        img_ir = Image.open(sample_info['ir']).convert('L').convert('RGB')

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_ir = self.transform(img_ir)

        return {
            "image_x": img_rgb,   # Standard name your trainer expects
            "image_ir": img_ir,   # New IR input
            "label": sample_info['label'],
            "name": sample_info['rgb']
        }

class CefaAFDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.data_list = []

        # Point to the AF folder inside CeFA-Race
        af_root = os.path.join(root_dir, 'AF')
        print(f"Scanning CeFA-AF Subset in {af_root}...")

        if os.path.exists(af_root):
            for subject in os.listdir(af_root):
                subj_path = os.path.join(af_root, subject)
                if not os.path.isdir(subj_path): continue

                for seq_name in os.listdir(subj_path):
                    seq_path = os.path.join(subj_path, seq_name)

                    # Parse Label: 1_000_1... -> Type is 3rd number
                    parts = seq_name.split('_')
                    if len(parts) < 3: continue
                    attack_type = parts[2]

                    # 1=Live, 2=Print, 3=Replay, 4=Mask
                    label = 0 if attack_type == '1' else 1

                    # Find folders
                    rgb_dir = os.path.join(seq_path, 'profile')
                    ir_dir = os.path.join(seq_path, 'ir')

                    if os.path.exists(rgb_dir) and os.path.exists(ir_dir):
                        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
                        ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.jpg')])

                        min_len = min(len(rgb_files), len(ir_files))
                        for i in range(min_len):
                            self.data_list.append({
                                'rgb': os.path.join(rgb_dir, rgb_files[i]),
                                'ir': os.path.join(ir_dir, ir_files[i]),
                                'label': label
                            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        try:
            img_rgb = Image.open(sample['rgb']).convert('RGB')
            img_ir = Image.open(sample['ir']).convert('L').convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_ir = self.transform(img_ir)

        return {
            "image_x": img_rgb,
            "image_ir": img_ir,
            "label": sample['label'],
            "name": sample['rgb']
        }

class CefaIROnlyDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.data_list = []

        # Point to the AF folder inside CeFA-Race
        af_root = os.path.join(root_dir, 'AF')
        print(f"Scanning CeFA-AF (IR ONLY) in {af_root}...")

        if os.path.exists(af_root):
            for subject in os.listdir(af_root):
                subj_path = os.path.join(af_root, subject)
                if not os.path.isdir(subj_path): continue

                for seq_name in os.listdir(subj_path):
                    seq_path = os.path.join(subj_path, seq_name)

                    # Parse Label: 1_000_1... -> Type is 3rd number
                    parts = seq_name.split('_')
                    if len(parts) < 3: continue
                    attack_type = parts[2]

                    # 1=Live, Others=Spoof
                    label = 0 if attack_type == '1' else 1

                    # Find IR folder
                    ir_dir = os.path.join(seq_path, 'ir')

                    if os.path.exists(ir_dir):
                        ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.jpg')])

                        for ir_file in ir_files:
                            self.data_list.append({
                                'path': os.path.join(ir_dir, ir_file),
                                'label': label
                            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        try:
            # VITAL: Load IR but convert to RGB (3 channels) so CLIP/ResNet accepts it
            img = Image.open(sample['path']).convert('L').convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        if self.transform:
            img = self.transform(img)

        return {
            "image_x": img,  # <--- We put IR data in the main 'image_x' slot
            "label": sample['label'],
            "image_ir": torch.zeros(1)  # Dummy value to prevent errors
        }