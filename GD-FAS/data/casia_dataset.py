import os
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