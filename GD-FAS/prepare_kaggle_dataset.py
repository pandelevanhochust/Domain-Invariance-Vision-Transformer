"""
Script to prepare Kaggle datasets for GD-FAS training
This script processes videos and images from various Kaggle FAS datasets
and organizes them into the required structure for GD-FAS.
"""

import os
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

# Configuration
SOURCE_DATASETS_PATH = r"D:\CODIng\Machine Learning\FAS\datasets"
TARGET_PATH = r"D:\CODIng\Machine Learning\FAS\GD-FAS\datasets"
TRAIN_RATIO = 0.8  # 80% train, 20% test
FRAME_INTERVAL = 5  # Extract every 5th frame

def create_directory_structure(base_path, dataset_name):
    """Create the required directory structure for a dataset."""
    paths = {
        'train_attack': os.path.join(base_path, dataset_name, 'train', 'attack'),
        'train_live': os.path.join(base_path, dataset_name, 'train', 'live'),
        'test_attack': os.path.join(base_path, dataset_name, 'test', 'attack'),
        'test_live': os.path.join(base_path, dataset_name, 'test', 'live'),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def extract_frames_from_video(video_path, output_dir, frame_interval=5):
    """Extract every nth frame from a video."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval == 0:
            frame_name = f"{saved_count:03d}.png"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
    
    cap.release()
    return saved_count

def copy_image_to_folder(image_path, output_dir):
    """Copy a single image to output directory with standard naming."""
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(image_path, os.path.join(output_dir, "000.png"))
    return 1

def process_30k_fas_dataset(source_path, target_path):
    """Process the 30k_fas dataset."""
    print("\n=== Processing 30k_fas Dataset ===")
    dataset_name = "30k_fas"
    paths = create_directory_structure(target_path, dataset_name)
    
    # Define attack and live categories
    attack_categories = ['cut-out printouts', 'printouts', 'replay']
    live_categories = ['live_video', 'live_selfie']
    
    video_info = {}
    
    # Collect all videos
    for category in attack_categories + live_categories:
        category_path = os.path.join(source_path, '30k_fas', category)
        if not os.path.exists(category_path):
            continue
        
        files = os.listdir(category_path)
        video_info[category] = []
        
        for file in files:
            file_path = os.path.join(category_path, file)
            if os.path.isfile(file_path):
                video_info[category].append((file, file_path))
    
    # Process each category
    for category, files in video_info.items():
        random.shuffle(files)
        split_idx = int(len(files) * TRAIN_RATIO)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        is_live = category in live_categories
        
        # Process training files
        for idx, (filename, filepath) in enumerate(tqdm(train_files, desc=f"Train {category}")):
            video_name = f"{category.replace(' ', '_')}_{idx:04d}"
            output_dir = paths['train_live'] if is_live else paths['train_attack']
            output_dir = os.path.join(output_dir, video_name)
            
            if filepath.lower().endswith(('.mp4', '.mov', '.avi')):
                extract_frames_from_video(filepath, output_dir, FRAME_INTERVAL)
            elif filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
                copy_image_to_folder(filepath, output_dir)
        
        # Process test files
        for idx, (filename, filepath) in enumerate(tqdm(test_files, desc=f"Test {category}")):
            video_name = f"{category.replace(' ', '_')}_{idx:04d}"
            output_dir = paths['test_live'] if is_live else paths['test_attack']
            output_dir = os.path.join(output_dir, video_name)
            
            if filepath.lower().endswith(('.mp4', '.mov', '.avi')):
                extract_frames_from_video(filepath, output_dir, FRAME_INTERVAL)
            elif filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
                copy_image_to_folder(filepath, output_dir)

def process_98k_real_dataset(source_path, target_path):
    """Process the 98k_real dataset (all live samples)."""
    print("\n=== Processing 98k_real Dataset ===")
    dataset_name = "98k_real"
    paths = create_directory_structure(target_path, dataset_name)
    
    dataset_path = os.path.join(source_path, '98k_real')
    all_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    random.shuffle(all_folders)
    
    split_idx = int(len(all_folders) * TRAIN_RATIO)
    train_folders = all_folders[:split_idx]
    test_folders = all_folders[split_idx:]
    
    # Process training folders
    for folder in tqdm(train_folders, desc="Train 98k_real"):
        folder_path = os.path.join(dataset_path, folder)
        output_dir = os.path.join(paths['train_live'], folder)
        
        # Look for video or image files
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith(('.mp4', '.mov')):
                extract_frames_from_video(file_path, output_dir, FRAME_INTERVAL)
                break  # Only process one video per folder
            elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                copy_image_to_folder(file_path, output_dir)
                break
    
    # Process test folders
    for folder in tqdm(test_folders, desc="Test 98k_real"):
        folder_path = os.path.join(dataset_path, folder)
        output_dir = os.path.join(paths['test_live'], folder)
        
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith(('.mp4', '.mov')):
                extract_frames_from_video(file_path, output_dir, FRAME_INTERVAL)
                break
            elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                copy_image_to_folder(file_path, output_dir)
                break

def process_archive_dataset(source_path, target_path):
    """Process the archive dataset with various attack types."""
    print("\n=== Processing Archive Dataset ===")
    dataset_name = "archive"
    paths = create_directory_structure(target_path, dataset_name)
    
    # Define attack and live categories
    attack_folders = [
        '3D_paper_mask',
        'Cutout_attacks',
        'Latex_mask',
        'Replay_display_attacks',
        'Replay_mobile_attacks',
        'Silicone_mask',
        'Textile 3D Face Mask Attack Sample',
        'Wrapped_3D_paper_mask'
    ]
    live_folders = ['Selfies']
    
    # Process attack videos
    for attack_folder in attack_folders:
        attack_path = os.path.join(source_path, 'archive', attack_folder)
        if not os.path.exists(attack_path):
            continue
        
        videos = []
        # Recursively find all video/image files
        for root, dirs, files in os.walk(attack_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.mov', '.avi', '.jpg', '.jpeg', '.png')):
                    videos.append(os.path.join(root, file))
        
        random.shuffle(videos)
        split_idx = int(len(videos) * TRAIN_RATIO)
        train_videos = videos[:split_idx]
        test_videos = videos[split_idx:]
        
        # Process training videos
        for idx, video_path in enumerate(tqdm(train_videos, desc=f"Train {attack_folder}")):
            video_name = f"{attack_folder.replace(' ', '_')}_{idx:04d}"
            output_dir = os.path.join(paths['train_attack'], video_name)
            
            if video_path.lower().endswith(('.mp4', '.mov', '.avi')):
                extract_frames_from_video(video_path, output_dir, FRAME_INTERVAL)
            else:
                copy_image_to_folder(video_path, output_dir)
        
        # Process test videos
        for idx, video_path in enumerate(tqdm(test_videos, desc=f"Test {attack_folder}")):
            video_name = f"{attack_folder.replace(' ', '_')}_{idx:04d}"
            output_dir = os.path.join(paths['test_attack'], video_name)
            
            if video_path.lower().endswith(('.mp4', '.mov', '.avi')):
                extract_frames_from_video(video_path, output_dir, FRAME_INTERVAL)
            else:
                copy_image_to_folder(video_path, output_dir)
    
    # Process live selfies
    selfie_path = os.path.join(source_path, 'archive', 'Selfies')
    if os.path.exists(selfie_path):
        selfies = [os.path.join(selfie_path, f) for f in os.listdir(selfie_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(selfies)
        split_idx = int(len(selfies) * TRAIN_RATIO)
        train_selfies = selfies[:split_idx]
        test_selfies = selfies[split_idx:]
        
        for idx, selfie_path in enumerate(tqdm(train_selfies, desc="Train Selfies")):
            video_name = f"selfie_{idx:04d}"
            output_dir = os.path.join(paths['train_live'], video_name)
            copy_image_to_folder(selfie_path, output_dir)
        
        for idx, selfie_path in enumerate(tqdm(test_selfies, desc="Test Selfies")):
            video_name = f"selfie_{idx:04d}"
            output_dir = os.path.join(paths['test_live'], video_name)
            copy_image_to_folder(selfie_path, output_dir)

def process_unidata_real_dataset(source_path, target_path):
    """Process the unidata_real dataset (all live samples)."""
    print("\n=== Processing unidata_real Dataset ===")
    dataset_name = "unidata_real"
    paths = create_directory_structure(target_path, dataset_name)
    
    dataset_path = os.path.join(source_path, 'unidata_real')
    all_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    random.shuffle(all_folders)
    
    split_idx = int(len(all_folders) * TRAIN_RATIO)
    train_folders = all_folders[:split_idx]
    test_folders = all_folders[split_idx:]
    
    # Process training folders
    for folder in tqdm(train_folders, desc="Train unidata_real"):
        folder_path = os.path.join(dataset_path, folder)
        output_dir = os.path.join(paths['train_live'], f"person_{folder}")
        
        # Process video file
        video_file = os.path.join(folder_path, 'video.mp4') if os.path.exists(os.path.join(folder_path, 'video.mp4')) else os.path.join(folder_path, 'video.MOV')
        if os.path.exists(video_file):
            extract_frames_from_video(video_file, output_dir, FRAME_INTERVAL)
    
    # Process test folders
    for folder in tqdm(test_folders, desc="Test unidata_real"):
        folder_path = os.path.join(dataset_path, folder)
        output_dir = os.path.join(paths['test_live'], f"person_{folder}")
        
        video_file = os.path.join(folder_path, 'video.mp4') if os.path.exists(os.path.join(folder_path, 'video.mp4')) else os.path.join(folder_path, 'video.MOV')
        if os.path.exists(video_file):
            extract_frames_from_video(video_file, output_dir, FRAME_INTERVAL)

def print_dataset_statistics(target_path):
    """Print statistics about the processed datasets."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for dataset_name in os.listdir(target_path):
        dataset_path = os.path.join(target_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        
        print(f"\n{dataset_name}:")
        for phase in ['train', 'test']:
            phase_path = os.path.join(dataset_path, phase)
            if not os.path.exists(phase_path):
                continue
            
            for category in ['attack', 'live']:
                category_path = os.path.join(phase_path, category)
                if not os.path.exists(category_path):
                    continue
                
                video_folders = [d for d in os.listdir(category_path) 
                               if os.path.isdir(os.path.join(category_path, d))]
                total_frames = 0
                for folder in video_folders:
                    folder_path = os.path.join(category_path, folder)
                    frames = len([f for f in os.listdir(folder_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    total_frames += frames
                
                print(f"  {phase}/{category}: {len(video_folders)} videos, {total_frames} frames")

def main():
    """Main function to process all datasets."""
    print("="*60)
    print("KAGGLE DATASET PREPARATION FOR GD-FAS")
    print("="*60)
    print(f"Source: {SOURCE_DATASETS_PATH}")
    print(f"Target: {TARGET_PATH}")
    print(f"Train/Test Split: {TRAIN_RATIO*100}% / {(1-TRAIN_RATIO)*100}%")
    print(f"Frame Interval: Every {FRAME_INTERVAL}th frame")
    print("="*60)
    
    # Create target directory
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    # Process each dataset
    try:
        process_30k_fas_dataset(SOURCE_DATASETS_PATH, TARGET_PATH)
    except Exception as e:
        print(f"Error processing 30k_fas: {e}")
    
    try:
        process_98k_real_dataset(SOURCE_DATASETS_PATH, TARGET_PATH)
    except Exception as e:
        print(f"Error processing 98k_real: {e}")
    
    try:
        process_archive_dataset(SOURCE_DATASETS_PATH, TARGET_PATH)
    except Exception as e:
        print(f"Error processing archive: {e}")
    
    try:
        process_unidata_real_dataset(SOURCE_DATASETS_PATH, TARGET_PATH)
    except Exception as e:
        print(f"Error processing unidata_real: {e}")
    
    # Print statistics
    print_dataset_statistics(TARGET_PATH)
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print("\nYou can now train the model using:")
    print("python training.py --protocol O_C_I_to_M --gs")
    print("\nNote: Update the protocol based on your dataset names.")
    print("Available datasets: 30k_fas, 98k_real, archive, unidata_real")

if __name__ == "__main__":
    main()
    main()
