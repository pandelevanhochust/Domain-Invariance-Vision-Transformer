import os
import shutil
from pathlib import Path

# ================= CONFIGURATION =================
SOURCE_ROOT = r"D:\CODIng\Machine Learning\FAS\GD-FAS\dataset\FAS"
# This will create 'dataset/Fold1', 'dataset/Fold2', 'dataset/Fold3'
OUTPUT_ROOT = r"D:\CODIng\Machine Learning\FAS\GD-FAS\dataset\CrossVal"
DATASET_NAME = "CustomFAS"

# Define your 3 Folds manually to ensure balance
# Every video ID (1-9) appears exactly once in the TEST lists below.
FOLDS = {
    "Fold1": ["1", "2", "3"],
    "Fold2": ["4", "5", "6"],
    "Fold3": ["7", "8", "9"]
}

FRAME_INTERVAL = 5


# =================================================

def process_video(file_path, video_stem, category, fold_name, is_test_video):
    import cv2

    split_type = 'test' if is_test_video else 'train'

    # Structure: dataset/CrossVal/Fold1/CustomFAS/train/attack/1/...
    target_dir = os.path.join(OUTPUT_ROOT, fold_name, DATASET_NAME, split_type, category, video_stem)
    os.makedirs(target_dir, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success: break

        if frame_count % FRAME_INTERVAL == 0:
            cv2.imwrite(os.path.join(target_dir, f"{frame_count:05d}.png"), frame)
        frame_count += 1
    cap.release()


def main():
    print(f"Generating 3-Fold Dataset at: {OUTPUT_ROOT}")

    categories = ['live', 'attack']
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv')

    for category in categories:
        source_cat_path = os.path.join(SOURCE_ROOT, category)
        if not os.path.exists(source_cat_path): continue

        files = os.listdir(source_cat_path)

        for file in files:
            if not file.lower().endswith(valid_extensions): continue

            video_stem = Path(file).stem
            file_path = os.path.join(source_cat_path, file)

            print(f"Processing {category}/{file}...")

            # Add this video to ALL 3 Folds
            for fold_name, test_ids in FOLDS.items():
                # Check if this video belongs to the TEST set for this specific fold
                is_test = video_stem in test_ids
                process_video(file_path, video_stem, category, fold_name, is_test)

    print("\nGeneration Complete.")
    print("To train, run training.py 3 times with different data roots:")
    print(f"1. --data_root {os.path.join(OUTPUT_ROOT, 'Fold1')}")
    print(f"2. --data_root {os.path.join(OUTPUT_ROOT, 'Fold2')}")
    print(f"3. --data_root {os.path.join(OUTPUT_ROOT, 'Fold3')}")


if __name__ == "__main__":
    main()