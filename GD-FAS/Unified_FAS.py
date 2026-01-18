import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Define your source paths (Adjust Fold1/Fold3 if needed)
source_replay_root = "dataset/CrossVal/Fold3/CustomFAS"
source_cutout_root = "dataset/CrossValCutout/Fold1/CustomFAS"

# 2. Define where you want the NEW unified dataset to be
dest_root = "dataset/Unified_FAS"

# 3. Define the mapping for your new multi-class structure
# Structure: "New_Folder_Name": ("Source_Root", "Original_Subfolder_Name")
folder_map = {
    "live": [(source_replay_root, "live"), (source_cutout_root, "live")],  # Merge live data from both
    "replay": [(source_replay_root, "attack")],  # Old "attack" folder becomes "replay"
    "cutout": [(source_cutout_root, "attack")]  # Old "attack" folder becomes "cutout"
}


def setup_directories():
    for subset in ["train", "test"]:
        for class_name in folder_map.keys():
            path = os.path.join(dest_root, subset, class_name)
            os.makedirs(path, exist_ok=True)
            print(f"Created: {path}")


def copy_files():
    print("\nStarting File Transfer...")

    # Loop through Train and Test sets
    for subset in ["train", "test"]:
        print(f"\nProcessing {subset.upper()} set...")

        # Loop through our new classes (live, replay, cutout)
        for class_name, sources in folder_map.items():

            # Where we are copying TO
            dest_dir = os.path.join(dest_root, subset, class_name)

            # Loop through where we are copying FROM
            for root_path, subfolder in sources:
                src_dir = os.path.join(root_path, subset, subfolder)

                if not os.path.exists(src_dir):
                    print(f"⚠️ Warning: Source not found: {src_dir}")
                    continue

                # Copy every file
                files = os.listdir(src_dir)
                for f in tqdm(files, desc=f"Copying to {class_name}", leave=False):
                    src_file = os.path.join(src_dir, f)
                    dst_file = os.path.join(dest_dir, f)

                    # Avoid copying directories, only files
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)

    print("\n✅ Success! Dataset Refactored at:", dest_root)


# --- EXECUTE ---
setup_directories()
copy_files()