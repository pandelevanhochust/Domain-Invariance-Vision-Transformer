"""
Dataset Verification Script for GD-FAS
This script checks if the datasets are properly prepared and ready for training.
"""

import os
import sys
from pathlib import Path

TARGET_PATH = r"D:\CODIng\Machine Learning\FAS\GD-FAS\datasets"
EXPECTED_DATASETS = ['30k_fas', '98k_real', 'archive', 'unidata_real']

def check_directory_structure(dataset_path, dataset_name):
    """Check if a dataset has the correct directory structure."""
    issues = []
    warnings = []
    
    # Check train and test directories
    for phase in ['train', 'test']:
        phase_path = os.path.join(dataset_path, dataset_name, phase)
        if not os.path.exists(phase_path):
            issues.append(f"Missing {phase} directory")
            continue
        
        # Check attack and live directories
        for category in ['attack', 'live']:
            category_path = os.path.join(phase_path, category)
            if not os.path.exists(category_path):
                issues.append(f"Missing {phase}/{category} directory")
                continue
            
            # Check if there are video folders
            video_folders = [d for d in os.listdir(category_path) 
                           if os.path.isdir(os.path.join(category_path, d))]
            
            if len(video_folders) == 0:
                issues.append(f"No video folders in {phase}/{category}")
            else:
                # Check if video folders have images
                empty_folders = 0
                total_frames = 0
                for folder in video_folders:
                    folder_path = os.path.join(category_path, folder)
                    frames = [f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if len(frames) == 0:
                        empty_folders += 1
                    total_frames += len(frames)
                
                if empty_folders > 0:
                    warnings.append(f"{empty_folders} empty folders in {phase}/{category}")
                
                if total_frames < 10:
                    warnings.append(f"Only {total_frames} frames in {phase}/{category} (might be too few)")
    
    return issues, warnings

def check_dataset_balance(dataset_path, dataset_name):
    """Check if the dataset has a reasonable balance between classes."""
    stats = {}
    
    for phase in ['train', 'test']:
        stats[phase] = {'attack': 0, 'live': 0}
        phase_path = os.path.join(dataset_path, dataset_name, phase)
        if not os.path.exists(phase_path):
            continue
        
        for category in ['attack', 'live']:
            category_path = os.path.join(phase_path, category)
            if not os.path.exists(category_path):
                continue
            
            video_folders = [d for d in os.listdir(category_path) 
                           if os.path.isdir(os.path.join(category_path, d))]
            stats[phase][category] = len(video_folders)
    
    return stats

def print_colored(text, color='green'):
    """Print colored text (works on Windows 10+)."""
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def main():
    print("="*70)
    print("GD-FAS DATASET VERIFICATION")
    print("="*70)
    print()
    
    # Check if datasets directory exists
    if not os.path.exists(TARGET_PATH):
        print_colored(f"❌ ERROR: Datasets directory not found at {TARGET_PATH}", 'red')
        print()
        print("Please run 'python prepare_kaggle_dataset.py' first to prepare the datasets.")
        return False
    
    # Check each expected dataset
    all_good = True
    total_issues = 0
    total_warnings = 0
    
    for dataset_name in EXPECTED_DATASETS:
        dataset_path = os.path.join(TARGET_PATH, dataset_name)
        
        print(f"Checking dataset: {dataset_name}")
        print("-" * 70)
        
        if not os.path.exists(dataset_path):
            print_colored(f"  ⚠️  Dataset not found (will be skipped)", 'yellow')
            print()
            continue
        
        # Check structure
        issues, warnings = check_directory_structure(TARGET_PATH, dataset_name)
        
        if issues:
            print_colored(f"  ❌ Issues found:", 'red')
            for issue in issues:
                print(f"     - {issue}")
            all_good = False
            total_issues += len(issues)
        
        if warnings:
            print_colored(f"  ⚠️  Warnings:", 'yellow')
            for warning in warnings:
                print(f"     - {warning}")
            total_warnings += len(warnings)
        
        # Check balance
        stats = check_dataset_balance(TARGET_PATH, dataset_name)
        
        if not issues:
            print_colored(f"  ✓ Structure is correct", 'green')
        
        print(f"\n  Dataset Statistics:")
        for phase in ['train', 'test']:
            if stats[phase]['attack'] + stats[phase]['live'] > 0:
                print(f"    {phase:5} - Attack: {stats[phase]['attack']:3d} videos, "
                      f"Live: {stats[phase]['live']:3d} videos")
                
                # Check balance
                if stats[phase]['attack'] == 0 or stats[phase]['live'] == 0:
                    print_colored(f"           ⚠️  Imbalanced: one class is missing!", 'yellow')
                else:
                    ratio = stats[phase]['attack'] / stats[phase]['live']
                    if ratio > 5 or ratio < 0.2:
                        print_colored(f"           ⚠️  Imbalanced: ratio is {ratio:.2f}", 'yellow')
        
        print()
    
    # Final summary
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if all_good and total_warnings == 0:
        print_colored("✓ All datasets are properly prepared and ready for training!", 'green')
    elif all_good:
        print_colored(f"✓ Datasets are usable, but there are {total_warnings} warnings to consider.", 'yellow')
    else:
        print_colored(f"❌ Found {total_issues} critical issues. Please fix them before training.", 'red')
    
    print()
    
    # Print available protocols
    available_datasets = []
    protocol_map = {
        '30k_fas': 'K',
        '98k_real': 'R',
        'archive': 'V',
        'unidata_real': 'U'
    }
    
    for dataset_name in EXPECTED_DATASETS:
        dataset_path = os.path.join(TARGET_PATH, dataset_name)
        if os.path.exists(dataset_path):
            available_datasets.append((dataset_name, protocol_map[dataset_name]))
    
    if available_datasets:
        print("Available datasets for training:")
        for name, code in available_datasets:
            print(f"  {code} = {name}")
        
        print()
        print("Example training commands:")
        if len(available_datasets) >= 3:
            codes = [d[1] for d in available_datasets]
            train_codes = '_'.join(codes[:-1])
            test_code = codes[-1]
            print(f"  python training.py --protocol {train_codes}_to_{test_code} --gs --log_name experiment1")
        
        if len(available_datasets) >= 2:
            print(f"  python training.py --protocol {available_datasets[0][1]}_to_{available_datasets[1][1]} --gs --log_name experiment2")
        
        if len(available_datasets) >= 1:
            print(f"  python training.py --protocol {available_datasets[0][1]}_to_{available_datasets[0][1]} --gs --log_name experiment3")
    
    print()
    print("For detailed training instructions, see KAGGLE_DATASET_SETUP.md")
    print("="*70)
    
    return all_good

if __name__ == "__main__":
    # Enable ANSI color codes on Windows
    os.system('')
    
    success = main()
    sys.exit(0 if success else 1)
