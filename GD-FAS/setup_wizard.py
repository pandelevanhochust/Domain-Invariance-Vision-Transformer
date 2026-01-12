"""
Interactive Setup Wizard for GD-FAS with Kaggle Datasets
This script guides you through the setup process step by step.
"""

import os
import subprocess
import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step_num, text):
    """Print a step number."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {text}")
    print(f"{'='*70}\n")

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {command}\n")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during {description}")
        print(f"Error: {e}")
        return False

def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"⚠️  Warning: Python {version.major}.{version.minor} detected.")
        print("   Python 3.8+ is recommended.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected.")
    return True

def check_gpu():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected. Training will be slow on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. Will check GPU after installation.")
        return None

def main():
    """Main setup wizard."""
    print_header("GD-FAS Setup Wizard for Kaggle Datasets")
    
    print("This wizard will guide you through:")
    print("  1. Checking system requirements")
    print("  2. Installing dependencies")
    print("  3. Preparing your datasets")
    print("  4. Verifying the setup")
    print("  5. Providing training examples")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check Python version
    print_step(1, "Checking System Requirements")
    
    if not check_python_version():
        print("\nSetup cancelled.")
        return
    
    gpu_available = check_gpu()
    
    if gpu_available is False:
        print("\n⚠️  Warning: Training without GPU will be very slow!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nSetup cancelled. Please ensure CUDA is properly installed.")
            return
    
    # Step 2: Install dependencies
    print_step(2, "Installing Dependencies")
    
    print("This will install the required Python packages:")
    print("  - PyTorch, NumPy, scikit-learn")
    print("  - OpenCV (for video processing)")
    print("  - Pillow, tqdm, and others")
    
    response = input("\nInstall dependencies now? (y/n): ")
    if response.lower() == 'y':
        if not run_command("pip install -r requirement.txt", "Installing dependencies"):
            print("\n⚠️  Installation failed. Please check the error messages above.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Check GPU again after PyTorch installation
        print("\nRechecking GPU availability...")
        check_gpu()
    else:
        print("Skipping dependency installation.")
        print("⚠️  Make sure to install them manually: pip install -r requirement.txt")
    
    # Step 3: Prepare datasets
    print_step(3, "Preparing Datasets")
    
    print("This will process your Kaggle datasets:")
    print("  - Extract frames from videos (every 5th frame)")
    print("  - Organize into train/test splits (80/20)")
    print("  - Create proper directory structure")
    print("\n⏱️  This may take 15-60 minutes depending on data size!")
    
    source_path = r"D:\CODIng\Machine Learning\FAS\datasets"
    if not os.path.exists(source_path):
        print(f"\n⚠️  Warning: Source datasets not found at {source_path}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✓ Source datasets found at {source_path}")
    
    response = input("\nPrepare datasets now? (y/n): ")
    if response.lower() == 'y':
        if not run_command("python prepare_kaggle_dataset.py", "Preparing datasets"):
            print("\n⚠️  Dataset preparation failed.")
            response = input("Continue to verification anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        print("Skipping dataset preparation.")
        print("⚠️  You'll need to run 'python prepare_kaggle_dataset.py' manually.")
    
    # Step 4: Verify setup
    print_step(4, "Verifying Setup")
    
    print("This will check if datasets are properly prepared.")
    
    response = input("\nRun verification now? (y/n): ")
    if response.lower() == 'y':
        run_command("python verify_dataset.py", "Verifying datasets")
    else:
        print("Skipping verification.")
        print("⚠️  You can run 'python verify_dataset.py' manually later.")
    
    # Step 5: Training examples
    print_step(5, "Training Examples")
    
    print("Your datasets are ready! Here are some training examples:\n")
    
    print("Example 1: Cross-domain generalization (Recommended)")
    print("  python training.py --protocol K_R_U_to_V --gs --log_name experiment1")
    print("  → Trains on 30k_fas, 98k_real, unidata_real; Tests on archive\n")
    
    print("Example 2: Attack-rich training")
    print("  python training.py --protocol K_V_to_R --gs --log_name experiment2 --save")
    print("  → Trains on 30k_fas, archive; Tests on 98k_real\n")
    
    print("Example 3: Quick test (fast)")
    print("  python training.py --protocol K_to_K --gs --log_name quick_test --max_iter 100")
    print("  → Quick pipeline test\n")
    
    print("\nDataset codes:")
    print("  K = 30k_fas")
    print("  R = 98k_real")
    print("  V = archive")
    print("  U = unidata_real")
    
    print("\n" + "="*70)
    response = input("\nStart training now? (y/n): ")
    if response.lower() == 'y':
        print("\nWhich example would you like to run?")
        print("  1 - Cross-domain generalization (K_R_U_to_V)")
        print("  2 - Attack-rich training (K_V_to_R)")
        print("  3 - Quick test (K_to_K)")
        print("  4 - Custom (enter your own protocol)")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            command = "python training.py --protocol K_R_U_to_V --gs --log_name experiment1"
        elif choice == '2':
            command = "python training.py --protocol K_V_to_R --gs --log_name experiment2 --save"
        elif choice == '3':
            command = "python training.py --protocol K_to_K --gs --log_name quick_test --max_iter 100"
        elif choice == '4':
            protocol = input("Enter protocol (e.g., K_R_to_V): ")
            log_name = input("Enter log name (e.g., my_experiment): ")
            command = f"python training.py --protocol {protocol} --gs --log_name {log_name}"
        else:
            print("Invalid choice. Exiting.")
            return
        
        print(f"\nStarting training with command:")
        print(f"  {command}\n")
        
        run_command(command, "Training model")
    else:
        print("\nSetup complete! You can start training manually when ready.")
    
    # Final summary
    print_header("Setup Complete!")
    
    print("📚 Documentation:")
    print("  - Quick Start: QUICK_START_KAGGLE.md")
    print("  - Detailed Guide: KAGGLE_DATASET_SETUP.md")
    print("  - Overview: README_KAGGLE.md")
    print("  - Changes: CHANGES_SUMMARY.md")
    
    print("\n🔧 Useful Commands:")
    print("  - Verify datasets: python verify_dataset.py")
    print("  - Train model: python training.py --protocol K_R_U_to_V --gs --log_name my_exp")
    print("  - Check results: Look in results/{log_name}/ folder")
    
    print("\n💡 Tips:")
    print("  - Use --save to save the best model checkpoint")
    print("  - Reduce --batch_size if you get out of memory errors")
    print("  - Check training logs in results/ directory")
    
    print("\n" + "="*70)
    print("Happy Training! 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("Please check the error message and try again.")
        sys.exit(1)
        sys.exit(1)
