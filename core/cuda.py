import torch
import sys


def check_cuda():
    print(f"--- CUDA & GPU Check ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    is_available = torch.cuda.is_available()
    print(f"\nCUDA Available:  {'✅ YES' if is_available else '❌ NO'}")

    if is_available:
        print(f"CUDA Version:    {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count:       {device_count}")

        for i in range(device_count):
            print(f"GPU {i}:          {torch.cuda.get_device_name(i)}")

        # Test a small tensor operation to ensure the driver is actually responding
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            print("\nTensor Test:     ✅ Passed (Tensor moved to GPU successfully)")
        except Exception as e:
            print(f"\nTensor Test:     ❌ Failed ({e})")
    else:
        print("\nPossible fixes:")
        print("1. Run 'nvidia-smi' in terminal to check drivers.")
        print("2. Reinstall PyTorch with: pip install torch --index-url https://download.pytorch.org/whl/cu121")


if __name__ == "__main__":
    check_cuda()