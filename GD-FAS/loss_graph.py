import matplotlib.pyplot as plt
import re
import os

# 1. Define the path to your log file
# Based on your previous run, it should be here:
file_path = '/content/Domain-Invariance-Vision-Transformer/GD-FAS/results/Run_Fold1/Custom_to_Custom.txt'

# 2. Storage for the data
epochs = []
class_losses = []
domain_losses = []
total_losses = []

# 3. Check if file exists, then parse it
if os.path.exists(file_path):
    print(f"Reading log file from: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # We look for lines starting with 'epoch :'
        # Pattern: epoch : 42 ... class 0.0316 domain 0.1578 total 0.4973 ...
        if line.strip().startswith('epoch :'):
            try:
                # Extract numbers using Regex
                epoch_match = re.search(r'epoch\s*:\s*(\d+)', line)
                class_match = re.search(r'class\s*([\d\.]+)', line)
                domain_match = re.search(r'domain\s*([\d\.]+)', line)
                total_match = re.search(r'total\s*([\d\.]+)', line)

                if epoch_match and class_match and domain_match and total_match:
                    epochs.append(int(epoch_match.group(1)))
                    class_losses.append(float(class_match.group(1)))
                    domain_losses.append(float(domain_match.group(1)))
                    total_losses.append(float(total_match.group(1)))
            except Exception as e:
                print(f"Skipping line due to error: {line.strip()}")

    # 4. Plotting
    if epochs:
        plt.figure(figsize=(12, 6))

        # Plot Class Loss
        plt.plot(epochs, class_losses, label='Class Loss', color='blue', linewidth=2)

        # Plot Domain Loss
        plt.plot(epochs, domain_losses, label='Domain Loss', color='green', linewidth=2)

        # Plot Total Loss (dashed to distinguish it)
        plt.plot(epochs, total_losses, label='Total Loss', color='red', linestyle='--', linewidth=2)

        plt.title('Training Loss over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()
        print(f"Successfully plotted data for {len(epochs)} epochs.")
    else:
        print("No valid epoch data found in the log file.")
else:
    print(f"File not found at {file_path}. Please check the path.")