import matplotlib.pyplot as plt
import re
import os


def plot_training_loss():
    # --- CONFIGURATION ---
    # Path to your log file (Adjust if your folder name is different)
    log_file_path = "results/Run_IR_Only/CASIA_IR.txt"

    # Check if file exists
    if not os.path.exists(log_file_path):
        print(f"❌ Error: Could not find log file at: {log_file_path}")
        print("Make sure you are running this from the project root (GD-FAS folder).")
        return

    print(f"Reading log file: {log_file_path}...")

    iterations = []
    losses = []

    # Regex pattern to extract "Iter" and "total" numbers
    # Matches lines like: "Iter 200: ... total 1.6671 ..."
    pattern = re.compile(r"Iter\s+(\d+):.*?total\s+([\d\.]+)")

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Group 1 is Iteration, Group 2 is Total Loss
                iterations.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    if not iterations:
        print("⚠️ Warning: No training data found in the log file.")
        print("Check if the log file format matches 'Iter X: ... total Y ...'")
        return

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))

    # Plot the line
    plt.plot(iterations, losses, label='Total Loss', color='#007acc', linewidth=2)

    # Highlight the specific data points
    plt.scatter(iterations, losses, color='#005f9e', s=30, zorder=5)

    # Add text labels for the most recent 5 points (to avoid clutter)
    for i in range(max(0, len(iterations) - 5), len(iterations)):
        plt.annotate(f"{losses[i]:.4f}",
                     (iterations[i], losses[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#007acc", alpha=0.8))

    plt.title(f'Training Loss: CASIA_IR Protocol', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot to a file
    output_img = "training_loss_plot.png"
    plt.savefig(output_img)
    print(f"✅ Plot saved as '{output_img}'")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot_training_loss()