import matplotlib.pyplot as plt

def save_training_graphs(history, filename="training_results_graph.png"):
    """
    Generates and saves training vs validation graphs.
    Args:
        history (dict): Must contain 'train_loss', 'val_loss', and 'val_acc' lists.
        filename (str): Output filename for the graph.
    """
    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # --- Subplot 1: Loss Curves ---
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Subplot 2: Accuracy Curve ---
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Liveness Accuracy', color='green', marker='o')
    plt.title('Validation Accuracy (Real vs Fake)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Save to File ---
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # Close memory to prevent leaks
    print(f"[GRAPH] Updated and saved to '{filename}'")