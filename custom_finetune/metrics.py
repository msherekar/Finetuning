# metrics.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

def plot_learning_curves(train_losses, val_losses, fold, output_dir):
    plt.figure()
    epochs = np.arange(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_loss_curve.png"), dpi=300)
    plt.close()

def generate_text_report(report_data, output_file):
    """
    report_data: dict containing keys for each fold and overall metrics.
    """
    with open(output_file, "w") as f:
        for fold, metrics in report_data.items():
            f.write(f"Fold {fold}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    print(f"Text report saved to {output_file}")
