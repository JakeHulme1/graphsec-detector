import os
import math
from typing import Dict, List
import matplotlib.pyplot as plt

def plot_training(
    histories: Dict[str, List[float]],
    out_path: str
) -> None:
    """
    Plot training and validation metrics with one subplot per metric.

    Args:
        histories: dict with keys like 'train_accuracy', 'val_accuracy', etc.,
                   each mapping to a list of floats for each epoch.
        out_path: filepath where to save the PNG figure.
    """
    # Sanity checks
    assert isinstance(histories, dict), "histories must be a dict"
    # Define which metrics to plot
    metrics = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    for metric in metrics:
        for phase in ["train", "val"]:
            key = f"{phase}_{metric}"
            assert key in histories, f"Missing '{key}' in histories"
            series = histories[key]
            assert isinstance(series, list), f"{key} must be a list"
            assert all(isinstance(x, (int, float)) for x in series), \
                f"All values in {key} must be numeric"

    # Setup subplots
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes_flat = axes.flatten()

    # X-axis: epochs
    epochs = range(1, len(histories["train_accuracy"]) + 1)

    # Plot each metric in its own subplot
    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        ax.plot(epochs, histories[f"train_{metric}"], label=f"train_{metric}")
        ax.plot(epochs, histories[f"val_{metric}"],   label=f"val_{metric}")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()

    # Hide any unused subplots
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].axis("off")

    # Save figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

