from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_training_curves(
    history: dict[str, list[float]],
    out_path: str | Path,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize_first_layer_weights(
    w1: np.ndarray,
    out_path: str | Path,
    image_size: int = 64,
    num_show: int = 16,
) -> None:
    n = min(num_show, w1.shape[1])
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        filt = w1[:, i].reshape(image_size, image_size, 3)
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        axes[i].imshow(filt)
        axes[i].set_title(f"n{i}", fontsize=8)
    fig.suptitle("First Layer Weights")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize_errors(
    errors: list[tuple[str, int, int]],
    class_names: list[str],
    out_path: str | Path,
    max_items: int = 12,
) -> None:
    subset = errors[:max_items]
    if not subset:
        return
    cols = 4
    rows = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (pth, gt, pd) in enumerate(subset):
        img = Image.open(pth).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(f"T:{class_names[gt]} P:{class_names[pd]}", fontsize=8)
    fig.suptitle("Misclassified Examples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

