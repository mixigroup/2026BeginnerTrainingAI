"""
Evaluation utilities for model training hands-on.

Provides:
- evaluate: compute loss and accuracy on a DataLoader
- plot_confusion_matrix: visualize classification results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader.

    Args:
        model: PyTorch model.
        loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot confusion matrix for model predictions.

    Args:
        model: PyTorch model (already trained).
        loader: DataLoader for evaluation data.
        class_names: List of class label strings.
        device: Device to run on (auto-detected if None).
        title: Plot title.

    Returns:
        matplotlib Figure object.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig
