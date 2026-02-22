"""
Training and evaluation utilities for model training hands-on.

Provides:
- train_one_epoch / evaluate: core training loop primitives
- train_model: full training loop with optional early stopping
- plot_learning_curves: visualize train/val loss and accuracy
- plot_confusion_matrix: visualize classification results
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """Container for training and validation metrics per epoch."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    device: torch.device | None = None,
    early_stopping_patience: int | None = None,
    verbose: bool = True,
    verbose_interval: int = 10,
) -> TrainingHistory:
    """Full training loop with optional early stopping.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Maximum number of training epochs.
        device: Device to run on (auto-detected if None).
        early_stopping_patience: Stop if val_loss does not improve for this many epochs.
            None means no early stopping.
        verbose: Whether to print progress.
        verbose_interval: Print progress every N epochs.

    Returns:
        TrainingHistory with per-epoch metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    history = TrainingHistory()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_accs.append(train_acc)
        history.val_accs.append(val_acc)

        if verbose and (epoch % verbose_interval == 0 or epoch == 1):
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # Early stopping check.
        # NOTE: This implementation stops at the last epoch, not necessarily the best epoch.
        # For production use, save model weights with copy.deepcopy(model.state_dict())
        # when val_loss improves, and restore them after stopping.
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                    break

    return history


def plot_learning_curves(
    history: TrainingHistory,
    title: str = "Learning Curves",
) -> plt.Figure:
    """Plot train/val loss and accuracy curves.

    Args:
        history: TrainingHistory from train_model().
        title: Figure title.

    Returns:
        matplotlib Figure object.
    """
    epochs = range(1, len(history.train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history.train_losses, label="Train")
    axes[0].plot(epochs, history.val_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history.train_accs, label="Train")
    axes[1].plot(epochs, history.val_accs, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def compare_learning_curves(
    histories: dict[str, TrainingHistory],
    metric: str = "val_loss",
) -> plt.Figure:
    """Compare multiple training runs on a single plot.

    Args:
        histories: Dict mapping run name to TrainingHistory.
        metric: One of 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    attr_map = {
        "train_loss": "train_losses",
        "val_loss": "val_losses",
        "train_acc": "train_accs",
        "val_acc": "val_accs",
    }
    if metric not in attr_map:
        raise ValueError(f"Unknown metric: {metric!r}. Choose from {list(attr_map)}")

    for name, history in histories.items():
        values = getattr(history, attr_map[metric])
        ax.plot(range(1, len(values) + 1), values, label=name)

    ylabel_map = {
        "train_loss": "Train Loss",
        "val_loss": "Validation Loss",
        "train_acc": "Train Accuracy",
        "val_acc": "Validation Accuracy",
    }
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.set_title(f"Comparison: {ylabel_map.get(metric, metric)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
    import seaborn as sns

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
