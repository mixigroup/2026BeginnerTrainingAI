"""
Standalone training script for Iris classification (no notebook required).

Usage:
    uv run python train_iris.py
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures


def main() -> None:
    sys.path.insert(0, ".")

    from src.dataset import load_iris_dataloaders
    from src.model import FCNet
    from src.evaluate import (
        train_model,
        evaluate,
        plot_learning_curves,
        plot_confusion_matrix,
    )

    # Hyperparameters
    BATCH_SIZE = 16
    HIDDEN_DIMS = [64, 32]
    LEARNING_RATE = 0.01
    EPOCHS = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    train_loader, val_loader, test_loader, class_names = load_iris_dataloaders(
        batch_size=BATCH_SIZE, val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    print(
        f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    # 2. Build model
    model = FCNet(input_dim=4, hidden_dims=HIDDEN_DIMS, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train
    print("Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        verbose=True,
        verbose_interval=20,
    )

    # 4. Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.1f}%")

    # 5. Plot and save
    import matplotlib.pyplot as plt

    fig_curves = plot_learning_curves(history, title="Iris FC-Net")
    fig_curves.savefig("learning_curves_iris.png", dpi=100, bbox_inches="tight")
    plt.close(fig_curves)
    print("Saved: learning_curves_iris.png")

    fig_cm = plot_confusion_matrix(model, test_loader, class_names, device=device)
    fig_cm.savefig("confusion_matrix_iris.png", dpi=100, bbox_inches="tight")
    plt.close(fig_cm)
    print("Saved: confusion_matrix_iris.png")


if __name__ == "__main__":
    main()
