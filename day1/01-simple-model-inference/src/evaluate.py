"""Evaluation utilities for Iris classification hands-on."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_learning_curves(history) -> plt.Figure:
    """Plot training and validation loss / accuracy curves.

    Args:
        history: Keras History object returned by model.fit()

    Returns:
        matplotlib Figure with two subplots (loss and accuracy)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="valid loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="valid accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def evaluate_nn(model, X_test: np.ndarray, y_test_onehot: np.ndarray) -> float:
    """Evaluate a Neural Network model on test data.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test_onehot: Test labels in one-hot encoded format

    Returns:
        Accuracy as a percentage (0-100)
    """
    prediction = model.predict(X_test)
    y_label = np.argmax(y_test_onehot, axis=1)
    predict_label = np.argmax(prediction, axis=1)
    accuracy = np.sum(y_label == predict_label) / len(prediction) * 100
    return accuracy


def get_nn_predictions(model, X_test: np.ndarray, y_test_onehot: np.ndarray):
    """Get predicted and true labels from a Neural Network model.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test_onehot: Test labels in one-hot encoded format

    Returns:
        y_true: True labels (integer class indices)
        y_pred: Predicted labels (integer class indices)
    """
    prediction = model.predict(X_test)
    y_true = np.argmax(y_test_onehot, axis=1)
    y_pred = np.argmax(prediction, axis=1)
    return y_true, y_pred


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: True class labels (integer indices)
        y_pred: Predicted class labels (integer indices)
        target_names: List of class name strings

    Returns:
        matplotlib Figure with the confusion matrix heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> str:
    """Generate a classification report string.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        target_names: List of class name strings

    Returns:
        Classification report as a formatted string
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    return f"Accuracy: {acc * 100:.2f}%\n\n{report}"
