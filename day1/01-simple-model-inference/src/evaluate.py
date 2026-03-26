"""推論結果の評価ユーティリティ"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """混同行列のヒートマップを描画する。"""
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
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> str:
    """分類レポート文字列を生成する。"""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    return f"Accuracy: {acc * 100:.2f}%\n\n{report}"
