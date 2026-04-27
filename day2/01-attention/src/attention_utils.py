"""Attention メカニズム可視化ハンズオン用ユーティリティ関数."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import torch


def compute_attention_weights(
    query_vec: np.ndarray,
    key_vecs: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Q・K 内積 → softmax で Attention 重みを計算する.

    Args:
        query_vec: Query ベクトル (D,)
        key_vecs: Key ベクトル (N, D)
        temperature: 温度パラメータ（高い = 均一、低い = 鋭い）

    Returns:
        Attention 重み (N,)。合計 1.0。
    """
    scores = key_vecs @ query_vec
    scaled = scores / temperature
    scaled -= scaled.max()
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum()


def compute_context_vector(
    weights: np.ndarray,
    value_vecs: np.ndarray,
) -> np.ndarray:
    """Attention 重みで Value ベクトルの加重平均を計算する.

    Args:
        weights: Attention 重み (N,)
        value_vecs: Value ベクトル (N, D)

    Returns:
        コンテキストベクトル (D,)
    """
    return weights @ value_vecs


def plot_attention_barplot(
    tokens: list[str],
    weights: np.ndarray,
    query_word: str,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Attention 重みを棒グラフで可視化する.

    Args:
        tokens: トークン（キーワード）のリスト
        weights: 各トークンの Attention 重み
        query_word: Query として使った単語
        ax: matplotlib Axes（None なら新規作成）
        figsize: 図のサイズ

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.barplot(x=list(tokens), y=weights, palette="viridis", ax=ax)
    ax.set_title(f"Attention Weights for Query: '{query_word}'", fontsize=14)
    ax.set_ylabel("Attention Weight")

    for i, w in enumerate(weights):
        ax.text(i, w + 0.005, f"{w:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def load_bert_japanese(
    model_name: str = "cl-tohoku/bert-base-japanese-v3",
) -> tuple:
    """日本語 BERT モデルとトークナイザをロードする.

    Args:
        model_name: Hugging Face Hub 上のモデル名

    Returns:
        (model, tokenizer) のタプル
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).eval()
    return model, tokenizer


def get_bert_attentions(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """テキストから BERT の Attention テンソルとトークンリストを取得する.

    Args:
        model: BERT モデル（output_attentions=True で初期化済み）
        tokenizer: BERT トークナイザ
        text: 入力テキスト
        device: 推論デバイス

    Returns:
        attentions: (num_layers, num_heads, seq_len, seq_len) の Tensor
        tokens: トークン文字列のリスト
    """
    import torch

    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    attention_tuple = outputs.attentions
    attentions = torch.stack(attention_tuple).squeeze(1)

    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    return attentions, tokens


def plot_attention_heatmap(
    attentions: torch.Tensor,
    tokens: list[str],
    layer: int,
    head: int,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """単一の Attention ヘッドをヒートマップで可視化する.

    Args:
        attentions: (num_layers, num_heads, seq_len, seq_len) の Tensor
        tokens: トークン文字列のリスト
        layer: 表示するレイヤー番号
        head: 表示するヘッド番号
        ax: matplotlib Axes（None なら新規作成）
        figsize: 図のサイズ

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    attn = attentions[layer, head].cpu().numpy()

    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        vmin=0,
        ax=ax,
        square=True,
    )
    ax.set_title(f"Layer {layer}, Head {head}", fontsize=12)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig


def plot_attention_heads_grid(
    attentions: torch.Tensor,
    tokens: list[str],
    layer: int,
    figsize: tuple[int, int] = (20, 15),
) -> plt.Figure:
    """1つのレイヤーの全ヘッドを 3x4 グリッドで表示する.

    Args:
        attentions: (num_layers, num_heads, seq_len, seq_len) の Tensor
        tokens: トークン文字列のリスト
        layer: 表示するレイヤー番号
        figsize: 図のサイズ

    Returns:
        matplotlib Figure
    """
    num_heads = attentions.shape[1]
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(
        f"Layer {layer}: 全 {num_heads} ヘッドの Attention パターン", fontsize=16
    )

    for h in range(num_heads):
        r, c = divmod(h, cols)
        attn = attentions[layer, h].cpu().numpy()
        ax = axes[r, c]
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            vmin=0,
            ax=ax,
            square=True,
            cbar=False,
        )
        ax.set_title(f"Head {h}", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)

    fig.tight_layout()
    return fig


def plot_attention_summary(
    attentions: torch.Tensor,
    tokens: list[str],
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """全層・全ヘッドの平均 Attention をヒートマップで表示する.

    Args:
        attentions: (num_layers, num_heads, seq_len, seq_len) の Tensor
        tokens: トークン文字列のリスト
        ax: matplotlib Axes（None なら新規作成）
        figsize: 図のサイズ

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    avg_attn = attentions.mean(dim=(0, 1)).cpu().numpy()

    sns.heatmap(
        avg_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        vmin=0,
        ax=ax,
        square=True,
    )
    ax.set_title("平均 Attention（全層・全ヘッド）", fontsize=14)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig


def plot_cls_attention(
    attentions: torch.Tensor,
    tokens: list[str],
    layer: int | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """[CLS] トークンの Attention 分布を棒グラフで表示する.

    Args:
        attentions: (num_layers, num_heads, seq_len, seq_len) の Tensor
        tokens: トークン文字列のリスト
        layer: 表示するレイヤー（None なら全層平均）
        ax: matplotlib Axes（None なら新規作成）
        figsize: 図のサイズ

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if layer is not None:
        cls_attn = attentions[layer, :, 0, :].mean(dim=0).cpu().numpy()
        title = f"[CLS] の Attention 分布 (Layer {layer}, 全ヘッド平均)"
    else:
        cls_attn = attentions[:, :, 0, :].mean(dim=(0, 1)).cpu().numpy()
        title = "[CLS] の Attention 分布 (全層・全ヘッド平均)"

    sns.barplot(x=list(tokens), y=cls_attn, palette="viridis", ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Attention Weight")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig
