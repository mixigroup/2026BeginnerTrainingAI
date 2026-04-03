"""SigLIP2 マルチモーダル埋め込みハンズオン用ユーティリティ関数."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from transformers import AutoModel, AutoProcessor


def decode_image(image_field) -> Image.Image:
    """datasets ライブラリの image フィールドを PIL Image に変換する.

    floschne/xm3600 の image フィールドは {'bytes': b'...', 'path': ...} の dict で
    返される場合がある。PIL Image ならそのまま返す。

    Args:
        image_field: PIL Image または dict（bytes キー付き）

    Returns:
        PIL Image
    """
    if isinstance(image_field, Image.Image):
        return image_field
    if isinstance(image_field, dict) and "bytes" in image_field:
        return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
    raise ValueError(f"Unknown image format: {type(image_field)}")


def load_siglip_model(
    model_name: str = "google/siglip2-base-patch16-224",
) -> tuple[AutoModel, AutoProcessor]:
    """SigLIP2 モデルとプロセッサをロードする.

    Args:
        model_name: Hugging Face Hub 上のモデル名

    Returns:
        (model, processor) のタプル
    """
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def encode_images(
    model: AutoModel,
    processor: AutoProcessor,
    images: list[Image.Image],
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """PIL 画像リストを L2 正規化済み埋め込みベクトルに変換する.

    Args:
        model: SigLIP2 モデル
        processor: SigLIP2 プロセッサ
        images: PIL Image のリスト
        device: 推論デバイス
        batch_size: バッチサイズ

    Returns:
        (N, D) の numpy 配列（L2 正規化済み）
    """
    if not images:
        return np.empty((0, 0), dtype=np.float32)

    model = model.to(device)
    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.get_image_features(**inputs)
        # get_image_features は BaseModelOutputWithPooling を返す場合がある
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        # L2 正規化
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def encode_texts(
    model: AutoModel,
    processor: AutoProcessor,
    texts: list[str],
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """テキストリストを L2 正規化済み埋め込みベクトルに変換する.

    Args:
        model: SigLIP2 モデル
        processor: SigLIP2 プロセッサ
        texts: テキストのリスト
        device: 推論デバイス
        batch_size: バッチサイズ

    Returns:
        (N, D) の numpy 配列（L2 正規化済み）
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = model.to(device)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = processor(
            text=batch_texts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model.get_text_features(**inputs)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """L2 正規化済みベクトル間のコサイン類似度行列を計算する.

    正規化済みなので単純な行列積で計算できる。

    Args:
        emb_a: (N, D) の埋め込みベクトル
        emb_b: (M, D) の埋め込みベクトル

    Returns:
        (N, M) のコサイン類似度行列
    """
    return emb_a @ emb_b.T


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "類似度マトリクス",
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """類似度行列をヒートマップとして描画する.

    Args:
        sim_matrix: (N, M) の類似度行列
        row_labels: 行ラベル
        col_labels: 列ラベル
        title: グラフタイトル
        ax: matplotlib Axes（None なら新規作成）
        figsize: 図のサイズ

    Returns:
        matplotlib Figure オブジェクト
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="コサイン類似度")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=14)

    # セルに数値を表示（小さいマトリクスの場合のみ）
    if sim_matrix.shape[0] <= 15 and sim_matrix.shape[1] <= 15:
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                val = sim_matrix[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    return fig


def prepare_thumbnail_tensor(
    images: list[Image.Image],
    size: int = 64,
) -> torch.Tensor:
    """PIL 画像リストを TensorBoardX 用のサムネイルテンソルに変換する.

    Args:
        images: PIL Image のリスト
        size: サムネイルの一辺のサイズ（px）

    Returns:
        (N, 3, H, W) の float テンソル（値は [0, 1]）
    """
    tensors = []
    for img in images:
        img_rgb = img.convert("RGB").resize((size, size), Image.LANCZOS)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        tensors.append(tensor)
    return torch.stack(tensors)


def create_text_thumbnail(
    text: str,
    size: int = 64,
    bg_color: tuple = (50, 50, 80),
    text_color: tuple = (255, 255, 255),
) -> Image.Image:
    """テキストからサムネイル画像を生成する（TensorBoardX 用）.

    Args:
        text: 表示するテキスト
        size: 画像サイズ（px）
        bg_color: 背景色 RGB
        text_color: 文字色 RGB

    Returns:
        PIL Image
    """
    img = Image.new("RGB", (size, size), bg_color)
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        # テキストを折り返して表示
        short = text[:20]
        draw.text((2, size // 3), short, fill=text_color)
    except Exception:
        pass
    return img


def export_embeddings_to_tensorboard(
    writer,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_labels: list[str],
    text_labels: list[str],
    images: list[Image.Image] | None = None,
    tag: str = "siglip2_embeddings",
) -> None:
    """画像・テキスト埋め込みを TensorBoardX にエクスポートする.

    画像とテキストの埋め込みを結合し、ラベル付きで書き出す。
    TensorBoard の PROJECTOR タブで PCA/t-SNE による3D可視化が可能。

    Args:
        writer: TensorBoardX の SummaryWriter
        image_embeddings: (N_img, D) の画像埋め込み
        text_embeddings: (N_txt, D) のテキスト埋め込み
        image_labels: 画像のラベルリスト
        text_labels: テキストのラベルリスト
        images: PIL Image のリスト（サムネイル用、None なら省略）
        tag: TensorBoard 上のタグ名
    """
    # 埋め込みを結合
    combined_embeddings = np.concatenate(
        [image_embeddings, text_embeddings], axis=0
    )

    # メタデータ（モダリティ + ラベル）
    metadata_header = ["label", "modality"]
    metadata = []
    for label in image_labels:
        metadata.append([label, "image"])
    for label in text_labels:
        metadata.append([label, "text"])

    # サムネイル画像の準備
    label_img = None
    if images is not None:
        # 画像のサムネイル
        img_thumbnails = prepare_thumbnail_tensor(images, size=64)
        # テキスト用のプレースホルダーサムネイル
        text_thumbnails = []
        for text in text_labels:
            text_thumbnails.append(create_text_thumbnail(text, size=64))
        txt_tensor = prepare_thumbnail_tensor(text_thumbnails, size=64)
        label_img = torch.cat([img_thumbnails, txt_tensor], dim=0)

    # TensorBoardX にエクスポート
    combined_tensor = torch.from_numpy(combined_embeddings)

    writer.add_embedding(
        mat=combined_tensor,
        metadata=metadata,
        metadata_header=metadata_header,
        label_img=label_img,
        tag=tag,
    )
