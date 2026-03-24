"""音声の前処理（リサンプリング・mel spectrogram変換・可視化）"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.figure
from transformers import AutoFeatureExtractor


TARGET_SAMPLING_RATE = 16000  # Whisper が要求するサンプリングレート


def load_sample_audio(
    dataset_name: str = "google/fleurs",
    language: str = "ja_jp",
    split: str = "test",
    index: int = 0,
) -> tuple[np.ndarray, int, str]:
    """HuggingFace datasets から日本語音声サンプルを取得する。

    Args:
        dataset_name: HuggingFace のデータセット名
        language: 言語コード（FLEURS では "ja_jp" など）
        split: データセットのスプリット
        index: 取得するサンプルのインデックス

    Returns:
        audio_array: 音声データ（numpy 配列、shape: (num_samples,)）
        sampling_rate: 元のサンプリングレート
        transcription: 正解テキスト
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, language, split=split)
    sample = dataset[index]

    audio = sample["audio"]
    audio_array = np.array(audio["array"], dtype=np.float32)
    sampling_rate = audio["sampling_rate"]
    transcription = sample["transcription"]

    return audio_array, sampling_rate, transcription


def load_feature_extractor(model_name: str) -> AutoFeatureExtractor:
    """Feature Extractor（log-mel spectrogram 変換器）を読み込む。

    Args:
        model_name: HuggingFace のモデル名（例: "openai/whisper-small"）

    Returns:
        AutoFeatureExtractor インスタンス
    """
    return AutoFeatureExtractor.from_pretrained(model_name)


def resample_audio(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLING_RATE,
) -> np.ndarray:
    """音声のサンプリングレートを変換する。

    Whisper は 16kHz を期待するため、異なるサンプリングレートの音声を変換する。

    Args:
        audio_array: 入力音声データ（numpy 配列）
        orig_sr: 元のサンプリングレート（Hz）
        target_sr: 変換後のサンプリングレート（Hz）、デフォルト 16000

    Returns:
        リサンプリング後の音声データ（numpy 配列）
    """
    if orig_sr == target_sr:
        return audio_array

    audio_tensor = torch.tensor(audio_array).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(audio_tensor)
    return resampled.squeeze(0).numpy()


def extract_features(
    feature_extractor: AutoFeatureExtractor,
    audio_array: np.ndarray,
    sampling_rate: int = TARGET_SAMPLING_RATE,
) -> dict:
    """音声から log-mel spectrogram を抽出する。

    Args:
        feature_extractor: load_feature_extractor() で取得したインスタンス
        audio_array: リサンプリング済みの音声データ（numpy 配列）
        sampling_rate: 音声のサンプリングレート（Hz）

    Returns:
        input_features を含む辞書
        - input_features shape: (1, 80, 3000) → 80 mel 帯域 × 30秒分
    """
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    return inputs


def visualize_mel_spectrogram(
    input_features: torch.Tensor,
    title: str = "Log-Mel Spectrogram",
) -> matplotlib.figure.Figure:
    """log-mel spectrogram を可視化する。

    Args:
        input_features: extract_features() の出力（shape: (1, 80, 3000)）
        title: グラフのタイトル

    Returns:
        matplotlib Figure オブジェクト
    """
    # (1, 80, 3000) → (80, 3000) に変換（GPU テンソルにも対応）
    spec = input_features[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("時間フレーム（1フレーム = 10ms）")
    ax.set_ylabel("Mel 帯域（0〜80）")
    ax.set_title(title)
    plt.colorbar(img, ax=ax, label="対数振幅")
    fig.tight_layout()
    return fig
