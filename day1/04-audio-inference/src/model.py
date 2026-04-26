"""Whisper モデルのロードと推論（encoder-decoder 構造）"""

from __future__ import annotations

import torch
from transformers import AutoModelForSpeechSeq2Seq


def load_model(model_name: str) -> AutoModelForSpeechSeq2Seq:
    """Whisper モデルをロードして推論モードで返す。

    Args:
        model_name: HuggingFace のモデル名（例: "openai/whisper-small"）

    Returns:
        AutoModelForSpeechSeq2Seq インスタンス（eval モード）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, device_map=device)
    model.eval()
    return model


def encode_audio(
    model: AutoModelForSpeechSeq2Seq,
    input_features: torch.Tensor,
):
    """Encoder フェーズ：音声特徴量を内部表現（encoder hidden states）に変換する。

    Whisper の encoder は CNN + Transformer Encoder で構成されており、
    log-mel spectrogram を固定長の内部表現（sequence of vectors）に変換する。

    Args:
        model: load_model() で取得した Whisper モデル
        input_features: extract_features() の出力（shape: (1, 80, 3000)）

    Returns:
        encoder_outputs: last_hidden_state など encoder の出力
        - last_hidden_state shape: (1, 1500, hidden_size)
    """
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features.to(model.device))
    return encoder_outputs


def generate_tokens(
    model: AutoModelForSpeechSeq2Seq,
    inputs: dict,
    language: str = "japanese",
) -> torch.Tensor:
    """Decoder フェーズ：encoder の出力から自己回帰的にトークン列を生成する。

    Whisper の decoder は Transformer Decoder で構成されており、
    encoder の出力をもとに1トークンずつ生成（自己回帰生成）する。

    Args:
        model: load_model() で取得した Whisper モデル
        inputs: extract_features() の出力（input_features を含む辞書）
        language: 認識言語（"japanese", "english" など）

    Returns:
        predicted_ids: 生成されたトークン列（shape: (1, seq_len)）
    """
    # `language` は Whisper の forced_decoder_ids を通じて制御される。
    # transformers 4.x 以降では generate() に直接渡せる。
    # 指定しない場合は Whisper が自動で言語を検出する。
    inputs_on_device = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        predicted_ids = model.generate(
            **inputs_on_device,
            language=language,
        )
    return predicted_ids
