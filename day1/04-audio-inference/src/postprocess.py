"""token ids → テキストへのデコード（後処理）"""

from __future__ import annotations

import torch
from transformers import AutoProcessor


def load_processor(model_name: str) -> AutoProcessor:
    """Processor（Feature Extractor + Tokenizer）を読み込む。

    AutoProcessor は Feature Extractor と Tokenizer を一つにまとめたクラス。
    後処理では Tokenizer 部分（processor.tokenizer）を使う。

    Args:
        model_name: HuggingFace のモデル名（例: "openai/whisper-small"）

    Returns:
        AutoProcessor インスタンス
    """
    return AutoProcessor.from_pretrained(model_name)


def decode_tokens(processor: AutoProcessor, token_ids: torch.Tensor) -> str:
    """token ids をテキスト文字列に変換する。

    Whisper の tokenizer で生成された token ids を
    human-readable なテキストにデコードする。
    特殊トークン（<|startoftranscript|> など）は除去される。

    Args:
        processor: load_processor() で取得した Processor
        token_ids: generate_tokens() の出力（shape: (1, seq_len)）

    Returns:
        デコードされたテキスト文字列
    """
    transcription = processor.batch_decode(token_ids, skip_special_tokens=True)
    return transcription[0]
