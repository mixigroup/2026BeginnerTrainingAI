"""感情分析（Sentiment Analysis）の前処理・推論・後処理ユーティリティ"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 評価用サンプルデータ（文章, 正解ラベル）
# 正解ラベルは Positive / Negative の2値
EVAL_DATA: list[tuple[str, str]] = [
    # ポジティブ（10件）
    ("料理が美味しかった", "Positive"),
    ("また行きたい", "Positive"),
    ("スタッフがとても親切だった", "Positive"),
    ("雰囲気が最高でした", "Positive"),
    ("値段の割においしかった", "Positive"),
    ("友達に勧めたい", "Positive"),
    ("盛り付けが素敵でした", "Positive"),
    ("接客が丁寧でとても良かった", "Positive"),
    ("新鮮な食材で美味しかった", "Positive"),
    ("また絶対来ます", "Positive"),
    # ネガティブ（10件）
    ("サービスが最悪", "Negative"),
    ("待ち時間が長すぎた", "Negative"),
    ("料理が冷めていた", "Negative"),
    ("値段が高いわりに量が少ない", "Negative"),
    ("店員の態度が悪かった", "Negative"),
    ("予約したのに待たされた", "Negative"),
    ("料理が口に合わなかった", "Negative"),
    ("二度と行かない", "Negative"),
    ("まあまあだった", "Negative"),  # モデルが迷いやすい例
    ("普通でした", "Negative"),  # モデルが迷いやすい例
]


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Tokenizerを読み込む"""
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str) -> AutoModelForSequenceClassification:
    """モデルを読み込み、推論モードで返す"""
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model


def preprocess(tokenizer: AutoTokenizer, text: str, max_length: int = 128):
    """文章をテンソルに変換（前処理）

    Args:
        tokenizer: HuggingFace Tokenizer
        text: 入力テキスト
        max_length: 最大トークン長

    Returns:
        input_ids と attention_mask を含む辞書（BatchEncoding）
    """
    return tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def forward(model: AutoModelForSequenceClassification, encoded) -> torch.Tensor:
    """モデルにテンソルを入力してlogitsを得る（推論）

    Args:
        model: 分類モデル
        encoded: preprocess() の出力

    Returns:
        logits テンソル shape: (1, num_labels)
    """
    with torch.no_grad():
        outputs = model(**encoded)
    return outputs.logits


def postprocess(
    model: AutoModelForSequenceClassification,
    logits: torch.Tensor,
) -> tuple[str, float, dict[str, float]]:
    """logitsからラベル名と確率に変換（後処理）

    Args:
        model: 分類モデル（id2label の取得に使用）
        logits: forward() の出力

    Returns:
        pred_label: 予測ラベル名
        pred_score: 予測ラベルの確率
        prob_dict: 全ラベルの確率辞書
    """
    id2label = model.config.id2label
    probs = F.softmax(logits, dim=-1)
    pred_id = probs.argmax().item()
    pred_label = id2label[pred_id]
    pred_score = probs[0][pred_id].item()
    prob_dict = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}
    return pred_label, pred_score, prob_dict


def coarse_label(label: str) -> str:
    """モデルの5段階ラベルを Positive / Negative / Neutral に変換"""
    if label in {"Positive", "Very Positive"}:
        return "Positive"
    elif label in {"Negative", "Very Negative"}:
        return "Negative"
    else:
        return "Neutral"


def predict(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    text: str,
) -> tuple[str, float, dict[str, float]]:
    """前処理・推論・後処理をまとめて実行

    Args:
        tokenizer: HuggingFace Tokenizer
        model: 分類モデル
        text: 入力テキスト

    Returns:
        pred_label: 予測ラベル名（5段階）
        pred_score: 予測ラベルの確率
        prob_dict: 全ラベルの確率辞書
    """
    encoded = preprocess(tokenizer, text)
    logits = forward(model, encoded)
    return postprocess(model, logits)


def evaluate(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    eval_data: list[tuple[str, str]],
) -> tuple[list[tuple], float]:
    """複数文の予測と Accuracy を計算

    Args:
        tokenizer: HuggingFace Tokenizer
        model: 分類モデル
        eval_data: (文章, 正解ラベル) のリスト

    Returns:
        results: (文章, 正解, raw予測, 変換後予測, スコア, 正解フラグ) のリスト
        accuracy: 正解率（0.0〜1.0）
    """
    results = []
    correct = 0

    for text, true_label in eval_data:
        raw_label, score, _ = predict(tokenizer, model, text)
        pred_label = coarse_label(raw_label)
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        results.append((text, true_label, raw_label, pred_label, score, is_correct))

    if len(eval_data) == 0:
        return [], 0.0
    accuracy = correct / len(eval_data)
    return results, accuracy
