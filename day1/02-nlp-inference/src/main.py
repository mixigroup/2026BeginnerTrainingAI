"""感情分析ハンズオン — pipeline の内部を3フェーズで再現するスクリプト"""

from transformers import pipeline

from src.sentiment import (
    EVAL_DATA,
    evaluate,
    forward,
    load_model,
    load_tokenizer,
    postprocess,
    preprocess,
)

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
SAMPLE_TEXT = "このレストランは最高でした！"


def section(title: str) -> None:
    """セクション区切りを表示"""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)


# ----------------------------------------------------------------
# pipeline で1行実行（比較用）
# ----------------------------------------------------------------
section("pipeline で1行実行（比較用）")

pipe = pipeline("text-classification", model=MODEL_NAME)
pipeline_result = pipe("あなたのことが大好きです。")
print("入力: あなたのことが大好きです。")
print(f"結果: {pipeline_result}")


# ----------------------------------------------------------------
# Phase 1: Preprocess（前処理）
# ----------------------------------------------------------------
section("Phase 1: Preprocess（前処理）")

# Tokenizer を読み込む
tokenizer = load_tokenizer(MODEL_NAME)

# トークン化：文章をサブワード列に分割
tokens = tokenizer.tokenize(SAMPLE_TEXT)
print(f"入力テキスト : {SAMPLE_TEXT}")
print(f"トークン列   : {tokens}")
print(f"トークン数   : {len(tokens)}")

# テンソル化：input_ids と attention_mask に変換
encoded = preprocess(tokenizer, SAMPLE_TEXT)
print(f"\ninput_ids      : {encoded['input_ids'].tolist()}")
print(f"attention_mask : {encoded['attention_mask'].tolist()}")
print(f"テンソル長     : {encoded['input_ids'].shape[1]}")


# ----------------------------------------------------------------
# Phase 2: Forward（推論）
# ----------------------------------------------------------------
section("Phase 2: Forward（推論）")

# モデルを読み込み、logits を取得
model = load_model(MODEL_NAME)
logits = forward(model, encoded)
print(f"logits shape : {list(logits.shape)}")
print(f"logits 値    : {logits.tolist()}")
print(f"ラベル種類   : {list(model.config.id2label.values())}")


# ----------------------------------------------------------------
# Phase 3: Postprocess（後処理）
# ----------------------------------------------------------------
section("Phase 3: Postprocess（後処理）")

# softmax → argmax → ラベル名にマッピング
pred_label, pred_score, prob_dict = postprocess(model, logits)
print("確率分布（softmax後）:")
for label, prob in prob_dict.items():
    marker = " ◀ 予測" if label == pred_label else ""
    print(f"  {label:<16}: {prob:.4f}{marker}")
print(f"\n予測結果: {pred_label}（{pred_score:.1%}）")
print("\n※ pipeline の出力と同じ結果になっているはず！")


# ----------------------------------------------------------------
# 評価指標：20件のサンプルで Accuracy を計算
# ----------------------------------------------------------------
section("評価（20件サンプル）")

results, accuracy = evaluate(tokenizer, model, EVAL_DATA)

print(f"{'文章':<20} {'正解':<10} {'モデル予測（raw）':<22} {'変換後':<10} 正解？")
print("-" * 75)
for text, true_label, raw_label, pred_label, score, is_correct in results:
    mark = "✅" if is_correct else "❌"
    raw_with_score = f"{raw_label}（{score:.0%}）"
    print(f"{text:<20} {true_label:<10} {raw_with_score:<22} {pred_label:<10} {mark}")

correct_count = sum(1 for r in results if r[5])
print(f"\nAccuracy: {accuracy:.1%}（{correct_count}/{len(EVAL_DATA)}件正解）")
