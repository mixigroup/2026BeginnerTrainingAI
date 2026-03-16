# 日本語感情分析ハンズオン
#
# タスク：Sentiment Analysis（感情分析）
# - 文章全体にポジティブ / ネガティブのラベルを付ける
# - 例：「私はとっても幸せ」→ ポジティブ、「私はとっても不幸」→ ネガティブ
#
# 使用モデル: tabularisai/multilingual-sentiment-analysis
# - 多言語対応のBERT系モデルで、文章全体の特徴を捉えて分類する
#
# このノートブックでやること
# pipeline の内部を3フェーズに分解して理解する：
# 1. Preprocess（前処理）  — 文章 → テンソル
# 2. Forward（推論）       — テンソル → logits
# 3. Postprocess（後処理） — logits → ラベル

from transformers import pipeline
from src.sentiment import (
    load_tokenizer,
    load_model,
    preprocess,
    forward,
    postprocess,
    predict,
    evaluate,
    EVAL_DATA,
)

# ============================================================
# pipeline とは
# ============================================================
# HuggingFace の pipeline は、前処理・推論・後処理を1行で実行できる便利なAPI。
#
# pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
# pipe("あなたのことが大好きです。")
# → [{'label': 'Very Positive', 'score': 0.96...}]
#
# このノートブックでは、この pipeline の内部を分解して理解する。

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
pipe = pipeline("text-classification", model=MODEL_NAME)
pipeline_result = pipe("あなたのことが大好きです。")
print(f"結果：{pipeline_result}")

# ============================================================
# Phase 1: Preprocess（前処理）
# ============================================================
# 文章を固定長テンソルに変換する3ステップ
#
# 1. Tokenizerで分割
#    例：「このレストランは最高でした！」
#    → ["この", "レストラン", "は", "最高", "でした", "！"]
#
# 2. 数値IDに変換
#    トークン → 語彙辞書で数値化 → input_ids
#
# 3. 長さ調整
#    padding / max_length / attention_mask

tokenizer = load_tokenizer(MODEL_NAME)
sample_text = "このレストランは最高でした！"
tokens = tokenizer.tokenize(sample_text)
print(f"トークン数：{len(tokens)}")

encoded = preprocess(tokenizer, sample_text)
print(f"input_ids: {encoded['input_ids'].tolist()}")
print(f"attention_mask: {encoded['attention_mask'].tolist()}")
print(f"テンソル長: {encoded['input_ids'].shape[1]}")
print(f"実トークン数（attention_mask の合計）: {encoded['attention_mask'].sum().item()}")

# ============================================================
# Phase 2: Forward（モデル計算）
# ============================================================
# Sentiment Analysis の推論フロー
#
# 1. 入力: input_ids + attention_mask
# 2. モデル処理: BERT系モデルで文章全体の特徴を抽出
#    [CLS]トークンの特徴量をもとに分類
# 3. 出力: logits
#    shape: (batch_size, num_labels)
#    例：(1, 5) → 5ラベル（Very Negative / Negative / Neutral / Positive / Very Positive）
#    各ラベルに対する生スコア（確率ではない）

model = load_model(MODEL_NAME)
logits = forward(model, encoded)

print(f"logits shape: {list(logits.shape)}")
print(f"logits の値（生スコア）: {logits.tolist()}")
print(f"ラベルの種類: {list(model.config.id2label.values())}")
print("※ logits はまだ生スコア（確率ではない）。次のPhaseで確率に変換する。")

# ============================================================
# Phase 3: Postprocess（後処理）
# ============================================================
# logits からラベルへの変換
#
# 1. softmax で確率に変換
#    logitsを確率分布に変換（合計が1になる）
#    例：[3.2, -1.5, 0.1, ...] → [0.88, 0.01, 0.05, ...]
#
# 2. argmax でラベルID取得
#    最大確率のインデックスを選択
#
# 3. ラベル名にマッピング
#    model.config.id2label でラベルIDを文字列に変換

pred_label, pred_score, prob_dict = postprocess(model, logits)

print("\n確率分布（softmax後）:")
print(f"{'ラベル':<20} {'確率':>8}")
print("-" * 30)
for label, score in prob_dict.items():
    print(f"{label:<20} {score:>8.4f}")
print(f"\n予測結果: {pred_label}（{pred_score:.1%}）")
print("※ pipeline の出力と同じ結果になっているはず！")

# ============================================================
# 評価指標
# ============================================================
# Sentiment Analysis では、文章全体の分類が正しいかを評価する。
#
# 最も基本的な指標は Accuracy（正解率）：
#   Accuracy = 正解数 / 全文章数
#
# 今回は20件のサンプルデータで精度を測定する。
# モデルの5段階ラベルは以下のように2値に変換して評価する：
#
# モデルのラベル                    | 変換後
# Very Positive / Positive        | Positive
# Very Negative / Negative        | Negative
# Neutral                         | Neutral

results, accuracy = evaluate(tokenizer, model, EVAL_DATA)
correct_count = sum(1 for r in results if r[5])

print(f"\n評価結果（{len(EVAL_DATA)}件）:")
print(f"{'文章':<25} {'正解':<10} {'モデル予測(raw)':<20} {'変換後':<10} {'正解?'}")
print("-" * 80)
for r in results:
    mark = "✅" if r[5] else "❌"
    print(f"{r[0]:<25} {r[1]:<10} {r[2]}（{r[4]:.0%}）{'':<5} {r[3]:<10} {mark}")

print(f"\nAccuracy: {accuracy:.1%}（{correct_count}/{len(EVAL_DATA)}件正解）")

# ============================================================
# インタラクティブデモ
# ============================================================
# 文章を入力すると感情分析を実行する。
# モデルが判定する5つのラベル：
# Very Negative / Negative / Neutral / Positive / Very Positive

print("\n--- インタラクティブデモ ---")
print("感情分析するテキストを入力してください（空白でスキップ）:")
input_text = input("> ").strip()

if not input_text:
    input_text = "あなたのことが大好きです。"
    print(f"デフォルトテキストを使用: {input_text}")

demo_label, demo_score, demo_prob_dict = predict(tokenizer, model, input_text)

print(f"\n入力: {input_text}")
print(f"予測ラベル: {demo_label}（{demo_score:.1%}）")
print(f"\n{'ラベル':<20} {'確率':>8}")
print("-" * 30)
for lbl, score in demo_prob_dict.items():
    marker = " ◀ 予測" if lbl == demo_label else ""
    print(f"{lbl:<20} {score:>8.4f}{marker}")
