# 02. NLP 推論ハンズオン

日本語テキストを使った感情分析（Sentiment Analysis）で、NLP モデルの推論フローを体験します。

- **モデル**: [`tabularisai/multilingual-sentiment-analysis`](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)
  - 多言語対応の BERT 系モデル（日本語テキストをそのまま入力できる）

---

## ハンズオンの目的

HuggingFace の `pipeline` は、前処理・推論・後処理を1行で実行できる便利な API ですが、中身はブラックボックスです。

このハンズオンでは `pipeline` の内部を **3フェーズ** に分解して理解します。

```
1. Preprocess  → 文章をトークン化して input_ids テンソルに変換
2. Forward     → モデルに入力して logits を得る
3. Postprocess → softmax → argmax → ラベル名にマッピング
```

最終的に `pipeline` の出力と、手動実装の結果が一致することを確認します。

---

## ディレクトリ構造

```
02-nlp-inference/
├── src/
│   └── sentiment.py  # 前処理・推論・後処理の関数 + 評価サンプルデータ
├── notebook.py        # marimo ノートブック（インタラクティブ版）
├── pyproject.toml     # 依存パッケージ定義
└── README.md
```

### src/sentiment.py の主な関数

| 関数 | 役割 |
|---|---|
| `load_tokenizer(model_name)` | Tokenizer の読み込み |
| `load_model(model_name)` | モデルの読み込み（推論モード） |
| `preprocess(tokenizer, text)` | 文章 → input_ids テンソルに変換 |
| `forward(model, encoded)` | テンソル → logits |
| `postprocess(model, logits)` | logits → ラベル名・確率 |
| `predict(tokenizer, model, text)` | 3フェーズをまとめて実行 |
| `evaluate(tokenizer, model, eval_data)` | 複数文の予測と Accuracy 計算 |
| `EVAL_DATA` | 評価用サンプルデータ 20件 |

---

## 環境セットアップ

```bash
# 依存パッケージをインストール（初回のみ）
uv sync
```

---

## 実行方法

```bash
uv run marimo edit notebook.py
```

ブラウザが自動的に開きます。上から順にセルを実行してください。

---

## ノートブックの構成

| セクション | 内容 |
|---|---|
| **pipeline で1行実行** | `pipeline` API を使って感情分析を1行で実行（比較基準） |
| **Phase 1: Preprocess** | Tokenizer でトークン化 → 数値ID変換 → padding / attention_mask |
| **Phase 2: Forward** | モデル読み込み → `model(**encoded)` → logits の確認 |
| **Phase 3: Postprocess** | softmax → argmax → `id2label` でラベル名にマッピング |
| **評価指標** | 複数文の予測結果と Accuracy の計算 |

---

## 推論フロー詳細

### Phase 1: Preprocess（前処理）

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 文章 → トークン列
tokens = tokenizer.tokenize("このレストランは最高でした！")
# → ['この', 'レストラン', 'は', '最高', 'でした', '！']

# トークン列 → テンソル
encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
# → {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]])}
```

### Phase 2: Forward（推論）

```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

with torch.no_grad():
    outputs = model(**encoded)

logits = outputs.logits  # shape: (1, 5) — 5ラベル分のスコア
```

### Phase 3: Postprocess（後処理）

```python
probs = F.softmax(logits, dim=-1)   # 確率に変換（合計=1）
pred_id = probs.argmax().item()      # 最大確率のインデックス
pred_label = model.config.id2label[pred_id]  # ラベル名に変換
```

---

## ラベルの種類

このモデルは5段階で感情を分類します。

| ラベル | 意味 |
|---|---|
| Very Positive | とてもポジティブ |
| Positive | ポジティブ |
| Neutral | 中立 |
| Negative | ネガティブ |
| Very Negative | とてもネガティブ |

---

## 評価指標

感情分析（分類タスク）では **Accuracy（正解率）** が基本指標です。

```
Accuracy = 正解数 / 全文章数
```

### 評価例

| 文章 | 正解 | 予測 | 正解？ |
|------|------|------|--------|
| 料理が美味しかった | Positive | Positive | ✅ |
| サービスが最悪 | Negative | Negative | ✅ |
| まあまあだった | Negative | Positive | ❌ |
| また行きたい | Positive | Positive | ✅ |

**Accuracy: 75%（3/4文章正解）**

