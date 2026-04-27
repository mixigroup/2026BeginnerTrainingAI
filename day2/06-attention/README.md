# 06. Attention メカニズムの可視化 — Word2Vec と BERT で理解する Attention

Attention メカニズムの数学的原理を Word2Vec で体験し、BERT の Multi-Head Self-Attention を可視化することで、Transformer の中核技術を理解する。

## 学習内容

1. **Attention の数学的原理** — Query, Key, Value とソフトマックスによる重み計算
2. **Word2Vec による Attention 体験** — 単語ベクトルで Attention スコアを手計算
3. **Temperature パラメータ** — Attention の鋭さを制御する温度パラメータ
4. **BERT の Multi-Head Attention 可視化** — 12層 × 12ヘッド = 144 の Attention パターン
5. **Attention パターンの解釈** — ヘッドごとの役割と層の深さによる変化

## ディレクトリ構成

```
06-attention/
├── notebook.py          # Marimo ノートブック（メインのハンズオン教材）
├── pyproject.toml       # プロジェクト設定・依存パッケージ
├── README.md            # この文書
├── .python-version      # Python 3.12
└── src/
    ├── __init__.py
    └── attention_utils.py  # ヘルパー関数（Attention 計算・可視化）
```

## セットアップ

```bash
cd day2/06-attention
uv sync
```

## 実行方法

### Marimo ノートブック（推奨）

```bash
uv run marimo edit notebook.py
```

ブラウザが開き、セルを順番に実行しながら学習できる。

> ⚠️ 初回実行時に Word2Vec モデル（約 1.6GB）のダウンロードが必要。
> ダウンロード後は `~/gensim-data/` にキャッシュされる。

---

## Attention メカニズムとは

Attention は「入力のどの部分に注目すべきか」を数値化する仕組み。3つのベクトルで構成される：

- **Query (Q)**: 「何を探しているか」
- **Key (K)**: 「各要素の見出し」
- **Value (V)**: 「各要素の中身」

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Simple Attention vs Self-Attention vs Multi-Head

| | Simple Attention | Self-Attention | Multi-Head Attention |
|---|---|---|---|
| **Q, K, V の出所** | 外部クエリ | 同じ入力 | 同じ入力（射影済み） |
| **ヘッド数** | 1 | 1 | h（例: 12） |
| **用途** | 検索 | 文脈理解 | Transformer |

---

## CLI での実行

ノートブックを使わずに Python CLI で各ステップを実行する方法。

### 1. Word2Vec + Attention 計算

```bash
uv run python -c "
import numpy as np
import gensim.downloader as api
from src.attention_utils import compute_attention_weights

w2v = api.load('word2vec-google-news-300')
query_vec = w2v['fruit']
words = ['apple', 'king', 'banana', 'queen', 'orange', 'computer']
key_vecs = np.array([w2v[w] for w in words])

weights = compute_attention_weights(query_vec, key_vecs)
for w, wt in zip(words, weights):
    print(f'  {w:>10}: {wt:.4f}')
"
```

### 2. BERT Attention 取得

```bash
uv run python -c "
from src.attention_utils import load_bert_japanese, get_bert_attentions

model, tokenizer = load_bert_japanese()
text = '今日は渋谷の会場で、AIのアテンションについて詳しく学んでいます。'
attentions, tokens = get_bert_attentions(model, tokenizer, text)

print(f'トークン: {tokens}')
print(f'Attention shape: {list(attentions.shape)}')
print(f'  → {attentions.shape[0]} layers × {attentions.shape[1]} heads × {attentions.shape[2]} tokens × {attentions.shape[3]} tokens')
"
```

### 3. Attention ヒートマップ生成

```bash
uv run python -c "
from src.attention_utils import load_bert_japanese, get_bert_attentions, plot_attention_heatmap
import matplotlib.pyplot as plt

model, tokenizer = load_bert_japanese()
attentions, tokens = get_bert_attentions(model, tokenizer, '猫が魚を食べた')

fig = plot_attention_heatmap(attentions, tokens, layer=0, head=0)
fig.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
print('attention_heatmap.png に保存しました')
"
```

---

## ハンズオン課題

- [ ] `query` を "queen", "doctor", "computer" に変えて Attention の変化を観察する
- [ ] `words` のリストを変えて、文脈による Attention の違いを比較する
- [ ] Temperature を 0.1, 0.5, 1.0, 5.0 で比較し、分布の変化をグラフで確認する
- [ ] BERT に異なる日本語文を入力し、Attention パターンの変化を観察する
- [ ] Layer 0 と Layer 11 の同じヘッドを比較し、浅い層と深い層の違いを考察する
- [ ] [CLS] トークンの Attention を複数文で比較し、重要語の抽出に使えるか検討する

## 使用モデル

| 名前 | リンク |
|------|--------|
| Word2Vec (Google News 300d) | [gensim-data/word2vec-google-news-300](https://github.com/RaRe-Technologies/gensim-data) |
| BERT base Japanese v3 | [cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3) |
