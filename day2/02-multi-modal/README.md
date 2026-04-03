# 02. SigLIP2 マルチモーダル — テキストと画像のベクトル空間理解

SigLIP2（Sigmoid Loss for Language-Image Pre-training 2）を使って、テキストと画像を同じベクトル空間にエンコードし、両者の距離を計算することで、マルチモーダルモデルの仕組みを理解する。

## 学習内容

1. **Contrastive Learning** — SigLIP2 の学習方法（Sigmoid vs Softmax）の理解
2. **Embedding 取得** — 画像・日本語テキストの特徴ベクトル取得
3. **コサイン類似度** — 画像↔テキスト、画像↔画像、テキスト↔テキスト の距離計測
4. **TensorBoardX** — ベクトル空間の 3D インタラクティブ可視化

## ディレクトリ構成

```
02-multi-modal/
├── notebook.py          # Marimo ノートブック（メインのハンズオン教材）
├── pyproject.toml       # プロジェクト設定・依存パッケージ
├── README.md            # この文書
├── plan.md              # 実装計画
├── .python-version      # Python 3.12
├── src/
│   ├── __init__.py
│   └── siglip_utils.py  # ヘルパー関数（エンコード・類似度計算・可視化）
├── images/
│   └── .gitkeep         # サンプル画像用ディレクトリ
└── runs/                # TensorBoardX の出力先（gitignore 推奨）
```

## セットアップ

```bash
cd day2/02-multi-modal
uv sync
```

## 実行方法

### Marimo ノートブック（推奨）

```bash
uv run marimo edit notebook.py
```

ブラウザが開き、セルを順番に実行しながら学習できる。

### TensorBoard の起動

ノートブック内で埋め込みのエクスポートを実行した後：

```bash
uv run tensorboard --logdir=runs
```

ブラウザで `http://localhost:6006` を開き、**PROJECTOR** タブで 3D 可視化を確認する。

---

## SigLIP2 とは

SigLIP2 は Google が開発したマルチモーダルモデル。画像とテキストを同じベクトル空間に埋め込み、両者の関連度を計算できる。

### CLIP との違い

| | CLIP | SigLIP2 |
|---|---|---|
| **損失関数** | Softmax（NxN 行列全体で正規化） | Sigmoid（各ペア独立に判定） |
| **学習方式** | バッチ内の全ペアを比較 | 各画像-テキストペアを独立に二値分類 |
| **多言語** | 英語中心 | 35言語以上対応（日本語含む） |
| **効率** | 大バッチサイズが必要 | 小バッチでも安定 |

### Sigmoid Loss の直感

```
CLIP:   「この画像は N 個のテキストのうち、どれに最も近い？」（多クラス分類）
SigLIP: 「この画像とこのテキストは一致する？ Yes/No」（二値分類 × 全ペア）
```

### 使用モデル

- **`google/siglip2-base-patch16-224`** — ViT-B/16、86M パラメータ、224px 入力

## コサイン類似度

2つのベクトル **a** と **b** のコサイン類似度：

```
cosine_sim(a, b) = (a · b) / (|a| × |b|)
```

L2 正規化済みベクトル（|a| = |b| = 1）の場合、単純なドット積で計算できる：

```
cosine_sim(a, b) = a · b
```

## xm3600 データセット

**Crossmodal-3600（xm3600）** は Google が公開した多言語マルチモーダルデータセット：

- 3,600 枚の地理的に多様な画像
- 36 言語の人手によるキャプション（日本語含む）
- 翻訳ではなく、各言語のネイティブスピーカーが独立に記述

このハンズオンでは日本語キャプション付きの 50 件サブセットを使用する。

## TensorBoardX による可視化

TensorBoardX の Embedding Projector を使うと、高次元の埋め込みベクトルを PCA や t-SNE で 3D/2D に射影し、インタラクティブに探索できる。

### 確認ポイント

- 画像とテキストがどのようにクラスタを形成するか
- 対応する画像-テキストペアが近くにあるか
- PCA / t-SNE を切り替えてクラスタの変化を観察

---

## CLI での実行

ノートブックを使わずに Python CLI で各ステップを実行する方法。

### 1. モデルのロード

```bash
uv run python -c "
from src.siglip_utils import load_siglip_model
model, processor = load_siglip_model('google/siglip2-base-patch16-224')
n_params = sum(p.numel() for p in model.parameters())
print(f'パラメータ数: {n_params / 1e6:.1f}M')
"
```

### 2. テキストの埋め込み取得

```bash
uv run python -c "
from src.siglip_utils import load_siglip_model, encode_texts

model, processor = load_siglip_model('google/siglip2-base-patch16-224')
texts = ['猫が座っている', '犬が走っている', '都市の風景']
emb = encode_texts(model, processor, texts)
print(f'形状: {emb.shape}')
print(f'先頭5次元: {emb[0, :5]}')
"
```

### 3. xm3600 データセットのロードとエンコード

```bash
uv run python -c "
from datasets import load_dataset
from src.siglip_utils import load_siglip_model, encode_images, encode_texts, cosine_similarity_matrix

# データセットロード
ds = load_dataset('floschne/xm3600', split='ja')
ds = ds.select(range(10))

# モデルロード
model, processor = load_siglip_model('google/siglip2-base-patch16-224')

# エンコード
images = [s['image'] for s in ds]
texts = [s['captions'][0] for s in ds]
img_emb = encode_images(model, processor, images)
txt_emb = encode_texts(model, processor, texts)

# 類似度計算
sim = cosine_similarity_matrix(img_emb, txt_emb)
print('画像↔テキスト類似度行列:')
print(sim.round(3))
"
```

### 4. TensorBoardX へのエクスポート

```bash
uv run python -c "
from datasets import load_dataset
from tensorboardX import SummaryWriter
from src.siglip_utils import (
    load_siglip_model, encode_images, encode_texts,
    export_embeddings_to_tensorboard,
)

ds = load_dataset('floschne/xm3600', split='ja')
ds = ds.select(range(50))

model, processor = load_siglip_model('google/siglip2-base-patch16-224')
images = [s['image'] for s in ds]
texts = [s['captions'][0] for s in ds]
img_emb = encode_images(model, processor, images)
txt_emb = encode_texts(model, processor, texts)

writer = SummaryWriter('runs/siglip2_embeddings')
export_embeddings_to_tensorboard(
    writer, img_emb, txt_emb,
    image_labels=[f'[IMG] {i}: {texts[i][:20]}' for i in range(len(images))],
    text_labels=[f'[TXT] {i}: {texts[i][:20]}' for i in range(len(texts))],
    images=images,
)
writer.close()
print('エクスポート完了。tensorboard --logdir=runs で確認')
"
```

```bash
uv run tensorboard --logdir=runs
```

---

## ハンズオン課題

- [ ] クエリテキストを変えて検索結果の変化を観察する
- [ ] `N_SAMPLES` を増やして、より多くのデータで可視化する
- [ ] 画像↔画像で最も類似度が高いペアを見つけ、共通点を考察する
- [ ] 英語のテキストで同じ検索を試し、日本語との結果を比較する
- [ ] TensorBoard で t-SNE の perplexity を変えてクラスタの変化を観察する

## 使用モデル・データセット

| 名前 | リンク |
|------|--------|
| SigLIP2 Base | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) |
| xm3600 | [floschne/xm3600](https://huggingface.co/datasets/floschne/xm3600) |
| TensorBoardX | [tensorboardX](https://tensorboardx.readthedocs.io/) |
