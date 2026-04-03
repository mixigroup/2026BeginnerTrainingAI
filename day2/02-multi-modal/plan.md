# 02. SigLIP2 マルチモーダル — 実装計画

## 目的

SigLIP2 を使ってテキストと画像を同じベクトル空間にエンコードし、両者の距離を計算することで、マルチモーダルモデルの仕組みを理解する。

## 学習内容

- SigLIP2 の仕組み（Sigmoid Contrastive Learning、CLIP との違い）
- テキストと画像の埋め込み（Embedding）
- コサイン類似度による3種の距離計測（画像↔テキスト、画像↔画像、テキスト↔テキスト）
- xm3600 データセットによる多言語評価
- TensorBoardX によるベクトル空間の 3D 可視化

## 依存パッケージ

```toml
dependencies = [
    "marimo>=0.20.2",
    "pyzmq>=27.1.0",
    "transformers>=4.48",
    "torch>=2.0",
    "Pillow>=10.0",
    "matplotlib>=3.8",
    "datasets>=3.0",
    "tensorboardX>=2.6",
    "tensorboard>=2.15",
    "numpy>=1.26",
    "requests>=2.31",
]
```

## notebook.py セル構成

### Part 1: 導入 + モデルロード

| # | タイプ | 内容 |
|---|--------|------|
| 1 | code | `import marimo as mo` |
| 2 | md | タイトル + 学習目標 |
| 3 | md | SigLIP2 の仕組み（Sigmoid vs Softmax、CLIP との比較表） |
| 4 | code | ライブラリ import + src ヘルパー |
| 5 | code | 設定（MODEL_NAME, DEVICE） |
| 6 | md | モデルロードの説明 |
| 7 | code | モデルロード + パラメータ数表示 |

### Part 2: 簡単なデモ

| # | タイプ | 内容 |
|---|--------|------|
| 8 | md | デモの導入 |
| 9 | code | デモ画像ダウンロード + 日本語テキスト定義 |
| 10 | code | 埋め込みベクトル取得 + 形状表示 |
| 11 | md | コサイン類似度の数式説明 |
| 12 | code | 画像↔テキスト類似度ヒートマップ |

### Part 3: xm3600 データセット

| # | タイプ | 内容 |
|---|--------|------|
| 13 | md | xm3600 の説明（36言語、3600画像） |
| 14 | code | N_SAMPLES 設定 |
| 15 | code | データセットロード |
| 16 | code | サンプル表示（画像+キャプション） |
| 17 | code | 全画像・テキストエンコード |

### Part 4: 3種の距離分析

| # | タイプ | 内容 |
|---|--------|------|
| 18 | md | 3つの距離の導入 |
| 19 | code | 画像↔テキスト類似度（10x10） |
| 20 | md | 画像↔画像の説明 |
| 21 | code | 画像↔画像類似度（15x15） |
| 22 | md | テキスト↔テキストの説明 |
| 23 | code | テキスト↔テキスト類似度（15x15） |
| 24 | md | テキスト→画像検索の説明 |
| 25 | code | Top-K 検索（テキスト→画像 + 画像→テキスト） |

### Part 5: TensorBoardX 可視化

| # | タイプ | 内容 |
|---|--------|------|
| 26 | md | TensorBoardX Embedding Projector の説明 |
| 27 | code | 埋め込みエクスポート（`[IMG]`/`[TXT]`ラベル付き） |
| 28 | md | TensorBoard 起動手順 |
| 29 | md | まとめ + ハンズオン課題 |

## src/ ファイル構成

- `src/__init__.py`
- `src/siglip_utils.py` — モデルロード、エンコード、類似度計算、ヒートマップ描画、TensorBoardX エクスポート

## 追加ディレクトリ

- `images/.gitkeep` — サンプル画像用
- `runs/` — TensorBoardX 出力（.gitignore 推奨）

## ハンズオン課題

- クエリテキストを変えて検索結果の変化を観察する
- `N_SAMPLES` を増やして可視化する
- 画像↔画像で最も類似度が高いペアを見つける
- 英語テキストで検索し、日本語との結果を比較する
- TensorBoard で t-SNE の perplexity を変えて観察する

## 使用モデル・データセット

- `google/siglip2-base-patch16-224`（Hugging Face Hub）
- `floschne/xm3600`（Crossmodal-3600 データセット）
