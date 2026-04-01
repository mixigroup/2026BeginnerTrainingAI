# 02. CLIP マルチモーダル — 実装計画

## 目的

CLIP を使ってテキストと画像を同じ特徴空間にエンコードし、両者の距離を計算することで、マルチモーダルモデルの仕組みを理解する。

## 学習内容

- CLIP の仕組み（Contrastive Learning）
- テキストと画像の埋め込み（Embedding）
- コサイン類似度による類似性の測定
- Zero-shot 画像分類

## 依存パッケージ

```toml
dependencies = [
    "marimo>=0.20.2",
    "pyzmq>=27.1.0",
    "transformers>=4.40",
    "torch>=2.0",
    "Pillow>=10.0",
    "matplotlib>=3.8",
    "requests>=2.31",
]
```

## notebook.py セル構成

| # | タイプ | 内容 |
|---|--------|------|
| 1 | code | `import marimo as mo` |
| 2 | md | タイトル: CLIP — テキストと画像の関係理解 |
| 3 | md | Contrastive Learning の説明（画像とテキストを同じ空間に埋め込む） |
| 4 | code | ライブラリ import（transformers, torch, PIL, matplotlib） |
| 5 | md | CLIP モデルのロード |
| 6 | code | `CLIPModel.from_pretrained()` / `CLIPProcessor.from_pretrained()` |
| 7 | md | 画像とテキストの Embedding 取得 |
| 8 | code | 画像・テキストをエンコードして特徴ベクトルを取得 |
| 9 | md | コサイン類似度の説明 |
| 10 | code | `F.normalize()` + 行列積で類似度計算、結果表示 |
| 11 | md | 類似度マトリクスの可視化 |
| 12 | code | 複数画像×複数テキストでヒートマップ作成（matplotlib） |
| 13 | md | Zero-shot 画像分類の説明 |
| 14 | code | CLIP でラベル候補を渡して分類を実装 |
| 15 | md | まとめ |

## src/ ファイル構成

- `src/__init__.py`
- `src/clip_utils.py` — サンプル画像ダウンロード、コサイン類似度計算、ヒートマップ描画のヘルパー

## 追加ディレクトリ

- `images/.gitkeep` — サンプル画像

## ハンズオン課題

- 異なる画像で試してみる
- テキストの記述を変えて類似度の変化を観察する
- 複数の画像と複数のテキストで類似度マトリクスを作成する
- Zero-shot 分類で独自のラベルセットを試す

## 使用モデル

- `openai/clip-vit-base-patch32`（Hugging Face Hub）
