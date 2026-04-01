# 01. Stable Diffusion 画像生成 — 実装計画

## 目的

拡散モデル（Diffusion Model）の基本的な仕組みを理解し、Stable Diffusion を使って実際に画像を生成する体験を通じて、生成 AI の動作を学ぶ。

## 学習内容

- 拡散モデルの基本概念（Forward Process / Reverse Process）
- プロンプトエンジニアリングの基礎
- パラメータ（steps, guidance_scale, seed 等）が生成結果に与える影響

## 依存パッケージ

```toml
dependencies = [
    "marimo>=0.20.2",
    "pyzmq>=27.1.0",
    "diffusers>=0.32.0",
    "torch>=2.0",
    "transformers>=4.40",
    "accelerate>=1.0",
    "Pillow>=10.0",
    "matplotlib>=3.8",
]
```

## notebook.py セル構成

| # | タイプ | 内容 |
|---|--------|------|
| 1 | code | `import marimo as mo` |
| 2 | md | タイトル: Stable Diffusion 画像生成ハンズオン / 拡散モデルの概要 |
| 3 | md | 拡散過程（Forward Process）の説明 |
| 4 | code | ライブラリ import（torch, diffusers, PIL, matplotlib） |
| 5 | code | Forward Process の可視化 — 画像にノイズを段階的に加えて表示 |
| 6 | md | 逆拡散過程（Reverse Process）の説明 |
| 7 | code | Reverse Process — ノイズ除去のステップを可視化 |
| 8 | md | Stable Diffusion パイプラインの説明 |
| 9 | code | `StableDiffusionPipeline.from_pretrained()` でモデルロード + 基本的な画像生成 |
| 10 | md | プロンプトエンジニアリングの説明 |
| 11 | code | プロンプト・ネガティブプロンプトを変えて画像生成 |
| 12 | md | パラメータの影響（steps, guidance_scale, seed）の説明 |
| 13 | code | パラメータ比較実験 — 異なるパラメータで生成し結果を並べて表示 |
| 14 | md | まとめ |

## src/ ファイル構成

- `src/__init__.py`
- `src/generate.py` — ノイズスケジュール可視化、画像グリッド表示などのヘルパー関数

## 追加ディレクトリ

- `images/.gitkeep` — 説明図（Forward/Reverse Process の図解など）

## ハンズオン課題

- プロンプトを変更して異なる画像を生成してみる
- `guidance_scale` や step 数を変えて品質の違いを観察する
- `negative_prompt` の効果を確認する
- seed を固定して再現性を確認する

## 使用モデル

- `stabilityai/stable-diffusion-2-1`（Hugging Face Hub）
