# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

機械学習・AI 新卒研修の Day1 ハンズオン教材。Python 環境入門からモデルの推論・学習・高速化・デプロイまでを段階的に体験する。
研修受講者が GCP Workbench (GPU インスタンス) 上で実行することを想定している。

## 技術スタック

- **パッケージ管理**: uv (Python >=3.12)
- **Notebook**: marimo (リアクティブ Notebook、`.py` 形式)
- **ML フレームワーク**: PyTorch, Lightning, torchvision, Hugging Face transformers, LightGBM, ONNX Runtime
- **デプロイ**: FastAPI + Vertex AI カスタムコンテナ
- **Linter/Formatter**: ruff (CI で `uvx ruff check .` / `uvx ruff format --check .` が走る)

## 開発コマンド

各ハンズオンは独立した uv プロジェクト（個別の `pyproject.toml`）を持つ。**必ず対象ディレクトリに移動してから実行する。**

```bash
cd day1/<NN>-<name>

# 依存パッケージのインストール（初回のみ）
uv sync

# marimo notebook を起動（ブラウザで対話的に実行）
uv run marimo edit notebook.py

# Lint & Format
uvx ruff check .
uvx ruff format .
```

## アーキテクチャ

### ハンズオン共通構造

各ハンズオン (`00`〜`07`) は以下の構成を持つ:

- `pyproject.toml` — そのハンズオン固有の依存定義
- `notebook.py` — marimo notebook 本体（`@app.cell` デコレータでセルを定義）
- `src/` — notebook から import して使うユーティリティモジュール（一部ハンズオンのみ）

### marimo notebook の書き方

- `marimo.App()` でアプリを作成し、`@app.cell` デコレータでセルを定義する
- セル間の変数の受け渡しは関数の引数と return で行う（marimo のリアクティブ依存解決）
- UI コンポーネントは `mo.ui.*`、Markdown は `mo.md()` を使用
- 画像埋め込みは `mo.image()` + `pathlib.Path.resolve()` を使用する

### PyTorch CPU インデックスの指定

`06-accelerate-ml-model` のように CPU 環境で PyTorch を使うハンズオンでは、`pyproject.toml` に `[tool.uv.sources]` と `[[tool.uv.index]]` で PyTorch CPU wheel のインデックスを明示的に指定している。新しいハンズオンで CPU 版 PyTorch が必要な場合は同様に設定すること。

### ハンズオンの流れ

| # | テーマ | 主要ライブラリ |
|---|---|---|
| 00 | Python 環境（uv・marimo）の基本操作 | marimo |
| 01 | テーブルデータで ML 推論の 3 フェーズを体験 | PyTorch, LightGBM, scikit-learn |
| 02 | テキストを使った NLP モデルの推論 | transformers |
| 03 | 画像を使った Object Detection の推論 | transformers, torchvision, timm |
| 04 | 音声を使った推論 | transformers |
| 05 | PyTorch でモデル学習・過学習・転移学習 | PyTorch, Lightning, torchvision |
| 06 | ONNX エクスポート・INT8 量子化でモデル高速化 | ONNX, ONNX Runtime, transformers (SAM) |
| 07 | FastAPI + カスタムコンテナで Vertex AI にデプロイ | FastAPI, ONNX Runtime, google-cloud-aiplatform |
