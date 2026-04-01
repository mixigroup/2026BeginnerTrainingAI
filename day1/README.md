# Day 1 - AI 新卒研修ハンズオン

機械学習の基礎から、推論・学習・デプロイまでを体験する 1 日目のハンズオン資料です。

---

## ハンズオン一覧

| # | ディレクトリ | テーマ |
|---|---|---|
| 00 | `00-intro-python-environment` | Python 環境（uv・marimo）の基本操作 |
| 01 | `01-simple-model-inference` | テーブルデータで ML 推論の 3 フェーズを体験 |
| 02 | `02-nlp-inference` | テキストを使った NLP モデルの推論 |
| 03 | `03-vision-inference` | 画像を使った Object Detection の推論 |
| 04 | `04-audio-inference` | 音声を使った推論 |
| 05 | `05-model-trainig` | PyTorch でモデル学習・過学習・転移学習を実践 |
| 06 | `06-accelerate-ml-model` | ONNX エクスポート・INT8 量子化でモデルを高速化 |
| 07 | `07-model-deploy` | FastAPI + カスタムコンテナで Vertex AI にデプロイ |

---

## ハンズオンの進め方

全てのハンズオンは **marimo notebook** で実行します。
ブラウザ上でコードを逐次実行しながら進められます。
変数を変更すると関連セルが自動再実行されるリアクティブな実行環境です。

```bash
cd day1/<ハンズオン番号>-<名前>

# 依存パッケージのインストール（初回のみ）
uv sync

# marimo notebook を起動
uv run marimo edit notebook.py
```

ブラウザが自動的に開き、上から順にセルを実行できます。

---

## 前提ツール

| ツール | 説明 | インストール |
|---|---|---|
| **uv** | Rust 製の高速 Python パッケージマネージャー | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **marimo** | `.py` ファイルで動くリアクティブ Notebook | `uv` 経由で各プロジェクトに含まれます |
