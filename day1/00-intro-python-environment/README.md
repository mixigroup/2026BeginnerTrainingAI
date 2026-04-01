# Python 環境入門

このプロジェクトでは、現代的な Python 開発環境のセットアップと基本的な使い方を学びます。

---

## 目次

1. [uv とは？](#uv-とは)
2. [プロジェクト構造](#プロジェクト構造)
3. [marimo の使い方](#marimo-の使い方)

---

## uv とは？

**uv** は Rust 製の超高速 Python パッケージマネージャーです。
従来の `pip` + `venv` の組み合わせを一つのツールで置き換えられます。

### インストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### 主要ファイルの説明

| ファイル | 役割 |
|---|---|
| `pyproject.toml` | プロジェクトの設定・依存パッケージの定義 |
| `uv.lock` | 依存パッケージの正確なバージョンを固定するファイル（自動生成） |
| `.venv/` | 仮想環境（自動生成、git には含めない） |

#### pyproject.toml の例

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "marimo>=0.10.0",
    "numpy>=2.0.0",
]
```

#### uv.lock について

`uv.lock` は `uv add` 実行時に自動生成されます。
チームで同じ環境を再現するために git に含めてください。

---

### パッケージの操作

#### パッケージを追加する

```bash
uv add numpy
uv add "numpy>=2.0.0"   # バージョン指定
```

#### パッケージを削除する

```bash
uv remove numpy
```

#### インストール済みパッケージ一覧を確認する

```bash
uv pip list
```

---

### 仮想環境の有効化（activate）

`uv run` を使う場合は activate 不要ですが、手動で有効化したい場合は以下を実行します。

```bash
# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

有効化を解除する場合：

```bash
deactivate
```

> **ポイント**: `uv run marimo edit notebook.py` のように `uv run` を使えば activate なしでも実行できます。

---

### uvx とは？

`uvx` はパッケージをインストールせずに**一時的に実行**できるコマンドです。
試しに使いたいツールや、一回だけ使うスクリプトに便利です。

```bash
# marimo を一時的に起動（インストール不要）
uvx marimo tutorial intro

# ruff（リンター）をインストールせずに実行
uvx ruff check .
```

---

## プロジェクト構造

```
00-intro-python-environment/
├── README.md          # このファイル（プロジェクトの説明）
├── pyproject.toml     # プロジェクト設定・依存パッケージ
├── uv.lock            # 依存パッケージのバージョン固定（自動生成）
├── .venv/             # 仮想環境（自動生成）
└── notebook.py        # marimo ノートブック（Jupyter Notebook の代替）
```

### 基本的な使い方

marimo notebook を起動して、ブラウザ上でインタラクティブにコードを実行します。

```bash
uv run marimo edit notebook.py
```

---

## marimo の使い方

**marimo** は Jupyter Notebook の代替となる次世代ノートブックツールです。

### Jupyter Notebook との違い

| 機能 | Jupyter Notebook | marimo |
|---|---|---|
| ファイル形式 | `.ipynb`（JSON） | `.py`（Python） |
| git との相性 | 悪い（JSON が複雑） | 良い（通常の .py ファイル） |
| Coding Agent との相性 | 悪い | 良い（.py なので AI が直接編集できる） |
| セルの依存関係 | 手動管理 | 自動で依存関係を解析・更新 |
| リアクティブ実行 | なし | あり（変数が変わると自動再計算） |

### marimo の起動

```bash
# notebook.py を編集モードで開く
uv run marimo edit notebook.py

# ブラウザが自動で開きます
```

### 基本的な操作

| 操作 | ショートカット |
|---|---|
| セルを実行 | `Ctrl + Enter` (Mac: `Cmd + Enter`) |
| セルを実行して次へ | `Shift + Enter` |
| セルを追加 | セル下部の `+` ボタン |
| セルを削除 | `Ctrl + Shift + Delete` |

### marimo の特徴：リアクティブ実行

marimo はセル間の依存関係を自動で追跡します。
あるセルで変数を変更すると、その変数を使っている**すべてのセルが自動で再実行**されます。

```python
# セル 1
x = 10

# セル 2（x に依存しているので、x が変わると自動再実行）
y = x * 2
```

### サンプルノートブックを開く

このプロジェクトには `notebook.py` にサンプルが用意されています。

```bash
uv run marimo edit notebook.py
```

---

## クイックスタート

```bash
# 1. 仮想環境のセットアップ（初回のみ）
uv sync

# 2. ノートブックを開く
uv run marimo edit notebook.py
```
