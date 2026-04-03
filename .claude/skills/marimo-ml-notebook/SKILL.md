---
name: marimo-ml-notebook
description: Marimo ノートブック + HuggingFace モデル/データセットを使った ML ハンズオン教材を作成する。notebook.py、src/ ヘルパーモジュール、pyproject.toml、README.md を一式生成する。「ノートブック作って」「ハンズオン教材」「marimo で ML」「HuggingFace モデルを使って」「学習用ノートブック」などのリクエストで使用する。Marimo のセル構成、src/ のモジュール分割、EDA、可視化パターンを含む。
---

# Marimo ML Notebook スキル

Marimo ノートブックと HuggingFace エコシステムを使った ML ハンズオン教材を効率的に作成するためのスキル。
このプロジェクトで繰り返し使われるパターンを標準化し、一貫性のある教材を生成する。

## 概要

このスキルが生成するファイル一式:

```
XX-project-name/
├── notebook.py          # Marimo ノートブック（メイン教材）
├── pyproject.toml       # プロジェクト設定・依存パッケージ
├── .python-version      # Python 3.12
├── README.md            # notebook と同等の説明 + CLI 実行手順
├── plan.md              # 実装計画
├── .gitignore           # runs/ 等の出力ディレクトリ除外
├── src/
│   ├── __init__.py      # 空
│   └── <domain>_utils.py  # ドメイン固有ヘルパー関数
├── images/
│   └── .gitkeep         # 画像用ディレクトリ
└── runs/                # TensorBoard 等の出力先（gitignore）
```

## ステップ 1: 要件の整理

ユーザーに以下を確認する:

1. **ドメイン**: 画像分類、物体検出、マルチモーダル、テキスト生成、画像生成 etc.
2. **使用モデル**: HuggingFace Hub 上のモデル名（例: `google/siglip2-base-patch16-224`）
3. **データセット**: HuggingFace Datasets のデータセット名（例: `floschne/xm3600`）
4. **可視化**: matplotlib のみか、TensorBoardX も使うか
5. **学習目標**: 何を学んでほしいか（3-5 個）

## ステップ 2: pyproject.toml の生成

```toml
[project]
name = "XX-project-name"
version = "0.1.0"
description = "短い日本語の説明"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # 常に含める
    "marimo>=0.20.2",
    "pyzmq>=27.1.0",
    "torch>=2.0",
    "transformers>=4.48",
    "Pillow>=10.0",
    "matplotlib>=3.8",
    "numpy>=1.26",
    # 必要に応じて追加
    # "datasets>=3.0",        # HF データセット使用時
    # "tensorboardX>=2.6",    # 埋め込み可視化時
    # "tensorboard>=2.15",    # 上記と一緒に
    # "diffusers>=0.32.0",    # 画像生成時
    # "accelerate>=1.0",      # 画像生成時
    # "torchvision>=0.15",    # 物体検出時
    # "requests>=2.31",       # URL から画像取得時
]
```

## ステップ 3: src/ ヘルパーモジュールの作成

`src/<domain>_utils.py` を作成する。以下のレイヤー構成に従う:

### レイヤー 1: データの読み込み・前処理

```python
"""<Domain> ハンズオン用ユーティリティ関数."""
from __future__ import annotations
import io
from typing import TYPE_CHECKING
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from transformers import AutoModel, AutoProcessor


def decode_image(image_field) -> Image.Image:
    """datasets ライブラリの image フィールドを PIL Image に変換する.

    HuggingFace Datasets の image フィールドは PIL Image の場合と
    {'bytes': b'...', 'path': ...} の dict の場合がある。両方に対応する。
    """
    if isinstance(image_field, Image.Image):
        return image_field
    if isinstance(image_field, dict) and "bytes" in image_field:
        return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
    raise ValueError(f"Unknown image format: {type(image_field)}")
```

### レイヤー 2: モデルのロード・推論

```python
def load_model(model_name: str) -> tuple[AutoModel, AutoProcessor]:
    """モデルとプロセッサをロードする."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(model_name).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def encode_images(model, processor, images: list, device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """PIL 画像リストを L2 正規化済み埋め込みベクトルに変換する."""
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    model = model.to(device)
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.get_image_features(**inputs)
        # get_image_features は BaseModelOutputWithPooling を返す場合がある
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)
```

**重要な注意点:**
- `model.get_image_features()` / `model.get_text_features()` は `BaseModelOutputWithPooling` を返すことがある。`.pooler_output` を取得すること
- L2 正規化は常にエンコード時に行う（下流でコサイン類似度 = dot product になる）
- 空リスト入力時のガード（`if not images:`）を忘れないこと
- テキストエンコード時は `padding="max_length"` と `max_length=64` を指定

### レイヤー 3: 可視化・エクスポート

```python
import matplotlib.pyplot as plt

def plot_similarity_heatmap(sim_matrix, row_labels, col_labels, title, ax=None, figsize=(10, 8)):
    """類似度行列をヒートマップとして描画する."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="コサイン類似度")
    # 小さいマトリクスなら数値をセルに表示
    if sim_matrix.shape[0] <= 15 and sim_matrix.shape[1] <= 15:
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                val = sim_matrix[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.tight_layout()
    return fig
```

**matplotlib の日本語フォント対応（macOS）:**
```python
import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "sans-serif"]
```

## ステップ 4: notebook.py の構成

Marimo ノートブックは以下のセクション構成に従う。markdown セルと code セルを交互に配置する。

### 標準セクション構成

```
Part 1: 導入 + モデルロード
  - import marimo as mo（hide_code=True）
  - タイトル + 学習目標（hide_code=True, markdown）
  - 技術説明（hide_code=True, markdown）
  - ライブラリ import + src ヘルパー import
  - 設定セル（MODEL_NAME, DEVICE）
  - モデルロード説明（hide_code=True, markdown）
  - モデルロード実行 + パラメータ数表示

Part 2: データセット + EDA
  - データセット説明（hide_code=True, markdown）
  - N_SAMPLES 設定
  - データセットロード
  - サンプル表示（画像+キャプション）
  - EDA: 画像サイズ分布（ヒストグラム）
  - EDA: テキスト特徴分析

Part 3: メインタスク
  - （ドメイン固有のセル群）
  - 例: エンコード → 類似度計算 → ヒートマップ → 検索

Part 4: 可視化（TensorBoardX 使用時）
  - TensorBoardX 説明（markdown）
  - エクスポート実行
  - TensorBoard 起動手順（markdown）

Part 5: まとめ
  - 学習内容のまとめ表
  - ハンズオン課題チェックリスト
```

### Marimo セルの書き方ルール

**返り値のルール:**
```python
# セルが他のセルから参照される変数を返す場合
@app.cell
def _(MODEL_NAME, load_model, mo):
    model, processor = load_model(MODEL_NAME)
    return model, processor  # 他セルで使う変数を return

# 表示のみのセルは return なし
@app.cell
def _(data, mo):
    mo.output.append(mo.md("結果"))
    return  # 何も返さない
```

**依存関係のルール — これが最も重要:**
- セル内で使う変数は**全て関数の引数に列挙**する必要がある
- `from src.xxx import yyy` で import した関数も return + 引数で受け渡す
- Marimo はこの引数リストでセル間の依存グラフを構築する

```python
# NG: decode_image を使っているが引数に含めていない
@app.cell
def _(xm_dataset, mo):
    img = decode_image(sample["image"])  # NameError!

# OK: 引数に decode_image を含める
@app.cell
def _(xm_dataset, decode_image, mo):
    img = decode_image(sample["image"])
```

**ループ変数の衝突回避 — 必ず `_` プレフィックスを使う:**

Marimo はセル内で代入された変数（ループ変数含む）を全てセルの「出力」として扱う。
`sample`, `i`, `t` のような一般的なループ変数が複数セルで使われると、
`"This cell redefines variables from other cells"` エラーになる。

`_` プレフィックスの変数はセルプライベートとして扱われ、衝突しない。

```python
# NG: i, sample が複数セルで衝突する
@app.cell
def _(dataset, mo):
    for i in range(10):
        sample = dataset[i]
        mo.output.append(mo.md(f"#{i}"))

# OK: _i, _s はセルプライベート
@app.cell
def _(dataset, mo):
    for _i in range(10):
        _s = dataset[_i]
        mo.output.append(mo.md(f"#{_i}"))

# リスト内包表記のループ変数も同様
labels = [f"{_i}:{_t[:10]}" for _i, _t in enumerate(texts)]
```

**表示パターン:**
```python
# 複数の出力を表示（mo.output.append）
@app.cell
def _(model, mo):
    mo.output.append(mo.md("ロード中..."))
    # ... 処理 ...
    mo.output.append(mo.md("✅ 完了"))

# matplotlib を表示
mo.output.append(mo.as_html(fig))

# 画像を表示
mo.output.append(mo.image(pil_image, width=200))

# 画像ギャラリー
items = [mo.vstack([mo.image(img, width=180), mo.md(f"#{i}")]) for i, img in enumerate(images)]
mo.output.append(mo.hstack(items, wrap=True))
```

**hide_code の使い分け:**
- `hide_code=True`: markdown 説明セル、`import marimo as mo` セル
- `hide_code=False`（デフォルト）: 学習者に見てほしいコードセル

## ステップ 5: README.md の作成

README は notebook と同等の情報を含み、CLI でも学習可能にする。`references/readme_template.md` を参照。

### 構成

1. タイトル + 1行フック
2. 学習内容（箇条書き 3-5 個）
3. ディレクトリ構成（ファイルツリー）
4. セットアップ: `uv sync`
5. 実行方法: `uv run marimo edit notebook.py`
6. 技術背景（モデル説明、数式、比較表など）
7. CLI での実行方法（`uv run python -c "..."` で各ステップを再現）
8. ハンズオン課題（チェックリスト）
9. 使用モデル・データセット（リンク付きテーブル）

## ステップ 6: 検証

1. `uv sync` が成功すること
2. `uv run python -c "import marimo"` でインポートできること
3. src/ モジュールのインポートが通ること
4. notebook.py の構文チェック: `python -c "import ast; ast.parse(open('notebook.py').read())"`

## よくあるバグと対策

| バグ | 原因 | 対策 |
|------|------|------|
| `BaseModelOutputWithPooling has no attribute 'norm'` | `get_image_features()` が tensor ではなくオブジェクトを返す | `.pooler_output` を取得する |
| `ValueError: Expected image, got dict` | HF Datasets の image フィールドが bytes dict | `decode_image()` で変換する |
| `NameError` in marimo cell | セルの関数引数に変数が不足 | 使う変数を全て引数に列挙する |
| `This cell redefines variables from other cells` | ループ変数 `i`, `sample` 等が複数セルで衝突 | `_i`, `_s` のように `_` プレフィックスを使う |
| 日本語フォントが豆腐に | matplotlib のフォント設定不足 | `Hiragino Sans` を rcParams に設定 |
| 外部 URL 画像のダウンロード失敗 | レート制限 / ネットワーク | HF Datasets の画像を使うか、PIL で生成する |
| `vmin=0` で負の類似度が見えない | ヒートマップの値域設定 | `vmin=-1, vmax=1` にする |

## HuggingFace Datasets のフィールド名

データセットのフィールド名はデータセットごとに異なる。実装前に必ず以下で確認する:

```python
from datasets import load_dataset
ds = load_dataset("dataset_name", split="train")
sample = ds[0]
print(list(sample.keys()))
for k, v in sample.items():
    print(f"{k}: type={type(v).__name__}")
```

例: `floschne/xm3600` の場合
- `image` → dict（`{'bytes': b'...', 'path': ...}`）、PIL Image ではない
- `captions` → list[str]（複数キャプション）、`caption` ではない
