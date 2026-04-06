# 06 モデルのエクスポート・高速化

## このハンズオンで学ぶこと

- PyTorch モデルを **ONNX 形式にエクスポート**する方法（`torch.onnx.export()`）
- **Netron** でモデルの構造を可視化する
- **INT8 量子化**でさらに高速化・軽量化する
- **プルーニング（枝刈り）**でモデルを軽量化する
- モデルのファイルサイズ・推論時間を比較する

> エクスポートとは何か・量子化・枝刈りなどの高速化手法については、スライドを参照してください。

今回は **SAM（Segment Anything Model）** の Image Encoder を題材にします。

---

## SAM（Segment Anything Model）とは

**SAM** は Meta が開発した汎用セグメンテーションモデルです。
画像上の任意の物体を、クリック（点プロンプト）やバウンディングボックスで指定してセグメントできます。

SAM は以下の **2 つのコンポーネント** で構成されます：

| コンポーネント | 役割 | サイズ |
|---|---|---|
| **Image Encoder（ViT）** | 画像から特徴量を抽出（重い） | ~358 MB |
| **Mask Decoder** | プロンプトからマスクを生成（軽い） | ~16 MB |

推論のボトルネックは **Image Encoder** です。
このノートブックでは **Encoder 部分を ONNX 化・量子化・プルーニング**して高速化を体験します。

---

## ONNX とは

**ONNX（Open Neural Network Exchange）** は、機械学習モデルの標準フォーマットを定義するオープンソースプロジェクトです。PyTorch や TensorFlow など、異なるフレームワークで学習したモデルを共通フォーマットに変換することで、フレームワークをまたいで利用できるようになります。

モデルは**計算グラフ**として表現され、各ノードが MatMul・LayerNorm・Softmax などの演算を表します。このグラフ表現が統一されているため、ONNX Runtime や各種推論エンジンがフレームワークに依存せず高速に実行できます。

---

## セットアップ

```bash
# 依存ライブラリをインストール（初回のみ）
uv sync
```

## 実行方法

```bash
uv run marimo edit notebook.py
```

---

## ノートブックの構成

| Step | 内容 |
|---|---|
| **Step 1** | ライブラリのインポート |
| **Step 2** | SAM モデルのロード（`facebook/sam-vit-base`） |
| **Step 3** | Image Encoder を `torch.onnx.export()` で ONNX にエクスポート |
| **Step 4** | Netron でモデル構造を確認 |
| **Step 5** | Gradio デモで推論時間を比較（PyTorch vs ONNX） |
| **Step 6** | INT8 量子化でエクスポート（`onnxruntime.quantization`） |
| **Step 7** | 量子化モデルの推論時間を比較 |
| **Step 8** | プルーニング（L1 Unstructured Pruning）を適用 |
| **Step 9** | プルーニング済みモデルの推論時間を比較 |

---

## 各 Step の詳細

### Step 3: ONNX エクスポート

`torch.onnx.export()` を使って SAM の Image Encoder を ONNX 形式に変換します。

```python
torch.onnx.export(
    encoder,
    dummy_input,           # トレース用ダミー入力
    "sam-vit-b-encoder.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    opset_version=18,
)
```

### Step 4: Netron で可視化

[https://netron.app](https://netron.app) にアクセスし、生成された `.onnx` ファイルをドラッグ＆ドロップしてください。

**確認してみよう：**

- モデルの入力・出力の形状（shape）はどうなっていますか？
- ViT（Vision Transformer）特有の演算ノード（MatMul, LayerNorm, Softmax など）が見えますか？
- CNN（Conv, BatchNorm, ReLU）との違いは何ですか？

### Step 5 / 7 / 9: Gradio デモで推論時間を比較

`src/gradio-demo.py` は画像をアップロードしてクリックした箇所をセグメントするデモです。
Encoder の推論時間をリアルタイムで表示します。

```bash
uv run python src/gradio-demo.py
```

`MODEL_FILES` のコメントアウトを外して、各モデルの推論時間を比較できます。

```python
MODEL_FILES = {
    "PyTorch": None,
    "ONNX FP32": "sam-vit-b-encoder.onnx",
    "ONNX INT8": "sam-vit-b-encoder-quantized.onnx",
    "Pruned ONNX": "sam-vit-b-encoder-pruned.onnx",
}
```

### Step 6: INT8 量子化

ONNX Runtime の `quantize_dynamic` を使って、重みを INT8（8ビット整数）に変換します。

- ファイルサイズが約 **1/4** に削減される
- 整数演算は浮動小数点演算より高速（特に CPU 上）
- わずかな精度低下が生じる場合がある

### Step 8: プルーニング（枝刈り）

`torch.nn.utils.prune` の **L1 Unstructured Pruning** を適用し、ViT の Linear 層の重みを **30%** ゼロ化します。

> **注意:** Unstructured Pruning は重みをゼロにするだけで、テンソルの形状は変わりません。そのため、スパース演算をサポートしないランタイム（通常の ONNX Runtime CPU など）では推論時間が大きく改善しない場合があります。実運用でスピードアップを狙う場合は、Structured Pruning（チャネルごと削除）やスパース対応のハードウェア・ランタイムの利用を検討してください。

---

## 追加課題：別サイズの SAM モデルを試してみよう

SAM には 3 つのサイズがあります。サイズが大きいほど精度が高いですが、推論も遅くなります。

| モデル | パラメータ数 | Hugging Face ID |
|---|---|---|
| SAM ViT-B（本ノートブック） | ~91M | `facebook/sam-vit-base` |
| SAM ViT-L | ~308M | `facebook/sam-vit-large` |
| SAM ViT-H | ~636M | `facebook/sam-vit-huge` |

より大きなモデルで ONNX エクスポート・量子化を試し、サイズ削減率や推論時間の変化を比較してみましょう。

---

## 参考リンク

- [SAM 論文 (Segment Anything)](https://arxiv.org/abs/2304.02643)
- [SAM GitHub リポジトリ](https://github.com/facebookresearch/segment-anything)
- [facebook/sam-vit-base (Hugging Face)](https://huggingface.co/facebook/sam-vit-base)
- [Netron（ブラウザ版）](https://netron.app)
- [ONNX Runtime 量子化ドキュメント](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
