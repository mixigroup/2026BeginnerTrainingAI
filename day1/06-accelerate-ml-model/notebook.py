import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 モデルのエクスポート・高速化

    このノートブックでは以下を学びます：

    - PyTorch モデルを **ONNX 形式にエクスポート**する方法（`torch.onnx.export()`）
    - **Netron** でモデルの構造を可視化する
    - **INT8 量子化**でさらに高速化・軽量化する
    - **プルーニング（枝刈り）**でモデルを軽量化する
    - モデルのファイルサイズ・推論時間を比較する

    > エクスポートとは何か・量子化・枝刈りなどの高速化手法については、スライドを参照してください。

    今回は **SAM（Segment Anything Model）** の Image Encoder を題材にします。

    ![SAM セグメンテーション例](https://github.com/facebookresearch/segment-anything/raw/main/assets/masks2.jpg)

    *画像出典: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (Apache 2.0 License)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SAM（Segment Anything Model）とは

    **SAM** は Meta が開発した汎用セグメンテーションモデルです。
    画像上の任意の物体を、クリック（点プロンプト）やバウンディングボックスで指定してセグメントできます。

    SAM は以下の **2 つのコンポーネント** で構成されます：

    | コンポーネント | 役割 | サイズ |
    |---|---|---|
    | **Image Encoder（ViT）** | 画像から特徴量を抽出（重い） | ~358 MB |
    | **Mask Decoder** | プロンプトからマスクを生成（軽い） | ~16 MB |

    ![SAM モデルアーキテクチャ](https://github.com/facebookresearch/segment-anything/blob/main/assets/model_diagram.png?raw=true)

    *画像出典: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (Apache 2.0 License)*

    推論のボトルネックは **Image Encoder** です。
    このノートブックでは **Encoder 部分を ONNX 化・量子化・プルーニング**して高速化を体験します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ONNX とは

    **ONNX（Open Neural Network Exchange）** は、機械学習モデルの標準フォーマットを定義するオープンソースプロジェクトです。
    PyTorch や TensorFlow など、異なるフレームワークで学習したモデルを共通フォーマットに変換することで、フレームワークをまたいで利用できるようになります。

    モデルは**計算グラフ**として表現され、各ノードが MatMul・LayerNorm・Softmax などの演算を表します。
    このグラフ表現が統一されているため、ONNX Runtime や各種推論エンジンがフレームワークに依存せず高速に実行できます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: ライブラリのインポート

    まず必要なライブラリをインポートします。
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import SamModel

    print("Libraries loaded successfully.")
    return Path, QuantType, SamModel, quantize_dynamic, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: SAM モデルのロード

    Hugging Face Hub から `facebook/sam-vit-base` をダウンロードします（初回のみ時間がかかります）。
    """)
    return


@app.cell
def _(SamModel, mo):
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    sam_model.eval()

    # パラメータ数を表示
    encoder_params = sum(p.numel() for p in sam_model.vision_encoder.parameters())
    decoder_params = sum(
        p.numel()
        for name in ("prompt_encoder", "mask_decoder")
        for p in getattr(sam_model, name).parameters()
    )

    mo.md(f"""
    **モデルロード完了！**

    | コンポーネント | パラメータ数 |
    |---|---|
    | Image Encoder (ViT) | {encoder_params:,} |
    | Prompt Encoder + Mask Decoder | {decoder_params:,} |
    | **合計** | **{encoder_params + decoder_params:,}** |
    """)
    return (sam_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Image Encoder を ONNX にエクスポート

    `torch.onnx.export()` を使って、SAM の **Image Encoder** を ONNX 形式に変換します。

    | パラメータ | 説明 |
    |---|---|
    | `model` | エクスポートするモデル（ここでは `vision_encoder`） |
    | `args` | ダミー入力（トレース用） |
    | `f` | 出力ファイルパス |
    | `input_names` | 入力テンソルの名前 |
    | `output_names` | 出力テンソルの名前 |
    | `opset_version` | ONNX opset バージョン（18 以上推奨） |

    下のセルを実行すると `sam-vit-b-encoder.onnx` が生成されます。
    """)
    return


@app.cell
def _(mo, sam_model, torch):
    _encoder = sam_model.vision_encoder
    _dummy_input = torch.randn(1, 3, 1024, 1024)
    _onnx_path = "sam-vit-b-encoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            _encoder,
            _dummy_input,
            _onnx_path,
            input_names=["pixel_values"],
            output_names=["image_embeddings"],
            opset_version=18,
            do_constant_folding=True,
        )
    mo.md(f"**エクスポート完了！** 出力先: `{_onnx_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Netron でモデル構造を確認する

    [https://netron.app](https://netron.app) にアクセスし、生成された `.onnx` ファイルをドラッグ＆ドロップしてください。

    **確認してみよう：**

    - モデルの入力・出力の形状（shape）はどうなっていますか？
    - ViT（Vision Transformer）特有の演算ノード（MatMul, LayerNorm, Softmax など）が見えますか？
    - CNN（Conv, BatchNorm, ReLU）との違いは何ですか？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 現在のファイルサイズ
    """)
    return


@app.cell
def _(Path):
    def onnx_size_bytes(path: Path) -> int:
        """ONNXファイルと外部データファイル(.onnx.data)の合計サイズを返す"""
        total = path.stat().st_size
        data_path = Path(str(path) + ".data")
        if data_path.exists():
            total += data_path.stat().st_size
        return total

    return (onnx_size_bytes,)


@app.cell
def _(Path, mo, onnx_size_bytes):
    files = {
        "sam-vit-b-encoder.onnx": "ONNX FP32 (.onnx)",
    }

    rows = []
    for fname, label in files.items():
        p = Path(fname)
        if p.exists():
            size_mb = onnx_size_bytes(p) / (1024 * 1024)
            rows.append(
                {
                    "モデル": label,
                    "ファイル名": fname,
                    "サイズ (MB)": f"{size_mb:.1f}",
                }
            )

    table = mo.ui.table(rows, selection=None)
    mo.md(f"""
    {table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: Gradio デモで推論時間を比較する

    `src/gradio-demo.py` は画像をアップロードしてクリックした箇所をセグメントするデモです。
    Encoder の推論時間をリアルタイムで表示します。

    ### 手順

    1. ターミナルで起動する

    ```bash
    uv run python src/gradio-demo.py
    ```

    2. ブラウザが自動で開くので、画像をアップロードしてクリックしてみましょう。

    3. ラジオボタンでモデル（PyTorch / ONNX）を切り替えて、推論時間を比較してみましょう。

    > PyTorch より ONNX の方が速くなっていますか？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: INT8 量子化でエクスポート

    ONNX Runtime の `quantize_dynamic` を使って、**Encoder** の重みを **INT8（8ビット整数）** に変換します。

    - ファイルサイズが約 **1/4** に削減される
    - 整数演算は浮動小数点演算より高速（特に CPU 上）
    - わずかな精度低下が生じる場合がある

    下のセルを実行すると `sam-vit-b-encoder-quantized.onnx` が生成されます（`sam-vit-b-encoder.onnx` が必要です）。
    """)
    return


@app.cell
def _(QuantType, mo, quantize_dynamic):
    model_fp32 = "sam-vit-b-encoder.onnx"
    model_quant = "sam-vit-b-encoder-quantized.onnx"

    quantize_dynamic(
        model_fp32,
        model_quant,
        weight_type=QuantType.QInt8,
    )
    mo.md(f"**量子化完了！** 出力先: `{model_quant}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### ファイルサイズの比較
    """)
    return


@app.cell
def _(Path, mo, onnx_size_bytes):
    quant_entries = [
        ("sam-vit-b-encoder.onnx", "ONNX FP32"),
        ("sam-vit-b-encoder-quantized.onnx", "ONNX INT8"),
    ]

    quant_rows = []
    quant_base = None
    for quant_name, quant_label in quant_entries:
        quant_path = Path(quant_name)
        if quant_path.exists():
            quant_size = onnx_size_bytes(quant_path) / (1024 * 1024)
            if quant_base is None:
                quant_base = quant_size
            quant_ratio = f"{quant_size / quant_base * 100:.0f}%" if quant_base else "-"
            quant_rows.append(
                {
                    "モデル": quant_label,
                    "ファイル名": quant_name,
                    "サイズ (MB)": f"{quant_size:.1f}",
                    "FP32 比": quant_ratio,
                }
            )

    mo.ui.table(quant_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: 量子化モデルの推論時間を比較する

    ```bash
    uv run python src/gradio-demo.py
    ```

    3 つのモデルを切り替えて、**推論時間と精度の変化**を確認してみましょう。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8: プルーニング（枝刈り）

    **プルーニング**は、モデルの重みのうち値が小さい（影響の少ない）ものを **ゼロに置き換える** 手法です。

    - `torch.nn.utils.prune` を使い、**L1 Unstructured Pruning** を適用します
    - ViT の **Linear 層**（Attention の QKV 射影や MLP）の重みを **30%** ゼロ化します
    - 追加の依存ライブラリは不要（PyTorch 標準機能のみ）

    > **CNN との違い:** CNN では Conv2d 層がパラメータの大部分を占めますが、
    > Vision Transformer (ViT) では **Linear 層が支配的**です。
    > そのため、プルーニングのターゲットも Linear 層になります。

    下のセルを実行すると、プルーニング前後のスパース率が表示され、`sam-vit-b-encoder-pruned.pt` が生成されます。
    """)
    return


@app.cell
def _(Path, mo, sam_model, torch):
    import copy

    import torch.nn.utils.prune as prune

    _ONNX_PATH = "sam-vit-b-encoder.onnx"
    _OUTPUT_MODEL = "sam-vit-b-encoder-pruned.pt"
    _PRUNE_AMOUNT = 0.30

    if not Path(_ONNX_PATH).exists():
        raise RuntimeError(
            f"`{_ONNX_PATH}` が見つかりません。先に Step 3 を実行してください。"
        )

    # Encoder のディープコピーを作成（元のモデルを汚染しない）
    pruned_encoder = copy.deepcopy(sam_model.vision_encoder)

    def _calc_sparsity(m):
        total = sum(p.numel() for p in m.parameters())
        zeros = sum((p == 0).sum().item() for p in m.parameters())
        return zeros / total if total > 0 else 0.0

    _sparsity_before = _calc_sparsity(pruned_encoder)

    # L1 Unstructured Pruning を Linear 層に適用
    for _module in pruned_encoder.modules():
        if isinstance(_module, torch.nn.Linear):
            prune.l1_unstructured(_module, name="weight", amount=_PRUNE_AMOUNT)

    # 再パラメータ化を除去（weight_orig + weight_mask → weight に統合）
    for _module in pruned_encoder.modules():
        if isinstance(_module, torch.nn.Linear):
            try:
                prune.remove(_module, "weight")
            except ValueError:
                pass

    _sparsity_after = _calc_sparsity(pruned_encoder)

    # プルーニング済み Encoder の state_dict を保存
    torch.save(pruned_encoder.state_dict(), _OUTPUT_MODEL)

    mo.md(f"""
    **プルーニング完了！** 出力先: `{_OUTPUT_MODEL}`

    | | スパース率 |
    |---|---|
    | プルーニング前 | {_sparsity_before:.2%} |
    | プルーニング後 | {_sparsity_after:.2%} |
    """)
    return (pruned_encoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### プルーニング済み Encoder を ONNX にエクスポート

    プルーニング済みの Encoder も ONNX に変換しておきましょう。
    """)
    return


@app.cell
def _(mo, pruned_encoder, torch):
    _dummy = torch.randn(1, 3, 1024, 1024)
    _pruned_onnx_path = "sam-vit-b-encoder-pruned.onnx"

    with torch.no_grad():
        torch.onnx.export(
            pruned_encoder,
            _dummy,
            _pruned_onnx_path,
            input_names=["pixel_values"],
            output_names=["image_embeddings"],
            opset_version=18,
            do_constant_folding=True,
        )
    mo.md(f"**エクスポート完了！** 出力先: `{_pruned_onnx_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### ファイルサイズの比較（プルーニング含む）
    """)
    return


@app.cell
def _(Path, mo, onnx_size_bytes):
    all_entries = [
        ("sam-vit-b-encoder.onnx", "ONNX FP32"),
        ("sam-vit-b-encoder-quantized.onnx", "ONNX INT8"),
        ("sam-vit-b-encoder-pruned.onnx", "Pruned ONNX FP32"),
    ]

    all_rows = []
    all_base = None
    for entry_name, entry_label in all_entries:
        entry_path = Path(entry_name)
        if entry_path.exists():
            entry_size = onnx_size_bytes(entry_path) / (1024 * 1024)
            if all_base is None:
                all_base = entry_size
            entry_ratio = f"{entry_size / all_base * 100:.0f}%" if all_base else "-"
            all_rows.append(
                {
                    "モデル": entry_label,
                    "ファイル名": entry_name,
                    "サイズ (MB)": f"{entry_size:.1f}",
                    "FP32 比": entry_ratio,
                }
            )

    mo.ui.table(all_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 9: プルーニング済みモデルの推論時間を比較する

    ```bash
    uv run python src/gradio-demo.py
    ```

    > **注意:** Unstructured Pruning は重みをゼロにするだけで、テンソルの形状は変わりません。
    > そのため、**スパース演算をサポートしないランタイム**（通常の ONNX Runtime CPU など）では推論時間が大きく改善しない場合があります。
    >
    > 実運用でスピードアップを狙う場合は、**Structured Pruning**（チャネルごと削除）やスパース対応のハードウェア・ランタイムの利用を検討してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 追加課題

    ### 別サイズの SAM モデルを試してみよう

    SAM には 3 つのサイズがあります。サイズが大きいほど精度が高いですが、推論も遅くなります。

    | モデル | パラメータ数 | Hugging Face ID |
    |---|---|---|
    | SAM ViT-B（本ノートブック） | ~91M | `facebook/sam-vit-base` |
    | SAM ViT-L | ~308M | `facebook/sam-vit-large` |
    | SAM ViT-H | ~636M | `facebook/sam-vit-huge` |

    より大きなモデルで ONNX エクスポート・量子化を試し、サイズ削減率や推論時間の変化を比較してみましょう。

    ## 参考リンク

    - [SAM 論文 (Segment Anything)](https://arxiv.org/abs/2304.02643)
    - [SAM GitHub リポジトリ](https://github.com/facebookresearch/segment-anything)
    - [facebook/sam-vit-base (Hugging Face)](https://huggingface.co/facebook/sam-vit-base)
    - [Netron（ブラウザ版）](https://netron.app)
    - [ONNX Runtime 量子化ドキュメント](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

    ## Citation

    このノートブックで使用している SAM（Segment Anything Model）および画像は、Meta AI Research が公開しているものです。

    ```
    @article{kirillov2023segment,
      title={Segment Anything},
      author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Dollar, Piotr and Girshick, Ross},
      journal={arXiv:2304.02643},
      year={2023}
    }
    ```

    - **リポジトリ**: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
    - **ライセンス**: Apache 2.0
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
