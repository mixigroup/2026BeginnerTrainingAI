import marimo

__generated_with = "0.20.1"
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

    - PyTorch モデル（`.pt`）を **ONNX 形式にエクスポート**する方法
    - **Netron** でモデルの構造を可視化する
    - **INT8 量子化**でさらに高速化・軽量化する
    - **プルーニング（枝刈り）**でモデルを軽量化する
    - モデルのファイルサイズ・推論時間を比較する

    > エクスポートとは何か・量子化・枝刈りなどの高速化手法については、スライドを参照してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ONNX とは

    **ONNX（Open Neural Network Exchange）** は、機械学習モデルの標準フォーマットを定義するオープンソースプロジェクトです。
    PyTorch や TensorFlow など、異なるフレームワークで学習したモデルを共通フォーマットに変換することで、フレームワークをまたいで利用できるようになります。

    モデルは**計算グラフ**として表現され、各ノードが Conv・BatchNorm・ReLU などの演算を表します。
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

    from ultralytics import YOLO
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Libraries loaded successfully.")
    return Path, QuantType, YOLO, quantize_dynamic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: ONNX へのエクスポート

    `YOLO.export()` を呼び出すだけで ONNX 形式に変換できます。

    | パラメータ | 説明 |
    |---|---|
    | `format="onnx"` | 出力フォーマット |
    | `imgsz=640` | 入力画像サイズ |
    | `opset=17` | ONNX opset バージョン |
    | `simplify=True` | onnxslim でグラフを最適化 |
    | `end2end=True` | エンドツーエンドモデルとしてエクスポート |
    | `nms=True` | NMS をモデルに組み込む |

    下のセルを実行すると `yolov8m-pose.onnx` が生成されます（初回は `.pt` のダウンロードが走ります）。
    """)
    return


@app.cell
def _(YOLO, mo):
    model = YOLO("yolov8m-pose.pt")
    export_path = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        end2end=True,
        nms=True,
        # half=True,  # FP16 export (disabled by default)
    )
    mo.md(f"**エクスポート完了！** 出力先: `{export_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Netron でモデル構造を確認する

    [https://netron.app](https://netron.app) にアクセスし、生成された `.onnx` ファイルをドラッグ＆ドロップしてください。

    **確認してみよう：**

    - モデルの入力・出力の形状（shape）はどうなっていますか？
    - どんな演算ノード（Conv, BatchNorm, Relu など）が使われていますか？
    """)
    return


@app.cell
def _(Path, mo):
    files = {
        "yolov8m-pose.pt": "PyTorch (.pt)",
        # "yolov8m-pose.onnx": "ONNX (.onnx)",
        # "yolov8m-pose-quantized.onnx": "ONNX qint8 (.onnx)",
    }

    rows = []
    for fname, label in files.items():
        p = Path(fname)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            rows.append(
                {
                    "モデル": label,
                    "ファイル名": fname,
                    "サイズ (MB)": f"{size_mb:.1f}",
                }
            )

    table = mo.ui.table(rows, selection=None)
    mo.md(f"""
    ### 現在のファイルサイズ
    {table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Gradio デモで FPS を比較する

    `src/gradio-demo.py` は Webcam 映像でポーズ推定を実行し、FPS・推論時間をリアルタイムで表示するデモです。

    ### 手順

    1. `src/gradio-demo.py` の `MODEL_FILES` を編集して ONNX 行のコメントアウトを外す

    ```python
    MODEL_FILES = {
        "PyTorch (.pt)": "yolov8m-pose.pt",
        "ONNX (.onnx)": "yolov8m-pose.onnx",  # コメントアウトを外す
    }
    ```

    2. ターミナルで起動する

    ```bash
    uv run python src/gradio-demo.py
    ```

    3. ブラウザが自動で開くので、ラジオボタンでモデルを切り替えながら FPS を比べてみましょう。

    > PyTorch より ONNX の方が速くなっていますか？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: INT8 量子化でエクスポート

    ONNX Runtime の `quantize_dynamic` を使って、モデルの重みを **INT8（8ビット整数）** に変換します。

    - ファイルサイズが約 **1/4** に削減される
    - 整数演算は浮動小数点演算より高速（特に CPU 上）
    - わずかな精度低下が生じる場合がある

    下のセルを実行すると `yolov8m-pose-quantized.onnx` が生成されます（`yolov8m-pose.onnx` が必要です）。
    """)
    return


@app.cell
def _(QuantType, mo, quantize_dynamic):
    model_fp32 = "yolov8m-pose.onnx"
    model_quant = "yolov8m-pose-quantized.onnx"

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
def _(Path, mo):
    quant_entries = [
        ("yolov8m-pose.pt", "PyTorch (.pt)"),
        ("yolov8m-pose.onnx", "ONNX FP32 (.onnx)"),
        ("yolov8m-pose-quantized.onnx", "ONNX INT8 (.onnx)"),
    ]

    quant_rows = []
    quant_base = None
    for quant_name, quant_label in quant_entries:
        quant_path = Path(quant_name)
        if quant_path.exists():
            quant_size = quant_path.stat().st_size / (1024 * 1024)
            if quant_base is None:
                quant_base = quant_size
            quant_ratio = f"{quant_size / quant_base * 100:.0f}%" if quant_base else "-"
            quant_rows.append(
                {
                    "モデル": quant_label,
                    "ファイル名": quant_name,
                    "サイズ (MB)": f"{quant_size:.1f}",
                    ".pt 比": quant_ratio,
                }
            )

    mo.ui.table(quant_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: 量子化モデルの FPS を比較する

    `src/gradio-demo.py` の `MODEL_FILES` に量子化モデルも追加します。
    エクスポートしたファイルを `yolov8m-pose-quantized.onnx` にリネームしてから、コメントアウトを外してください。

    ```python
    MODEL_FILES = {
        "PyTorch (.pt)": "yolov8m-pose.pt",
        "ONNX (.onnx)": "yolov8m-pose.onnx",
        "ONNX qint8 (.onnx)": "yolov8m-pose-quantized.onnx",  # コメントアウトを外す
    }
    ```

    ```bash
    uv run python src/gradio-demo.py
    ```

    3つのモデルを切り替えて、**FPS と精度の変化**を確認してみましょう。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: プルーニング（枝刈り）

    **プルーニング**は、モデルの重みのうち値が小さい（影響の少ない）ものを **ゼロに置き換える** 手法です。

    - `torch.nn.utils.prune` を使い、**L1 Unstructured Pruning** を適用します
    - Conv2d / Linear 層の重みを **30%** ゼロ化します
    - 追加の依存ライブラリは不要（PyTorch 標準機能のみ）

    下のセルを実行すると、プルーニング前後のスパース率が表示され、`yolov8m-pose-pruned.pt` が生成されます。
    """)
    return


@app.cell
def _(Path, YOLO, mo):
    import torch
    import torch.nn.utils.prune as prune

    _INPUT_MODEL = "yolov8m-pose.pt"
    _OUTPUT_MODEL = "yolov8m-pose-pruned.pt"
    _PRUNE_AMOUNT = 0.30

    if not Path(_INPUT_MODEL).exists():
        raise RuntimeError(
            f"`{_INPUT_MODEL}` が見つかりません。先に Step 2 を実行してください。"
        )

    def _calc_sparsity(m):
        total = sum(p.numel() for p in m.parameters())
        zeros = sum((p == 0).sum().item() for p in m.parameters())
        return zeros / total if total > 0 else 0.0

    _yolo = YOLO(_INPUT_MODEL)
    _model = _yolo.model

    _sparsity_before = _calc_sparsity(_model)

    # L1 Unstructured Pruning を適用
    for _module in _model.modules():
        if isinstance(_module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(_module, name="weight", amount=_PRUNE_AMOUNT)

    # 再パラメータ化を除去（weight_orig + weight_mask → weight に統合）
    for _module in _model.modules():
        if isinstance(_module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.remove(_module, "weight")
            except ValueError:
                pass

    _sparsity_after = _calc_sparsity(_model)

    _yolo.save(_OUTPUT_MODEL)

    mo.md(f"""
    **プルーニング完了！** 出力先: `{_OUTPUT_MODEL}`

    | | スパース率 |
    |---|---|
    | プルーニング前 | {_sparsity_before:.2%} |
    | プルーニング後 | {_sparsity_after:.2%} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### プルーニング済みモデルを ONNX にエクスポート

    プルーニング済みの `.pt` モデルも ONNX に変換しておきましょう。
    """)
    return


@app.cell
def _(YOLO, mo):
    _PRUNED_MODEL = "yolov8m-pose-pruned.pt"

    _pruned_yolo = YOLO(_PRUNED_MODEL)
    _pruned_export_path = _pruned_yolo.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        end2end=True,
        nms=True,
    )
    mo.md(f"**エクスポート完了！** 出力先: `{_pruned_export_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### ファイルサイズの比較（プルーニング含む）
    """)
    return


@app.cell
def _(Path, mo):
    all_entries = [
        ("yolov8m-pose.pt", "PyTorch (.pt)"),
        ("yolov8m-pose.onnx", "ONNX FP32 (.onnx)"),
        ("yolov8m-pose-quantized.onnx", "ONNX INT8 (.onnx)"),
        ("yolov8m-pose-pruned.pt", "Pruned PyTorch (.pt)"),
        ("yolov8m-pose-pruned.onnx", "Pruned ONNX (.onnx)"),
    ]

    all_rows = []
    all_base = None
    for entry_name, entry_label in all_entries:
        entry_path = Path(entry_name)
        if entry_path.exists():
            entry_size = entry_path.stat().st_size / (1024 * 1024)
            if all_base is None:
                all_base = entry_size
            entry_ratio = f"{entry_size / all_base * 100:.0f}%" if all_base else "-"
            all_rows.append(
                {
                    "モデル": entry_label,
                    "ファイル名": entry_name,
                    "サイズ (MB)": f"{entry_size:.1f}",
                    ".pt 比": entry_ratio,
                }
            )

    mo.ui.table(all_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8: プルーニング済みモデルの FPS を比較する

    `src/gradio-demo.py` の `MODEL_FILES` にプルーニング済みモデルも追加して FPS を比較してみましょう。

    ```python
    MODEL_FILES = {
        "PyTorch (.pt)": "yolov8m-pose.pt",
        "ONNX (.onnx)": "yolov8m-pose.onnx",
        "ONNX qint8 (.onnx)": "yolov8m-pose-quantized.onnx",
        "Pruned ONNX (.onnx)": "yolov8m-pose-pruned.onnx",  # コメントアウトを外す
    }
    ```

    ```bash
    uv run python src/gradio-demo.py
    ```

    > **注意:** Unstructured Pruning は重みをゼロにするだけで、テンソルの形状は変わりません。
    > そのため、**スパース演算をサポートしないランタイム**（通常の ONNX Runtime CPU など）では
    > FPS が大きく改善しない場合があります。
    >
    > 実運用でスピードアップを狙う場合は、**Structured Pruning**（チャネルごと削除）や
    > スパース対応のハードウェア・ランタイムの利用を検討してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 追加課題：別フォーマットにエクスポートしてみよう

    Ultralytics は `format=` を変えるだけで様々なフォーマットにエクスポートできます。

    | フォーマット | 向いているデバイス | `format=` の値 |
    |---|---|---|
    | CoreML | Mac / iPhone / iPad（macOS のみ） | `"coreml"` |
    | LiteRT (TFLite) | Android / 組み込みデバイス | `"tflite"` |
    | OpenVINO | Intel CPU（要 `uv add openvino`） | `"openvino"` |

    エクスポートしたファイルを Netron に読み込み、ONNX との構造の違いも観察してみましょう。

    ## 参考リンク

    - [Ultralytics Export ドキュメント](https://docs.ultralytics.com/ja/modes/export/)
    - [Netron（ブラウザ版）](https://netron.app)
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
