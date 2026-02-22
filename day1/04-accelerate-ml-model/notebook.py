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
    # 04 モデルのエクスポート・高速化

    このノートブックでは以下を学びます：

    - PyTorch モデル（`.pt`）を **ONNX 形式にエクスポート**する方法
    - **Netron** でモデルの構造を可視化する
    - **INT8 量子化**でさらに高速化・軽量化する
    - 3つのモデルのファイルサイズ・推論時間を比較する

    > エクスポートとは何か・量子化・枝刈りなどの高速化手法については、スライドを参照してください。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ONNX とは

    **ONNX（Open Neural Network Exchange）** は、機械学習モデルの標準フォーマットを定義するオープンソースプロジェクトです。
    PyTorch や TensorFlow など、異なるフレームワークで学習したモデルを共通フォーマットに変換することで、フレームワークをまたいで利用できるようになります。

    モデルは**計算グラフ**として表現され、各ノードが Conv・BatchNorm・ReLU などの演算を表します。
    このグラフ表現が統一されているため、ONNX Runtime や各種推論エンジンがフレームワークに依存せず高速に実行できます。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: ライブラリのインポート

    まず必要なライブラリをインポートします。
    """)
    return


@app.cell
def _():
    import os
    import time

    from ultralytics import YOLO
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort

    print("Libraries loaded successfully.")
    return QuantType, YOLO, os, quantize_dynamic


@app.cell
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

    下のセルを実行すると `yolo26m-pose.onnx` が生成されます（初回は `.pt` のダウンロードが走ります）。
    """)
    return


@app.cell
def _(mo):
    run_export_button = mo.ui.run_button(label="ONNX エクスポートを実行")
    run_export_button
    return (run_export_button,)


@app.cell
def _(YOLO, mo, run_export_button):
    mo.stop(not run_export_button.value, mo.md("▲ ボタンを押して ONNX エクスポートを実行してください。"))

    _model = YOLO("yolo26m-pose.pt")
    _export_path = _model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        end2end=True,
        nms=True,
        # half=True,  # FP16 export (disabled by default)
    )
    mo.md(f"**エクスポート完了！** 出力先: `{_export_path}`")
    return


@app.cell
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
def _(mo, os):
    _files = {
        "yolo26m-pose.pt": "PyTorch (.pt)",
        "yolo26m-pose.onnx": "ONNX (.onnx)",
        "yolo26m-pose-quantized.onnx": "ONNX qint8 (.onnx)",
    }

    _rows = []
    for _fname, _label in _files.items():
        if os.path.exists(_fname):
            _size_mb = os.path.getsize(_fname) / (1024 * 1024)
            _rows.append({"モデル": _label, "ファイル名": _fname, "サイズ (MB)": f"{_size_mb:.1f}"})

    if _rows:
        mo.md(
            "### 現在のファイルサイズ\n\n"
            + mo.as_html(
                mo.ui.table(
                    _rows,
                    selection=None,
                )
            ).text
        )
    else:
        mo.md("_まだモデルファイルが見つかりません。Step 2 を実行してください。_")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Gradio デモで FPS を比較する

    `src/gradio-demo.py` は Webcam 映像でポーズ推定を実行し、FPS・推論時間をリアルタイムで表示するデモです。

    ### 手順

    1. `src/gradio-demo.py` の `MODEL_FILES` を編集して ONNX 行のコメントアウトを外す

    ```python
    MODEL_FILES = {
        "PyTorch (.pt)": "yolo26m-pose.pt",
        "ONNX (.onnx)": "yolo26m-pose.onnx",  # コメントアウトを外す
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: INT8 量子化でエクスポート

    ONNX Runtime の `quantize_dynamic` を使って、モデルの重みを **INT8（8ビット整数）** に変換します。

    - ファイルサイズが約 **1/4** に削減される
    - 整数演算は浮動小数点演算より高速（特に CPU 上）
    - わずかな精度低下が生じる場合がある

    下のセルを実行すると `yolo26m-pose-quantized.onnx` が生成されます（`yolo26m-pose.onnx` が必要です）。
    """)
    return


@app.cell
def _(mo):
    run_quant_button = mo.ui.run_button(label="INT8 量子化を実行")
    run_quant_button
    return (run_quant_button,)


@app.cell
def _(QuantType, mo, os, quantize_dynamic, run_quant_button):
    mo.stop(not run_quant_button.value, mo.md("▲ ボタンを押して INT8 量子化を実行してください。"))

    _model_fp32 = "yolo26m-pose.onnx"
    _model_quant = "yolo26m-pose-quantized.onnx"

    if not os.path.exists(_model_fp32):
        mo.stop(True, mo.callout(mo.md(f"`{_model_fp32}` が見つかりません。先に Step 2 を実行してください。"), kind="warn"))

    quantize_dynamic(
        _model_fp32,
        _model_quant,
        weight_type=QuantType.QInt8,
    )
    mo.md(f"**量子化完了！** 出力先: `{_model_quant}`")
    return


@app.cell
def _(mo):
    mo.md("""
    ### ファイルサイズの比較
    """)
    return


@app.cell
def _(mo, os):
    _files = [
        ("yolo26m-pose.pt", "PyTorch (.pt)"),
        ("yolo26m-pose.onnx", "ONNX FP32 (.onnx)"),
        ("yolo26m-pose-quantized.onnx", "ONNX INT8 (.onnx)"),
    ]

    _rows = []
    _base_size = None
    for _fname, _label in _files:
        if os.path.exists(_fname):
            _size_mb = os.path.getsize(_fname) / (1024 * 1024)
            if _base_size is None:
                _base_size = _size_mb
            _ratio = f"{_size_mb / _base_size * 100:.0f}%" if _base_size else "-"
            _rows.append({
                "モデル": _label,
                "ファイル名": _fname,
                "サイズ (MB)": f"{_size_mb:.1f}",
                ".pt 比": _ratio,
            })

    if _rows:
        mo.ui.table(_rows, selection=None)
    else:
        mo.md("_モデルファイルが見つかりません。Step 2・Step 5 を実行してください。_")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6: 量子化モデルの FPS を比較する

    `src/gradio-demo.py` の `MODEL_FILES` に量子化モデルも追加します。
    エクスポートしたファイルを `yolo26m-pose-quantized.onnx` にリネームしてから、コメントアウトを外してください。

    ```python
    MODEL_FILES = {
        "PyTorch (.pt)": "yolo26m-pose.pt",
        "ONNX (.onnx)": "yolo26m-pose.onnx",
        "ONNX qint8 (.onnx)": "yolo26m-pose-quantized.onnx",  # コメントアウトを外す
    }
    ```

    ```bash
    uv run python src/gradio-demo.py
    ```

    3つのモデルを切り替えて、**FPS と精度の変化**を確認してみましょう。
    """)
    return


@app.cell
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


if __name__ == "__main__":
    app.run()
