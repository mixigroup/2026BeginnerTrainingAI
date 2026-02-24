import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys

    sys.path.insert(0, ".")

    import torch
    import matplotlib.pyplot as plt

    from src.preprocess import load_image, get_processor, preprocess_image
    from src.inference import load_model, run_inference
    from src.postprocess import postprocess_results, visualize_results
    from src.utils import download_sample_image, get_label_name, build_label_map

    return (
        sys,
        torch,
        plt,
        load_image,
        get_processor,
        preprocess_image,
        load_model,
        run_inference,
        postprocess_results,
        visualize_results,
        download_sample_image,
        get_label_name,
        build_label_map,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Object Detection ハンズオン：画像のML推論

        このノートブックでは、**DETR（Detection Transformer）** を使って、
        画像中の物体を検出するObject Detectionを体験します。

        ## 推論の3つの共通フェーズ

        どんなモデルでも、推論は以下の3フェーズで構成されます。

        | フェーズ | 内容 |
        |---|---|
        | **1. Preprocess** | 画像を tensor（多次元配列）に変換・正規化 |
        | **2. Forward** | モデルに tensor を入れて、出力 tensor を得る |
        | **3. Postprocess** | 出力 tensor から bounding box とクラスを取り出す |

        ## 今回扱うモデル

        - **`facebook/detr-resnet-50`**（DETR: Detection Transformer）
          - Transformer ベースの Object Detection モデル
          - COCO データセット（80クラス）で学習済み
          - HuggingFace `transformers` で簡単に利用可能
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 1: Preprocess（前処理）

        生の画像データをモデルが受け付ける形式に変換します。

        ### 前処理で行うこと

        1. **画像の読み込み** - URL またはローカルパスから PIL Image として読み込む
        2. **リサイズ** - モデルが期待するサイズに変換（DETR は短辺 800px を基準）
        3. **正規化** - ピクセル値を ImageNet の mean/std で標準化
        4. **テンソル変換** - NumPy/PIL → PyTorch Tensor（形状: `[1, 3, H, W]`）
        """
    )
    return


@app.cell
def _():
    # --- Settings: Edit these to experiment! ---
    MODEL_NAME = "facebook/detr-resnet-50"
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return MODEL_NAME, IMAGE_URL


@app.cell
def _(IMAGE_URL, download_sample_image, mo):
    sample_image = download_sample_image(IMAGE_URL)
    mo.md(
        f"""
        ### サンプル画像

        - URL: `{IMAGE_URL}`
        - サイズ: **{sample_image.width} x {sample_image.height}** px
        - モード: **{sample_image.mode}**
        """
    )
    return (sample_image,)


@app.cell
def _(plt, sample_image):
    fig_original, ax_original = plt.subplots(figsize=(10, 7))
    ax_original.imshow(sample_image)
    ax_original.axis("off")
    ax_original.set_title("Input Image")
    fig_original
    return ax_original, fig_original


@app.cell
def _(MODEL_NAME, get_processor, mo):
    processor = get_processor(MODEL_NAME)
    mo.md(
        f"""
        ### Image Processor のロード

        - モデル名: `{MODEL_NAME}`
        - Processor クラス: `{type(processor).__name__}`
        - 正規化 mean: `{processor.image_mean}`
        - 正規化 std: `{processor.image_std}`
        """
    )
    return (processor,)


@app.cell
def _(mo, preprocess_image, processor, sample_image):
    inputs = preprocess_image(sample_image, processor)
    pixel_values = inputs["pixel_values"]

    mo.md(
        f"""
        ### 前処理結果：pixel_values の中身

        | 項目 | 値 |
        |---|---|
        | shape | `{tuple(pixel_values.shape)}` |
        | dtype | `{pixel_values.dtype}` |
        | 最小値 | `{pixel_values.min().item():.4f}` |
        | 最大値 | `{pixel_values.max().item():.4f}` |
        | 平均値 | `{pixel_values.mean().item():.4f}` |

        > **ポイント**: 正規化後の値は 0〜255 ではなく、負の値も含む範囲になっています。
        > これは ImageNet の mean/std で標準化されているためです。
        """
    )
    return inputs, pixel_values


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 2: Forward（推論）

        前処理済みの tensor をモデルに入力し、生の出力（logits）を得ます。

        ### DETR の出力

        DETR は Transformer ベースのモデルで、以下を出力します：

        | 出力 | 形状 | 内容 |
        |---|---|---|
        | `logits` | `[1, 100, 92]` | 各クエリのクラス スコア（92 = 80クラス + "no object" + padding） |
        | `pred_boxes` | `[1, 100, 4]` | 各クエリの bounding box（cx, cy, w, h、正規化済み [0, 1]） |

        > DETR は 100 個のオブジェクトクエリを使って同時に複数の物体を検出します。
        """
    )
    return


@app.cell
def _(MODEL_NAME, load_model, mo):
    model = load_model(MODEL_NAME)
    mo.md(
        f"""
        ### モデルのロード

        - モデル名: `{MODEL_NAME}`
        - モデルクラス: `{type(model).__name__}`
        - パラメータ数: **{sum(p.numel() for p in model.parameters()):,}**
        - クラス数: **{model.config.num_labels}**（COCO 80クラス）
        """
    )
    return (model,)


@app.cell
def _(inputs, mo, model, run_inference):
    outputs = run_inference(model, inputs)
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes

    mo.md(
        f"""
        ### 推論結果（生の出力）

        | テンソル | 形状 | 内容 |
        |---|---|---|
        | `logits` | `{tuple(logits.shape)}` | クラス スコア（softmax 前） |
        | `pred_boxes` | `{tuple(pred_boxes.shape)}` | bbox（cx, cy, w, h）|

        #### logits の先頭クエリ（上位5スコアのクラス）

        ```
        scores = softmax(logits[0, 0]) の上位5クラス:
        {", ".join(f"class_{i}: {v:.3f}" for i, v in sorted(enumerate(logits.softmax(-1)[0, 0].tolist()), key=lambda x: -x[1])[:5])}
        ```

        #### pred_boxes の先頭クエリ（cx, cy, w, h）

        ```
        {pred_boxes[0, 0].tolist()}
        ```

        > **ポイント**: この時点では 100 クエリ全員が何らかの値を持っています。
        > 次フェーズの後処理で confidence 閾値を使って絞り込みます。
        """
    )
    return logits, outputs, pred_boxes


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 3: Postprocess（後処理）

        生の出力 tensor から、人間が理解できる検出結果に変換します。

        ### 後処理で行うこと

        1. **Softmax** - logits を確率に変換
        2. **閾値フィルタリング** - confidence スコアが閾値以上の検出のみ残す
        3. **座標変換** - 正規化 [0, 1] の相対座標 → 画像上のピクセル座標
        4. **ラベル変換** - クラスid → クラス名（"cat"、"dog" など）
        """
    )
    return


@app.cell
def _():
    # --- Threshold: Edit this to see more/fewer detections ---
    CONFIDENCE_THRESHOLD = 0.9
    return (CONFIDENCE_THRESHOLD,)


@app.cell
def _(
    CONFIDENCE_THRESHOLD,
    build_label_map,
    mo,
    model,
    outputs,
    postprocess_results,
    sample_image,
    processor,
):
    label_map = build_label_map(model)
    detections = postprocess_results(
        outputs,
        processor,
        image_size=(sample_image.width, sample_image.height),
        threshold=CONFIDENCE_THRESHOLD,
    )

    rows = "\n".join(
        f"| {label_map.get(d['label'], d['label'])} | {d['score']:.3f} "
        f"| ({d['box']['xmin']:.0f}, {d['box']['ymin']:.0f}, "
        f"{d['box']['xmax']:.0f}, {d['box']['ymax']:.0f}) |"
        for d in detections
    )

    mo.md(
        f"""
        ### 後処理結果（閾値: {CONFIDENCE_THRESHOLD}）

        **検出数: {len(detections)} 個**

        | クラス | スコア | bbox (xmin, ymin, xmax, ymax) |
        |---|---|---|
        {rows if rows else "| - | - | - |"}

        > **閾値を変えて試してみよう**: `CONFIDENCE_THRESHOLD` を 0.5 にすると、より多くの検出が現れます。
        """
    )
    return detections, label_map, rows


@app.cell
def _(detections, label_map, sample_image, visualize_results):
    result_fig = visualize_results(sample_image, detections, label_names=label_map)
    result_fig
    return (result_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## まとめ

        このハンズオンで体験した3フェーズをふり返ります。

        | フェーズ | 処理 | 入出力 |
        |---|---|---|
        | **Preprocess** | 画像 → tensor | `PIL.Image` → `{pixel_values: Tensor[1,3,H,W]}` |
        | **Forward** | tensor → 生出力 | `{pixel_values}` → `{logits: [1,100,92], pred_boxes: [1,100,4]}` |
        | **Postprocess** | 生出力 → 検出結果 | `logits + pred_boxes` → `[{label, score, box}, ...]` |

        ### 発展課題

        - `CONFIDENCE_THRESHOLD` を変えて検出数の変化を確認する
        - `IMAGE_URL` を別の画像に変えて試す
        - `MODEL_NAME` を `facebook/detr-resnet-101` に変えて精度を比較する
        """
    )
    return


if __name__ == "__main__":
    app.run()
