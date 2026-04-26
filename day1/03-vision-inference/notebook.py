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
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    from src.preprocess import get_processor, preprocess_image
    from src.inference import load_model, run_inference
    from src.postprocess import postprocess_results, visualize_results
    from src.utils import download_sample_image, build_label_map

    return (
        build_label_map,
        download_sample_image,
        get_processor,
        load_model,
        plt,
        postprocess_results,
        preprocess_image,
        run_inference,
        visualize_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Phase 1: Preprocess（前処理）

    生の画像データをモデルが受け付ける形式に変換します。

    ### 前処理で行うこと

    1. **画像の読み込み** - URL またはローカルパスから PIL Image として読み込む
    2. **リサイズ** - モデルが期待するサイズに変換（DETR は短辺 800px を基準）
    3. **正規化** - ピクセル値を ImageNet の mean/std で標準化
    4. **テンソル変換** - NumPy/PIL → PyTorch Tensor（形状: `[1, 3, H, W]`）
    """)
    return


@app.cell
def _():
    # --- Settings: Edit these to experiment! ---
    MODEL_NAME = "facebook/detr-resnet-50"
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return IMAGE_URL, MODEL_NAME


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
    return


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
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(MODEL_NAME, load_model):
    model = load_model(MODEL_NAME)
    return (model,)


@app.cell
def _(MODEL_NAME, mo, model):

    mo.md(
        f"""
        ### モデルのロード

        - モデル名: `{MODEL_NAME}`
        - モデルクラス: `{type(model).__name__}`
        - パラメータ数: **{sum(p.numel() for p in model.parameters()):,}**
        - クラス数: **{model.config.num_labels}**（COCO 80クラス）
        - クラス: {model.config.id2label.values()}
        """
    )
    return


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
    return (outputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Phase 3: Postprocess（後処理）

    生の出力 tensor から、人間が理解できる検出結果に変換します。

    ### 後処理で行うこと

    1. **Softmax** - logits を確率に変換
    2. **閾値フィルタリング** - confidence スコアが閾値以上の検出のみ残す
    3. **座標変換** - 正規化 [0, 1] の相対座標 → 画像上のピクセル座標
    4. **ラベル変換** - クラスid → クラス名（"cat"、"dog" など）
    """)
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
    processor,
    sample_image,
):
    label_map = build_label_map(model)
    detections = postprocess_results(
        outputs,
        processor,
        image_size=(sample_image.width, sample_image.height),
        threshold=CONFIDENCE_THRESHOLD,
    )

    rows = "\n   ".join(
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
    return detections, label_map


@app.cell
def _(detections, label_map, sample_image, visualize_results):
    result_fig = visualize_results(sample_image, detections, label_names=label_map)
    result_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 評価（COCO val2017 で精度を測る）

    1 枚の画像で動作確認しただけでは、モデルの精度はわかりません。
    **COCO val2017** の一部を使って **mAP** を計測します。

    ### mAP が出るまで（3 ステップ）

    **① IoU で「正解 / 不正解」を判定 → ② Precision/Recall から AP を計算 → ③ AP を平均して mAP**

    #### ① IoU で TP / FP / FN を分類

    予測 bbox と正解 bbox の **IoU（重なり度合い, 0〜1）** を計算し、
    **IoU 閾値（例: 0.5）** 以上 & クラス一致なら「当たり」とみなします。

    ![](imgs/iou-explanation.svg)

    - **TP**（正解）: IoU ≥ 閾値 かつ クラス一致
    - **FP**（誤検出）: IoU < 閾値 または クラス不一致 の予測
    - **FN**（見逃し）: 対応する予測がない正解 bbox

    #### ② AP（Average Precision）= PR 曲線の下の面積

    - **Precision = TP / (TP + FP)** … 予測した中で当たりの割合（誤検出の少なさ）
    - **Recall = TP / (TP + FN)** … 正解のうち見つけられた割合（見逃しの少なさ）
    - confidence 閾値を高 → 低 へ動かすと PR 曲線が描ける。**その下の面積が AP**（1 クラスにつき 1 つ）

    ##### 例: confidence を下げながら Precision/Recall を計算する

    画像内に正解 bbox が **5 個** あり、モデルが **10 個の予測** を返したとします。
    予測を **score の高い順** に並べ、上から **累積** で TP/FP を足していくと、
    各行でその時点の Precision と Recall が求まります
    （= confidence 閾値を上から下へ下げていく操作と等価）。

    | 順位 | score | 判定 | 累積TP | 累積FP | Precision = 累積TP/(累積TP+累積FP) | Recall = 累積TP/5 |
    |---|---|---|---|---|---|---|
    | 1 | 0.95 | TP | 1 | 0 | 1.00 | 0.20 |
    | 2 | 0.91 | TP | 2 | 0 | 1.00 | 0.40 |
    | 3 | 0.85 | FP | 2 | 1 | 0.67 | 0.40 |
    | 4 | 0.80 | TP | 3 | 1 | 0.75 | 0.60 |
    | 5 | 0.72 | FP | 3 | 2 | 0.60 | 0.60 |
    | 6 | 0.65 | TP | 4 | 2 | 0.67 | 0.80 |
    | 7 | 0.50 | FP | 4 | 3 | 0.57 | 0.80 |
    | 8 | 0.42 | TP | 5 | 3 | 0.62 | 1.00 |
    | 9 | 0.30 | FP | 5 | 4 | 0.56 | 1.00 |
    | 10 | 0.15 | FP | 5 | 5 | 0.50 | 1.00 |

    各行の **(Recall, Precision)** を点として打ち、左から右へ結んだのが **PR 曲線**。
    その下の面積を計算した値が **AP** です。上の表を実際にプロットすると、

    ![](imgs/pr-curve-sample.svg)

    水色の塗り面積（≈ 0.81）が AP。赤点は表の各行（順位 #1〜#10）に対応しています。

    #### ③ mAP = AP の平均

    複数の AP を平均した値が **mAP**（mean Average Precision）。
    「何で平均するか」は指標によって違います（次節）：

    - 80 クラスの AP を平均 → クラス方向の平均
    - 10 段階の IoU 閾値での AP を平均 → IoU 方向の平均
    - その両方を平均 → COCO の `mAP@[.50:.95]`

    ### AP50 / AP75 / mAP@[.50:.95] の違い

    **どれも「mAP」**。違いは **① で使う IoU 閾値** だけです。

    | 指標 | 使う IoU 閾値 | 何を見ているか |
    |---|---|---|
    | **AP50** | 0.50（1 個）| bbox がだいたい当たれば OK。**検出漏れ** を見る |
    | **AP75** | 0.75（1 個）| bbox がきっちり当たる必要あり。**位置精度** を見る |
    | **mAP@[.50:.95]** | 0.50, 0.55, …, 0.95（**10 個の平均**）| **COCO 公式の総合スコア**。論文で「mAP」といえばこれ |

    → **AP50 → AP75 → mAP@[.50:.95]** の順に基準が厳しくなり、スコアは下がります。

    ### このノートブックでやること

    1. `instances_val2017.json`（正解 bbox とラベル）をダウンロード
    2. 各画像で推論し、予測 bbox + クラス + スコアを集める
    3. `pycocotools` で AP50 / AP75 / mAP@[.50:.95] を一括計算

    > val2017 は 5000 枚あります。フル評価は時間がかかるためデフォルトでは
    > 先頭 50 枚に絞っています（`NUM_EVAL_IMAGES` で変更可能）。
    """)
    return


@app.cell
def _():
    # --- Eval settings: increase NUM_EVAL_IMAGES for a more reliable score ---
    NUM_EVAL_IMAGES = 50
    EVAL_SCORE_THRESHOLD = 0.0
    return EVAL_SCORE_THRESHOLD, NUM_EVAL_IMAGES


@app.cell
def _(EVAL_SCORE_THRESHOLD, NUM_EVAL_IMAGES, mo, model, processor):
    from pathlib import Path

    from src.evaluate import (
        download_coco_annotations,
        evaluate_on_coco_val2017,
    )

    _cache_root = Path(".cache/coco")
    _ann_path = download_coco_annotations(_cache_root)

    metrics = evaluate_on_coco_val2017(
        model,
        processor,
        ann_path=_ann_path,
        image_cache_dir=_cache_root / "val2017",
        num_images=NUM_EVAL_IMAGES,
        score_threshold=EVAL_SCORE_THRESHOLD,
        progress_wrapper=mo.status.progress_bar,
    )

    mo.md(
        f"""
        ### 評価結果（val2017 先頭 {metrics["num_images"]} 枚）

        | 指標 | スコア | 意味 |
        |---|---|---|
        | **mAP@[.50:.95]** | `{metrics["map"]:.3f}` | IoU を 0.50→0.95 で 10 段階平均（COCO 標準）|
        | **AP50** | `{metrics["map50"]:.3f}` | IoU ≥ 0.50（検出漏れを見る）|
        | **AP75** | `{metrics["map75"]:.3f}` | IoU ≥ 0.75（位置精度を見る）|

        - 予測 bbox 総数: **{metrics["num_predictions"]}**
        - 参考: 公式 DETR-ResNet-50 の val2017 全件 mAP は **約 0.42**

        > サブセットなのでスコアは公式値とぶれます。`NUM_EVAL_IMAGES` を増やすほど
        > 公式値に近づきます（その分、推論時間も増えます）。
        """
    )
    return (metrics,)


@app.cell
def _(metrics, mo):
    mo.md(f"pycocotools の生サマリーを表示\n```\n{metrics['summary']}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PR 曲線を描いてみる

    AP の正体は **PR 曲線の下の面積** でした。実際の曲線を見れば、
    「IoU 閾値を厳しくすると曲線が下にシフトして面積（=AP）が小さくなる」
    という関係が一目でわかります。

    - 右に行くほど **Recall**（見逃しが少ない）
    - 上に行くほど **Precision**（誤検出が少ない）
    - 全クラスの平均をプロット（pycocotools 内部の `precision[T, R, K, A, M]` を利用）
    """)
    return


@app.cell
def _(metrics, plt):
    import numpy as np

    _precision = metrics["precision"]  # [T, R, K, A, M]
    _recall = metrics["recall_thresholds"]  # 101 点 (0.0, 0.01, ..., 1.0)

    # area=all (idx 0), maxDets=100 (idx 2) でクラス平均
    def _mean_pr(iou_idx):
        p = _precision[iou_idx, :, :, 0, 2]  # [R, K]
        p = np.where(p == -1, np.nan, p)  # GT がないクラスは除外
        return np.nanmean(p, axis=1)

    pr_fig, pr_ax = plt.subplots(figsize=(8, 6))
    for _label, _iou_idx, _color in [
        ("IoU=0.50 (AP50)", 0, "tab:blue"),
        ("IoU=0.75 (AP75)", 5, "tab:orange"),
    ]:
        _p = _mean_pr(_iou_idx)
        _ap = float(np.nanmean(_p))
        pr_ax.plot(
            _recall, _p, label=f"{_label}, AP={_ap:.3f}", color=_color, linewidth=2
        )
        pr_ax.fill_between(_recall, 0, _p, color=_color, alpha=0.15)

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.05)
    pr_ax.set_title("PR Curve (mean over classes, val2017 subset)")
    pr_ax.legend(loc="lower left")
    pr_ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    このハンズオンで体験した3フェーズをふり返ります。

    | フェーズ | 処理 | 入出力 |
    |---|---|---|
    | **Preprocess** | 画像 → tensor | `PIL.Image` → `{{pixel_values: Tensor[1,3,H,W]}}` |
    | **Forward** | tensor → 生出力 | `{{pixel_values}}` → `{{logits: [1,100,92], pred_boxes: [1,100,4]}}` |
    | **Postprocess** | 生出力 → 検出結果 | `logits + pred_boxes` → `[{{label, score, box}}, ...]` |

    ### 発展課題

    - `CONFIDENCE_THRESHOLD` を変えて検出数の変化を確認する
    - `IMAGE_URL` を別の画像に変えて試す
    - `MODEL_NAME` を `facebook/detr-resnet-101` に変えて精度を比較する
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
