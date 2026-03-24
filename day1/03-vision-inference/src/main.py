"""
Object Detection ハンズオン：画像のML推論

DETR（Detection Transformer）を使って画像中の物体を検出します。
推論の3フェーズ（Preprocess → Forward → Postprocess）を順番に実行します。
"""

import torch  # noqa: F401  # テンソル操作の依存ライブラリとして明示
import matplotlib.pyplot as plt

from src.inference import load_model, run_inference
from src.postprocess import postprocess_results, visualize_results
from src.preprocess import get_processor, preprocess_image
from src.utils import build_label_map, download_sample_image

# --- 設定 ---
MODEL_NAME = "facebook/detr-resnet-50"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
CONFIDENCE_THRESHOLD = 0.9


def main() -> None:
    # =============================================================================
    # Phase 1: Preprocess（前処理）
    # =============================================================================
    print("=" * 60)
    print("Phase 1: Preprocess（前処理）")
    print("=" * 60)

    # サンプル画像のダウンロード
    print(f"\n[1/3] サンプル画像をダウンロード中... URL: {IMAGE_URL}")
    sample_image = download_sample_image(IMAGE_URL)
    print(f"      サイズ: {sample_image.width} x {sample_image.height} px")
    print(f"      モード: {sample_image.mode}")

    # 元画像を保存（カレントディレクトリに出力）
    fig_original, ax_original = plt.subplots(figsize=(10, 7))
    ax_original.imshow(sample_image)
    ax_original.axis("off")
    ax_original.set_title("Input Image")
    fig_original.savefig("output_original.png", bbox_inches="tight")
    plt.close(fig_original)
    print("      -> output_original.png に保存しました")

    # Image Processor のロード
    print(f"\n[2/3] Image Processor をロード中... モデル: {MODEL_NAME}")
    processor = get_processor(MODEL_NAME)
    print(f"      Processor クラス: {type(processor).__name__}")
    print(f"      正規化 mean: {processor.image_mean}")
    print(f"      正規化 std:  {processor.image_std}")

    # 前処理の実行
    print("\n[3/3] 前処理（リサイズ・正規化・テンソル変換）を実行中...")
    inputs = preprocess_image(sample_image, processor)
    pixel_values = inputs["pixel_values"]
    print(f"      pixel_values.shape: {tuple(pixel_values.shape)}")
    print(f"      dtype:  {pixel_values.dtype}")
    print(f"      最小値: {pixel_values.min().item():.4f}")
    print(f"      最大値: {pixel_values.max().item():.4f}")
    print(f"      平均値: {pixel_values.mean().item():.4f}")

    # =============================================================================
    # Phase 2: Forward（推論）
    # =============================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Forward（推論）")
    print("=" * 60)

    # モデルのロード
    print(f"\n[1/2] モデルをロード中... モデル: {MODEL_NAME}")
    model = load_model(MODEL_NAME)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"      モデルクラス: {type(model).__name__}")
    print(f"      パラメータ数: {num_params:,}")
    print(f"      クラス数: {model.config.num_labels}（COCO 80クラス）")

    # 推論の実行
    print("\n[2/2] 推論（Forward pass）を実行中...")
    outputs = run_inference(model, inputs)
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes
    print(f"      logits.shape:     {tuple(logits.shape)}  （クラス スコア）")
    print(f"      pred_boxes.shape: {tuple(pred_boxes.shape)}  （bbox: cx,cy,w,h）")

    top5 = sorted(enumerate(logits.softmax(-1)[0, 0].tolist()), key=lambda x: -x[1])[:5]
    print("      先頭クエリの上位5クラス（softmax後）:")
    for cls_id, score in top5:
        print(f"        class_{cls_id}: {score:.3f}")

    # =============================================================================
    # Phase 3: Postprocess（後処理）
    # =============================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Postprocess（後処理）")
    print("=" * 60)

    # 後処理の実行
    print(f"\n[1/2] 後処理を実行中... 閾値: {CONFIDENCE_THRESHOLD}")
    label_map = build_label_map(model)
    detections = postprocess_results(
        outputs,
        processor,
        image_size=(sample_image.width, sample_image.height),
        threshold=CONFIDENCE_THRESHOLD,
    )
    print(f"      検出数: {len(detections)} 個")
    print(f"\n      {'クラス':<15} {'スコア':>6}  bbox (xmin, ymin, xmax, ymax)")
    print("      " + "-" * 55)
    for d in detections:
        class_name = label_map.get(d["label"], str(d["label"]))
        b = d["box"]
        print(
            f"      {class_name:<15} {d['score']:>6.3f}  "
            f"({b['xmin']:.0f}, {b['ymin']:.0f}, {b['xmax']:.0f}, {b['ymax']:.0f})"
        )

    # 結果の可視化（カレントディレクトリに出力）
    print("\n[2/2] 結果を可視化中...")
    result_fig = visualize_results(sample_image, detections, label_names=label_map)
    result_fig.savefig("output_detections.png", bbox_inches="tight")
    plt.close(result_fig)
    print("      -> output_detections.png に保存しました")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
