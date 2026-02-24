"""
Standalone script for DETR object detection.

Runs the full inference pipeline (Preprocess -> Forward -> Postprocess)
without marimo. Saves the visualized result as an image file.

Usage:
    uv run python run.py
    uv run python run.py --image <URL or path> --threshold 0.7 --output result.png
"""

import argparse

from src.preprocess import get_processor, preprocess_image
from src.inference import load_model, run_inference
from src.postprocess import postprocess_results, visualize_results
from src.utils import download_sample_image, build_label_map

DEFAULT_MODEL = "facebook/detr-resnet-50"
DEFAULT_IMAGE = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_THRESHOLD = 0.9
DEFAULT_OUTPUT = "result.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="DETR object detection demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE, help="Image URL or local path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold (0.0 - 1.0)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output image file path"
    )
    args = parser.parse_args()

    # -------------------------------------------------------
    # Phase 1: Preprocess
    # -------------------------------------------------------
    print("[Phase 1] Preprocess")
    print(f"  Loading image: {args.image}")
    image = download_sample_image(args.image)
    print(f"  Image size: {image.width} x {image.height} px")

    print(f"  Loading processor: {args.model}")
    processor = get_processor(args.model)

    inputs = preprocess_image(image, processor)
    pixel_values = inputs["pixel_values"]
    print(f"  pixel_values shape : {tuple(pixel_values.shape)}")
    print(
        f"  pixel_values range : [{pixel_values.min().item():.3f}, {pixel_values.max().item():.3f}]"
    )

    # -------------------------------------------------------
    # Phase 2: Forward
    # -------------------------------------------------------
    print("\n[Phase 2] Forward")
    print(f"  Loading model: {args.model}")
    model = load_model(args.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    print("  Running inference ...")
    outputs = run_inference(model, inputs)
    print(f"  logits shape    : {tuple(outputs.logits.shape)}")
    print(f"  pred_boxes shape: {tuple(outputs.pred_boxes.shape)}")

    # -------------------------------------------------------
    # Phase 3: Postprocess
    # -------------------------------------------------------
    print(f"\n[Phase 3] Postprocess  (threshold={args.threshold})")
    label_map = build_label_map(model)
    detections = postprocess_results(
        outputs,
        processor,
        image_size=(image.width, image.height),
        threshold=args.threshold,
    )
    print(f"  Detected {len(detections)} object(s)")
    for det in detections:
        name = label_map.get(det["label"], str(det["label"]))
        b = det["box"]
        print(
            f"    {name}: score={det['score']:.3f}  "
            f"box=({b['xmin']:.0f}, {b['ymin']:.0f}, {b['xmax']:.0f}, {b['ymax']:.0f})"
        )

    # -------------------------------------------------------
    # Save result image
    # -------------------------------------------------------
    print(f"\n[Result] Saving visualization -> {args.output}")
    fig = visualize_results(image, detections, label_names=label_map)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print("Done.")


if __name__ == "__main__":
    main()
