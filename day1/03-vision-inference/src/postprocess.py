"""
Postprocessing module for object detection with DETR.

Handles converting raw model outputs into human-readable detection results
and visualizing bounding boxes on images.
"""

from __future__ import annotations

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor


def postprocess_results(
    outputs: object,
    processor: AutoImageProcessor,
    image_size: tuple[int, int],
    threshold: float = 0.9,
) -> list[dict]:
    """
    Convert raw model outputs into a list of detected objects.

    Args:
        outputs: Model outputs from run_inference().
        processor: AutoImageProcessor used during preprocessing.
        image_size: (width, height) of the original image.
        threshold: Confidence score threshold. Only detections above this
                   value are returned.

    Returns:
        List of detection dicts, each containing:
          - 'label': int class id
          - 'score': float confidence score
          - 'box': dict with keys 'xmin', 'ymin', 'xmax', 'ymax' (pixels)
    """
    # processor expects (height, width) as a Tensor
    height, width = image_size[1], image_size[0]
    target_sizes = torch.tensor([[height, width]])
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box_dict = {
            "xmin": box[0].item(),
            "ymin": box[1].item(),
            "xmax": box[2].item(),
            "ymax": box[3].item(),
        }
        detections.append(
            {
                "label": label.item(),
                "score": score.item(),
                "box": box_dict,
            }
        )

    return detections


def visualize_results(
    image: Image.Image,
    detections: list[dict],
    label_names: dict[int, str] | None = None,
) -> plt.Figure:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: Original PIL Image.
        detections: Detection list from postprocess_results().
        label_names: Optional mapping from label id to class name string.
                     When None, numeric ids are used.

    Returns:
        Matplotlib Figure with bounding boxes and confidence scores drawn.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = matplotlib.colormaps["tab20"].colors  # up to 20 distinct colors

    for i, det in enumerate(detections):
        box = det["box"]
        label_id = det["label"]
        score = det["score"]

        x = box["xmin"]
        y = box["ymin"]
        w = box["xmax"] - box["xmin"]
        h = box["ymax"] - box["ymin"]

        color = colors[label_id % len(colors)]

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = (
            label_names[label_id]
            if label_names and label_id in label_names
            else str(label_id)
        )
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x,
            y - 5,
            label_text,
            color="white",
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
        )

    ax.axis("off")
    ax.set_title(f"Object Detection Results ({len(detections)} objects detected)")
    plt.tight_layout()

    return fig
