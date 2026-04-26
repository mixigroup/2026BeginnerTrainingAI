"""
Evaluation module for object detection on COCO val2017.

Downloads COCO val2017 annotations + images on demand (cached locally),
runs inference, and computes COCO-style mAP / AP50 / AP75 via pycocotools.
"""

from __future__ import annotations

import contextlib
import io
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

COCO_VAL2017_IMG_BASE = "http://images.cocodataset.org/val2017"
COCO_ANN_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_ANN_ZIP_MEMBER = "annotations/instances_val2017.json"


def download_coco_annotations(cache_dir: Path) -> Path:
    """
    Download and cache the COCO val2017 instance annotations json.

    The official COCO archive bundles train + val into a single ~240MB zip;
    this function downloads it once, extracts only the val2017 json (~20MB)
    into ``cache_dir``, and returns its path. Subsequent calls are no-ops.

    Args:
        cache_dir: Directory used to cache the extracted json.

    Returns:
        Path to ``instances_val2017.json``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ann_path = cache_dir / "instances_val2017.json"
    if ann_path.exists():
        return ann_path

    response = requests.get(COCO_ANN_ZIP_URL, timeout=600)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with zf.open(COCO_ANN_ZIP_MEMBER) as src:
            ann_path.write_bytes(src.read())

    return ann_path


def download_val2017_image(file_name: str, cache_dir: Path) -> Image.Image:
    """
    Download a single COCO val2017 image (cached) and return it as RGB.

    Args:
        file_name: e.g. ``"000000039769.jpg"``.
        cache_dir: Directory used to cache downloaded images.

    Returns:
        PIL Image in RGB mode.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / file_name
    if not local.exists():
        url = f"{COCO_VAL2017_IMG_BASE}/{file_name}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        local.write_bytes(response.content)
    return Image.open(local).convert("RGB")


def _resolve_device(model: AutoModelForObjectDetection) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return next(model.parameters()).device


def evaluate_on_coco_val2017(
    model: AutoModelForObjectDetection,
    processor: AutoImageProcessor,
    ann_path: Path,
    image_cache_dir: Path,
    num_images: int = 50,
    score_threshold: float = 0.0,
    progress_wrapper: Callable[[Iterable], Iterable] | None = None,
) -> dict:
    """
    Run inference on a subset of COCO val2017 and compute COCO-style metrics.

    Args:
        model: Loaded DETR model.
        processor: Matching ``AutoImageProcessor``.
        ann_path: Path to ``instances_val2017.json`` (use
            :func:`download_coco_annotations`).
        image_cache_dir: Directory to cache downloaded val2017 images.
        num_images: Evaluate on the first ``num_images`` image ids
            (sorted ascending). Use a small number for a quick sanity check;
            5000 = full val2017.
        score_threshold: Confidence floor passed to
            ``post_process_object_detection``. Keep low (0.0–0.05) so the
            AP curve has enough recall points.
        progress_wrapper: Optional callable that wraps an iterable with a
            progress bar, e.g. ``mo.status.progress_bar`` or ``tqdm``.

    Returns:
        Dict with keys:
          - ``map``  : mAP@[.50:.95]
          - ``map50``: AP at IoU=0.50
          - ``map75``: AP at IoU=0.75
          - ``num_images``     : number of images evaluated
          - ``num_predictions``: total predicted boxes across the subset
          - ``summary``        : full pycocotools summary text
          - ``precision``      : ndarray ``[T, R, K, A, M]`` from COCOeval
            (T=IoU thresholds, R=recall points, K=classes, A=area ranges,
            M=max detections). Used to draw PR curves.
          - ``recall_thresholds``: ndarray of recall points (length R)
          - ``iou_thresholds``   : ndarray of IoU thresholds (length T)
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ann_path))

    img_ids = sorted(coco_gt.getImgIds())[:num_images]
    device = _resolve_device(model)
    model.to(device)
    model.eval()

    iterable: Iterable = (
        progress_wrapper(img_ids) if progress_wrapper is not None else img_ids
    )

    predictions: list[dict] = []
    for img_id in iterable:
        info = coco_gt.loadImgs(img_id)[0]
        image = download_val2017_image(info["file_name"], image_cache_dir)

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )[0]

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            predictions.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(label.item()),
                    # COCO expects [x, y, w, h]
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item()),
                }
            )

    if not predictions:
        return {
            "map": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "num_images": len(img_ids),
            "num_predictions": 0,
            "summary": "(no predictions above threshold)",
            "precision": None,
            "recall_thresholds": None,
            "iou_thresholds": None,
        }

    summary_buf = io.StringIO()
    with contextlib.redirect_stdout(summary_buf):
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return {
        "map": float(coco_eval.stats[0]),
        "map50": float(coco_eval.stats[1]),
        "map75": float(coco_eval.stats[2]),
        "num_images": len(img_ids),
        "num_predictions": len(predictions),
        "summary": summary_buf.getvalue(),
        "precision": coco_eval.eval["precision"],
        "recall_thresholds": coco_eval.params.recThrs,
        "iou_thresholds": coco_eval.params.iouThrs,
    }
