"""
Preprocessing module for object detection with DETR.

Handles image loading and preprocessing for the DETR model.
"""

from __future__ import annotations

from PIL import Image
from transformers import AutoImageProcessor


def load_image(path_or_url: str) -> Image.Image:
    """
    Load an image from a local path or URL.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL.

    Returns:
        PIL Image in RGB mode.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests
        from io import BytesIO

        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(path_or_url).convert("RGB")

    return image


def get_processor(model_name: str) -> AutoImageProcessor:
    """
    Load the image processor (feature extractor) for the given model.

    Args:
        model_name: HuggingFace model name (e.g. 'facebook/detr-resnet-50').

    Returns:
        AutoImageProcessor instance.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    return processor


def preprocess_image(image: Image.Image, processor: AutoImageProcessor) -> dict:
    """
    Preprocess an image into model inputs.

    Applies resizing, normalization, and tensor conversion as required by
    the DETR processor.

    Args:
        image: PIL Image to preprocess.
        processor: AutoImageProcessor from get_processor().

    Returns:
        Dictionary with 'pixel_values' and other tensors required by the model.
    """
    inputs = processor(images=image, return_tensors="pt")
    return inputs
