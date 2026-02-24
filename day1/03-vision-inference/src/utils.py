"""
Utility functions for the vision inference handson.

Provides helpers for downloading sample images and retrieving label names.
"""

from __future__ import annotations

from io import BytesIO

import requests
from PIL import Image


def download_sample_image(url: str) -> Image.Image:
    """
    Download an image from the given URL and return it as a PIL Image.

    Args:
        url: HTTP/HTTPS URL pointing to an image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        requests.HTTPError: If the download fails.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def get_label_name(label_id: int, model: object) -> str:
    """
    Return the human-readable class name for a given label id.

    Args:
        label_id: Integer class id from model output.
        model: Loaded HuggingFace model that has a config.id2label mapping.

    Returns:
        Class name string (e.g. 'cat', 'dog').
        Falls back to str(label_id) if the mapping is unavailable.
    """
    try:
        return model.config.id2label[label_id]
    except (AttributeError, KeyError):
        return str(label_id)


def build_label_map(model: object) -> dict[int, str]:
    """
    Build a complete label id -> name mapping from the model config.

    Args:
        model: Loaded HuggingFace model with config.id2label.

    Returns:
        Dictionary mapping int label ids to class name strings.
    """
    try:
        return dict(model.config.id2label)
    except AttributeError:
        return {}
