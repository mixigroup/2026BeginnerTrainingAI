"""
Inference module for object detection with DETR.

Handles model loading and forward pass.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForObjectDetection


def load_model(model_name: str) -> AutoModelForObjectDetection:
    """
    Load the DETR object detection model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name (e.g. 'facebook/detr-resnet-50').

    Returns:
        Pretrained model in eval mode.
    """
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.eval()
    return model


def run_inference(model: AutoModelForObjectDetection, inputs: dict) -> object:
    """
    Run the forward pass (inference) on preprocessed inputs.

    Args:
        model: Loaded DETR model from load_model().
        inputs: Preprocessed inputs dict from preprocess_image().

    Returns:
        Model outputs containing 'logits' (class scores) and 'pred_boxes'
        (bounding box coordinates in cxcywh format, normalized to [0, 1]).
    """
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs
