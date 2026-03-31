"""
YOLOv8 物体検出の推論モジュール。

GCS から ONNX モデルをダウンロードし、ONNX Runtime で推論を行う。
"""

from urllib.parse import urlparse

import os

import cv2
import numpy as np
import onnxruntime as ort
from google.cloud import storage


def download_model(gcs_uri: str, local_path: str) -> None:
    """Download model file from GCS to local filesystem.

    Args:
        gcs_uri: GCS URI in the form gs://bucket/path/to/model.onnx
        local_path: Destination path on the local filesystem.
    """
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Expected a gs:// URI, got: {gcs_uri}")

    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    print(f"Downloading model from gs://{bucket_name}/{blob_path} → {local_path}")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print("Download complete.")


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess a raw image for YOLO inference.

    Steps:
    1. Resize to 640x640 (letterbox-free, simple resize)
    2. BGR -> RGB channel swap
    3. HWC -> CHW layout
    4. Normalize to [0, 1]
    5. Add batch dimension

    Args:
        image: H x W x 3 uint8 numpy array (BGR, as returned by cv2).

    Returns:
        1 x 3 x 640 x 640 float32 numpy array.
    """
    resized = cv2.resize(image, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)  # HWC -> CHW
    normalized = chw.astype(np.float32) / 255.0
    batched = np.expand_dims(normalized, axis=0)  # 1 x C x H x W
    return batched


def _postprocess(
    output: np.ndarray,
    orig_h: int,
    orig_w: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """YOLOv8 の生出力を検出結果に変換する。

    YOLOv8 の出力形状: (1, 84, 8400)
    84 = 4 (bbox: cx, cy, w, h) + 80 (COCO クラススコア)

    Args:
        output: ONNX の生出力配列。
        orig_h: リサイズ前の元画像の高さ。
        orig_w: リサイズ前の元画像の幅。
        conf_threshold: 検出を残す最小信頼度スコア。
        iou_threshold: NMS の IoU 閾値。

    Returns:
        検出結果の辞書リスト。各辞書のキー:
            - bbox: [x1, y1, x2, y2] 元画像座標
            - score: float 信頼度
            - class_id: int クラスインデックス
    """
    # output: (1, 84, 8400)
    pred = output[0].T  # (8400, 84)

    # 最初の4値: cx, cy, w, h（640x640 空間）
    boxes_xywh = pred[:, :4]

    # 列 4..84 はクラスごとの確率
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(pred)), class_ids]

    # --- 信頼度でフィルタ ---
    keep_mask = scores >= conf_threshold
    boxes_xywh = boxes_xywh[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    if len(scores) == 0:
        return []

    # --- cx,cy,w,h → x1,y1,x2,y2 (640x640 空間) ---
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # --- NMS (greedy) ---
    order = np.argsort(-scores)
    selected = []
    while len(order) > 0:
        idx = order[0]
        selected.append(idx)
        if len(order) == 1:
            break
        rest = order[1:]
        inter_x1 = np.maximum(boxes_xyxy[idx, 0], boxes_xyxy[rest, 0])
        inter_y1 = np.maximum(boxes_xyxy[idx, 1], boxes_xyxy[rest, 1])
        inter_x2 = np.minimum(boxes_xyxy[idx, 2], boxes_xyxy[rest, 2])
        inter_y2 = np.minimum(boxes_xyxy[idx, 3], boxes_xyxy[rest, 3])
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(
            0, inter_y2 - inter_y1
        )
        area_idx = (boxes_xyxy[idx, 2] - boxes_xyxy[idx, 0]) * (
            boxes_xyxy[idx, 3] - boxes_xyxy[idx, 1]
        )
        area_rest = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (
            boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1]
        )
        iou = inter_area / (area_idx + area_rest - inter_area + 1e-6)
        order = rest[iou <= iou_threshold]

    # --- 元画像サイズにスケール ---
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    results = []
    for i in selected:
        results.append({
            "bbox": [
                float(boxes_xyxy[i, 0] * scale_x),
                float(boxes_xyxy[i, 1] * scale_y),
                float(boxes_xyxy[i, 2] * scale_x),
                float(boxes_xyxy[i, 3] * scale_y),
            ],
            "score": float(scores[i]),
            "class_id": int(class_ids[i]),
        })

    return results


class YOLOPredictor:
    """Wrapper around an ONNX YOLO model for inference."""

    def __init__(self, model_path: str) -> None:
        """Load the ONNX model.

        Args:
            model_path: Local path to the .onnx file.
        """
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model loaded: input='{self.input_name}'")

    def predict(self, image: np.ndarray) -> dict:
        """Run YOLO inference on a single image.

        Args:
            image: H x W x 3 uint8 numpy array (BGR).

        Returns:
            Dict with key 'detections': list of detection dicts.
        """
        orig_h, orig_w = image.shape[:2]
        input_tensor = _preprocess(image)

        raw_outputs = self.session.run(None, {self.input_name: input_tensor})
        output = raw_outputs[0]  # shape: (1, channels, anchors)

        detections = _postprocess(output, orig_h, orig_w)
        return {"detections": detections}
