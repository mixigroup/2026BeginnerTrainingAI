"""
SAM (Segment Anything Model) の推論モジュール。

GCS からモデルをダウンロードし、PyTorch で推論を行う。
エンコーダ + デコーダのフルパイプラインを実行し、セグメンテーションマスクを返す。
"""

import base64
import io
import os
from urllib.parse import urlparse

import numpy as np
import torch
from google.cloud import storage
from PIL import Image
from transformers import SamModel, SamProcessor


def download_model(gcs_uri: str, local_dir: str) -> None:
    """GCS からモデルディレクトリをダウンロードする。

    Args:
        gcs_uri: GCS URI (gs://bucket/path/to/model-dir/)
        local_dir: ローカルの保存先ディレクトリ
    """
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Expected a gs:// URI, got: {gcs_uri}")

    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    print(f"Downloading model from gs://{bucket_name}/{prefix} → {local_dir}")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    os.makedirs(local_dir, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        # prefix 以降の相対パスを取得
        relative_path = blob.name[len(prefix) :]
        if not relative_path:
            continue

        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"  Downloaded: {relative_path}")

    print("Download complete.")


class SAMPredictor:
    """SAM モデルのラッパー。PyTorch でフルパイプライン推論を行う。"""

    def __init__(self, model_dir: str) -> None:
        """モデルをロードする。

        Args:
            model_dir: save_pretrained() で保存したモデルディレクトリのパス。
        """
        self.model = SamModel.from_pretrained(model_dir)
        self.processor = SamProcessor.from_pretrained(model_dir)
        self.model.eval()
        print(f"SAM model loaded from {model_dir}")

    def predict(
        self,
        image: Image.Image,
        input_points: list[list[int]],
        input_labels: list[int],
    ) -> dict:
        """画像とポイント座標からセグメンテーションマスクを生成する。

        Args:
            image: 入力画像 (PIL Image)。
            input_points: ポイント座標のリスト [[x, y], ...]。
            input_labels: 各ポイントのラベル (1=前景, 0=背景)。

        Returns:
            {"mask_b64": "<base64 PNG>", "iou_score": float} 形式の辞書。
        """
        # 前処理
        inputs = self.processor(
            images=image,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        )

        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)

        # マスクの後処理
        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )

        # IoU スコアが最も高いマスクを選択
        iou_scores = outputs.iou_scores[0, 0]
        best_idx = iou_scores.argmax().item()
        mask = masks[0][0, best_idx].numpy()
        best_iou = float(iou_scores[best_idx].item())

        # マスクを PNG 画像として base64 エンコード
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode="L")
        buf = io.BytesIO()
        mask_image.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"mask_b64": mask_b64, "iou_score": best_iou}
