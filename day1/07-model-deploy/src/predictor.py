"""
SAM (Segment Anything Model) の推論モジュール。

GCS からモデルをダウンロードし、PyTorch で推論を行う。
エンコーダ + デコーダのフルパイプラインを実行し、セグメンテーションマスクを返す。
"""

import base64
import io
import os
import time
from urllib.parse import urlparse

import numpy as np
import torch
from google.cloud import storage
from loguru import logger
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

    logger.info(f"GCS download: gs://{bucket_name}/{prefix} → {local_dir}")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    logger.debug(f"GOOGLE_CLOUD_PROJECT = {project!r}")
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    os.makedirs(local_dir, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=prefix))
    logger.info(f"Found {len(blobs)} blob(s) in GCS")

    for i, blob in enumerate(blobs):
        relative_path = blob.name[len(prefix) :]
        if not relative_path:
            continue

        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        t0 = time.perf_counter()
        blob.download_to_filename(local_path)
        elapsed = time.perf_counter() - t0
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(
            f"  [{i + 1}/{len(blobs)}] {relative_path} ({size_mb:.1f} MB, {elapsed:.1f}s)"
        )

    logger.info("GCS download complete")


class SAMPredictor:
    """SAM モデルのラッパー。PyTorch でフルパイプライン推論を行う。"""

    def __init__(self, model_dir: str) -> None:
        """モデルをロードする。

        Args:
            model_dir: save_pretrained() で保存したモデルディレクトリのパス。
        """
        logger.info(f"Loading SamModel from {model_dir} ...")
        t0 = time.perf_counter()
        self.model = SamModel.from_pretrained(model_dir)
        elapsed_model = time.perf_counter() - t0
        logger.info(f"SamModel loaded ({elapsed_model:.1f}s)")

        logger.info(f"Loading SamProcessor from {model_dir} ...")
        t1 = time.perf_counter()
        self.processor = SamProcessor.from_pretrained(model_dir)
        elapsed_proc = time.perf_counter() - t1
        logger.info(f"SamProcessor loaded ({elapsed_proc:.1f}s)")

        self.model.eval()

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"SAM model ready — {num_params:,} parameters")

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
        logger.debug(
            f"predict() called: image={image.width}x{image.height}, "
            f"points={input_points}, labels={input_labels}"
        )

        # 前処理
        t0 = time.perf_counter()
        inputs = self.processor(
            images=image,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        )
        elapsed_pre = time.perf_counter() - t0
        logger.debug(
            f"Preprocess: {elapsed_pre:.3f}s — input keys: {list(inputs.keys())}"
        )

        # 推論
        t1 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        elapsed_fwd = time.perf_counter() - t1
        logger.debug(
            f"Forward: {elapsed_fwd:.3f}s — "
            f"pred_masks shape={list(outputs.pred_masks.shape)}, "
            f"iou_scores shape={list(outputs.iou_scores.shape)}"
        )

        # マスクの後処理
        t2 = time.perf_counter()
        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        elapsed_post = time.perf_counter() - t2
        logger.debug(f"Post-process masks: {elapsed_post:.3f}s")

        # IoU スコアが最も高いマスクを選択
        iou_scores = outputs.iou_scores[0, 0]
        best_idx = iou_scores.argmax().item()
        mask = masks[0][0, best_idx].numpy()
        best_iou = float(iou_scores[best_idx].item())
        logger.debug(
            f"Best mask: index={best_idx}, iou={best_iou:.4f}, "
            f"all_iou_scores={[f'{s:.4f}' for s in iou_scores.tolist()]}"
        )

        # マスクを PNG 画像として base64 エンコード
        t3 = time.perf_counter()
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode="L")
        buf = io.BytesIO()
        mask_image.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        elapsed_enc = time.perf_counter() - t3
        mask_size_kb = len(buf.getvalue()) / 1024
        logger.debug(
            f"Mask encode: {elapsed_enc:.3f}s — "
            f"mask shape={mask.shape}, png size={mask_size_kb:.1f} KB"
        )

        total = time.perf_counter() - t0
        logger.info(
            f"predict() total: {total:.3f}s "
            f"(pre={elapsed_pre:.3f} fwd={elapsed_fwd:.3f} post={elapsed_post:.3f} enc={elapsed_enc:.3f})"
        )

        return {"mask_b64": mask_b64, "iou_score": best_iou}
