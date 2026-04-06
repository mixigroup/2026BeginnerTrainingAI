"""
FastAPI inference server for SAM (Segment Anything Model).

The model is downloaded from GCS at startup (not baked into the container).

Vertex AI custom container requirements:
  - Health check:  GET  /health  → 200 OK
  - Prediction:    POST /predict → JSON response
  - Port: 8080

Vertex AI sends prediction requests in JSON format:
  {
    "instances": [
      {
        "image": "<base64-encoded image bytes>",
        "input_points": [[x1, y1], [x2, y2]],
        "input_labels": [1, 0]
      },
      ...
    ]
  }

And expects a response in the format:
  {
    "predictions": [
      {"mask_b64": "<base64 PNG>", "iou_score": 0.95},
      ...
    ]
  }

Environment variables:
  AIP_STORAGE_URI - (Vertex AI が自動設定) モデルアーティファクトのコピー先 GCS URI。
                    artifact_uri を指定して Model を登録すると自動で設定される。
  MODEL_GCS_URI   - (ローカルテスト用フォールバック) GCS URI of the model directory, e.g.
                    gs://my-bucket/models/alice/sam-model/
"""

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from src.predictor import SAMPredictor, download_model

# loguru の設定: uvicorn のログと区別しやすいフォーマット
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "")
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "")  # ローカルテスト用フォールバック
LOCAL_MODEL_DIR = "/tmp/sam-model"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

logger.info("=== SAM Inference Server ===")
logger.info(f"AIP_STORAGE_URI = {AIP_STORAGE_URI!r}")
logger.info(f"MODEL_GCS_URI   = {MODEL_GCS_URI!r}")
logger.info(f"LOCAL_MODEL_DIR = {LOCAL_MODEL_DIR}")
logger.info(f"MAX_IMAGE_BYTES = {MAX_IMAGE_BYTES}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class Instance(BaseModel):
    """A single prediction instance sent by Vertex AI."""

    image: str  # base64-encoded image bytes
    input_points: list[list[int]]  # [[x, y], ...]
    input_labels: list[int]  # [1, 0, ...] (1=foreground, 0=background)


class PredictRequest(BaseModel):
    """Vertex AI prediction request format."""

    instances: list[Instance]


# ---------------------------------------------------------------------------
# App lifecycle: download model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download the model from GCS once at startup, then serve requests.

    Vertex AI 上では AIP_STORAGE_URI（Vertex AI 管理バケットのコピー）から、
    ローカルテスト時は MODEL_GCS_URI からモデルをダウンロードする。
    """
    logger.info("--- Lifespan startup begin ---")

    if AIP_STORAGE_URI:
        gcs_uri = AIP_STORAGE_URI
        logger.info(f"Using AIP_STORAGE_URI: {gcs_uri}")
    elif MODEL_GCS_URI:
        gcs_uri = MODEL_GCS_URI
        logger.info(f"Using MODEL_GCS_URI (local fallback): {gcs_uri}")
    else:
        logger.error("Neither AIP_STORAGE_URI nor MODEL_GCS_URI is set!")
        raise RuntimeError(
            "Neither AIP_STORAGE_URI nor MODEL_GCS_URI is set. "
            "On Vertex AI, AIP_STORAGE_URI is set automatically via artifact_uri. "
            "For local testing, pass -e MODEL_GCS_URI=gs://bucket/path/to/sam-model/"
        )

    logger.info(f"Downloading model from {gcs_uri} to {LOCAL_MODEL_DIR} ...")
    t0 = time.perf_counter()
    download_model(gcs_uri, LOCAL_MODEL_DIR)
    elapsed_dl = time.perf_counter() - t0
    logger.info(f"Model download complete ({elapsed_dl:.1f}s)")

    logger.info("Loading SAM model into memory ...")
    t1 = time.perf_counter()
    app.state.predictor = SAMPredictor(LOCAL_MODEL_DIR)
    elapsed_load = time.perf_counter() - t1
    logger.info(f"SAM model loaded ({elapsed_load:.1f}s)")

    logger.info("--- Lifespan startup complete — ready to serve ---")
    yield
    logger.info("--- Lifespan shutdown ---")


app = FastAPI(title="SAM Inference Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Health check endpoint required by Vertex AI.

    Returns 503 until the model is fully loaded so that Vertex AI waits
    before routing traffic to this instance.
    """
    if not hasattr(app.state, "predictor"):
        logger.warning("Health check: model not ready yet (503)")
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    logger.debug("Health check: ok")
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: PredictRequest) -> JSONResponse:
    """Run SAM inference on a batch of images."""
    num_instances = len(request.instances)
    logger.info(f"POST /predict — {num_instances} instance(s) received")
    t_total = time.perf_counter()

    predictions = []

    for i, instance in enumerate(request.instances):
        logger.info(
            f"  [instance {i}] input_points={instance.input_points}, "
            f"input_labels={instance.input_labels}"
        )

        # --- Decode base64 image ---
        t_decode = time.perf_counter()
        try:
            image_bytes = base64.b64decode(instance.image)
        except Exception as e:
            logger.error(f"  [instance {i}] base64 decode failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

        image_size_kb = len(image_bytes) / 1024
        logger.info(f"  [instance {i}] image size: {image_size_kb:.1f} KB")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            logger.error(
                f"  [instance {i}] image too large: {image_size_kb:.1f} KB > "
                f"{MAX_IMAGE_BYTES // 1024} KB"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Max size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
            )

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"  [instance {i}] image decode failed: {e}")
            raise HTTPException(status_code=400, detail="Could not decode image.")

        elapsed_decode = time.perf_counter() - t_decode
        logger.info(
            f"  [instance {i}] image decoded: {image.width}x{image.height} "
            f"({elapsed_decode:.3f}s)"
        )

        # --- Run inference ---
        t_infer = time.perf_counter()
        try:
            result = app.state.predictor.predict(
                image=image,
                input_points=instance.input_points,
                input_labels=instance.input_labels,
            )
        except Exception as e:
            logger.error(f"  [instance {i}] inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        elapsed_infer = time.perf_counter() - t_infer
        logger.info(
            f"  [instance {i}] inference done: iou_score={result['iou_score']:.4f} "
            f"({elapsed_infer:.3f}s)"
        )

        predictions.append(result)

    elapsed_total = time.perf_counter() - t_total
    logger.info(
        f"POST /predict — completed {num_instances} instance(s) in {elapsed_total:.3f}s"
    )

    return JSONResponse(content={"predictions": predictions})
