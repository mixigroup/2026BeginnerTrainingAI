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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# Resolve sibling module (predictor.py lives next to app.py in src/)
sys.path.insert(0, os.path.dirname(__file__))
from predictor import SAMPredictor, download_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "")
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "")  # ローカルテスト用フォールバック
LOCAL_MODEL_DIR = "/tmp/sam-model"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


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
    if AIP_STORAGE_URI:
        # Vertex AI が artifact_uri のコピーを管理バケットに作成し、
        # その gs:// URI を AIP_STORAGE_URI に設定する。
        # コンテナのデフォルト SA にはこの URI への読み取り権限が自動付与される。
        gcs_uri = AIP_STORAGE_URI
    elif MODEL_GCS_URI:
        # ローカル Docker テスト用フォールバック
        gcs_uri = MODEL_GCS_URI
    else:
        raise RuntimeError(
            "Neither AIP_STORAGE_URI nor MODEL_GCS_URI is set. "
            "On Vertex AI, AIP_STORAGE_URI is set automatically via artifact_uri. "
            "For local testing, pass -e MODEL_GCS_URI=gs://bucket/path/to/sam-model/"
        )
    download_model(gcs_uri, LOCAL_MODEL_DIR)
    app.state.predictor = SAMPredictor(LOCAL_MODEL_DIR)
    yield


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
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: PredictRequest) -> JSONResponse:
    """Run SAM inference on a batch of images.

    Accepts the Vertex AI prediction request format:
      {"instances": [{"image": "<base64>", "input_points": [[x,y]], "input_labels": [1]}, ...]}

    Returns:
      {"predictions": [{"mask_b64": "<base64 PNG>", "iou_score": 0.95}, ...]}

    Example curl (local test):
      IMAGE_B64=$(base64 -i sample.jpg)
      curl -X POST http://localhost:8080/predict \\
           -H "Content-Type: application/json" \\
           -d '{"instances": [{"image": "'"$IMAGE_B64"'", "input_points": [[100, 100]], "input_labels": [1]}]}'
    """
    predictions = []

    for instance in request.instances:
        # --- Decode base64 image ---
        try:
            image_bytes = base64.b64decode(instance.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Max size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
            )

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # --- Run inference ---
        try:
            result = app.state.predictor.predict(
                image=image,
                input_points=instance.input_points,
                input_labels=instance.input_labels,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        predictions.append(result)

    return JSONResponse(content={"predictions": predictions})
