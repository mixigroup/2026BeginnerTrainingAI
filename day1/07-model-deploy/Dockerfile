# ============================================================
# SAM Inference Server - Custom Container for Vertex AI
#
# Key design decision: the model is NOT copied into the image.
# On Vertex AI, artifact_uri で指定したモデルが Vertex AI 管理バケットにコピーされ、
# AIP_STORAGE_URI 環境変数 (gs:// URI) として自動設定される。
# コンテナ起動時にこの URI からモデルをダウンロードする。
#
# ローカルテスト時は MODEL_GCS_URI 環境変数で GCS URI を直接指定する。
# ============================================================

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Avoid writing .pyc files and enable unbuffered stdout for better logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy dependency files first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# Install production dependencies only (--frozen ensures exact reproducibility)
RUN uv sync --no-dev --frozen

# Copy server source code (NOT model files)
COPY src/ ./src/

# Vertex AI prediction containers must listen on port 8080
EXPOSE 8080

# Start the FastAPI server with uvicorn
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
