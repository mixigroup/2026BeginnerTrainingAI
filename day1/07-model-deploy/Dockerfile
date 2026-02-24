# ============================================================
# YOLO Inference Server - Custom Container for Vertex AI
#
# Key design decision: the model is NOT copied into the image.
# At container startup, the model is downloaded from GCS via
# the MODEL_GCS_URI environment variable.
#
# This means:
#   - Updating the model → only update GCS, no docker rebuild needed
#   - Container stays lightweight (no multi-hundred MB model file)
#   - Follows the Vertex AI recommended pattern (AIP_STORAGE_URI)
# ============================================================

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
