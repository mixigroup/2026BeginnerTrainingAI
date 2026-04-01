"""
deploy.py - One-shot deployment script for Vertex AI custom container.

Automates the following steps:
  1. Upload the SAM model to GCS
  2. Build and push the Docker image to Artifact Registry
  3. Register the model in Vertex AI Model Registry
  4. Create an endpoint
  5. Deploy the model to the endpoint

Usage:
  python deploy.py --user alice
  python deploy.py --user alice --skip-docker   # skip docker build/push
  python deploy.py --user alice --skip-upload   # skip GCS upload

Requirements:
  - gcloud CLI authenticated (gcloud auth application-default login)
  - Docker configured for Artifact Registry:
      gcloud auth configure-docker asia-northeast1-docker.pkg.dev
  - SAM model saved with save_pretrained() at ../06-accelerate-ml-model/sam-vit-base/
"""

import argparse
import os
import subprocess
import sys

from google.cloud import aiplatform, storage

# ---------------------------------------------------------------------------
# Configuration (change if needed)
# ---------------------------------------------------------------------------
PROJECT_ID = "hr-mixi"
REGION = "asia-northeast1"
GCS_BUCKET = "hr-mixi-ml-hands-on"
AR_REPO = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-hands-on"
LOCAL_MODEL_DIR = "../06-accelerate-ml-model/sam-vit-base"
MACHINE_TYPE = "n1-standard-2"


# ---------------------------------------------------------------------------
# Helper: run a shell command and stream output
# ---------------------------------------------------------------------------
def run(cmd: list[str], description: str = "") -> None:
    """Run a shell command, streaming stdout/stderr. Raise on failure."""
    label = description or " ".join(cmd[:3])
    print(f"\n>>> {label}")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(
            f"[ERROR] Command failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def run_output(cmd: list[str]) -> str:
    """Run a command and return stripped stdout. Raise on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Step 0: Upload model to GCS
# ---------------------------------------------------------------------------
def upload_model(user: str) -> str:
    """Upload the SAM model directory to GCS and return the GCS URI."""
    gcs_prefix = f"2026/models/{user}/sam-model"
    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_prefix}/"
    print(f"\n[Step 0] Uploading model to GCS: {gcs_uri}")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)

    for root, _dirs, files in os.walk(LOCAL_MODEL_DIR):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, LOCAL_MODEL_DIR)
            blob_path = f"{gcs_prefix}/{relative_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"  Uploaded: {relative_path}")

    print(f"  Upload complete: {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Step 1: Build Docker image
# ---------------------------------------------------------------------------
def build_image(image_uri: str) -> None:
    """Build the Docker image locally."""
    print(f"\n[Step 1] Building Docker image: {image_uri}")
    run(
        ["docker", "build", "--platform", "linux/amd64", "-t", image_uri, "."],
        "docker build",
    )


# ---------------------------------------------------------------------------
# Step 2: Push to Artifact Registry
# ---------------------------------------------------------------------------
def push_image(image_uri: str) -> None:
    """Push the Docker image to Artifact Registry."""
    print(f"\n[Step 2] Pushing image to Artifact Registry: {image_uri}")
    run(["docker", "push", image_uri], "docker push")


# ---------------------------------------------------------------------------
# Step 3: Register model in Vertex AI Model Registry
# ---------------------------------------------------------------------------
def upload_vertex_model(user: str, image_uri: str, gcs_uri: str) -> str:
    """Upload model to Vertex AI and return the model resource name."""
    print("\n[Step 3] Registering model in Vertex AI Model Registry...")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=f"sam-server-{user}",
        artifact_uri=gcs_uri,
        serving_container_image_uri=image_uri,
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
        serving_container_ports=[8080],
    )

    print(f"  Model registered: {model.resource_name}")  # pyright: ignore
    return model.resource_name  # pyright: ignore


# ---------------------------------------------------------------------------
# Step 4: Create endpoint
# ---------------------------------------------------------------------------
def create_endpoint(user: str) -> str:
    """Create a Vertex AI endpoint and return its resource name."""
    print("\n[Step 4] Creating Vertex AI endpoint...")

    endpoint = aiplatform.Endpoint.create(
        display_name=f"sam-endpoint-{user}",
    )

    print(f"  Endpoint created: {endpoint.resource_name}")
    return endpoint.resource_name


# ---------------------------------------------------------------------------
# Step 5: Deploy model to endpoint
# ---------------------------------------------------------------------------
def deploy_model(model_name: str, endpoint_name: str, user: str) -> None:
    """Deploy the model to the endpoint."""
    print("\n[Step 5] Deploying model to endpoint (this takes ~5-10 minutes)...")

    model = aiplatform.Model(model_name)
    endpoint = aiplatform.Endpoint(endpoint_name)

    _ = endpoint.deploy(
        model=model,
        deployed_model_display_name=f"sam-{user}",
        machine_type=MACHINE_TYPE,
        traffic_percentage=100,
    )

    print("  Deployment complete.")
    print(f"  Endpoint: {endpoint.resource_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy SAM model to Vertex AI custom container."
    )
    parser.add_argument(
        "--user",
        required=True,
        help="Your name (lowercase English). Used as a namespace in GCS and display names.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip Step 0 (GCS model upload). Use this if the model is already uploaded.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Steps 1-2 (Docker build and push). Use this if the image is already pushed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user = args.user.lower()

    gcs_uri = f"gs://{GCS_BUCKET}/2026/models/{user}/sam-model/"
    image_uri = f"{AR_REPO}/sam-server:{user}"

    print("=" * 60)
    print("Vertex AI Deployment")
    print("=" * 60)
    print(f"  User        : {user}")
    print(f"  Project     : {PROJECT_ID}")
    print(f"  Region      : {REGION}")
    print(f"  Model GCS   : {gcs_uri}")
    print(f"  Image URI   : {image_uri}")
    print("=" * 60)

    # Step 0: GCS upload
    if not args.skip_upload:
        upload_model(user)
    else:
        print("\n[Step 0] Skipped (--skip-upload)")

    # Steps 1-2: Docker build and push
    if not args.skip_docker:
        build_image(image_uri)
        push_image(image_uri)
    else:
        print("\n[Step 1-2] Skipped (--skip-docker)")

    # Step 3: Register model in Vertex AI
    model_name = upload_vertex_model(user, image_uri, gcs_uri)

    # Step 4: Create endpoint
    endpoint_name = create_endpoint(user)

    # Step 5: Deploy
    deploy_model(model_name, endpoint_name, user)

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)
    print(f"  Model    : {model_name}")
    print(f"  Endpoint : {endpoint_name}")
    print()
    print("To run inference, update ENDPOINT_ID in notebook.py:")
    # Extract the endpoint ID (last segment of the resource name)
    endpoint_id = endpoint_name.split("/")[-1]
    print(f'  ENDPOINT_ID = "{endpoint_id}"')


if __name__ == "__main__":
    main()
