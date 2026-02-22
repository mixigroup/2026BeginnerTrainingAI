"""
deploy.py - One-shot deployment script for Vertex AI custom container.

Automates the following steps:
  1. Upload the ONNX model to GCS
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
  - ONNX model file at ../06-accelerate-ml-model/yolo26m-pose.onnx
"""

import argparse
import subprocess
import sys
import time

from google.cloud import aiplatform, storage

# ---------------------------------------------------------------------------
# Configuration (change if needed)
# ---------------------------------------------------------------------------
PROJECT_ID = "hr-mixi"
REGION = "asia-northeast1"
GCS_BUCKET = "mixi-ml-handson-2026"
AR_REPO = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-handson"
LOCAL_ONNX = "../06-accelerate-ml-model/yolo26m-pose.onnx"
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
    """Upload the ONNX model to GCS and return the GCS URI."""
    gcs_uri = f"gs://{GCS_BUCKET}/models/{user}/yolo.onnx"
    print(f"\n[Step 0] Uploading model to GCS: {gcs_uri}")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"models/{user}/yolo.onnx")
    blob.upload_from_filename(LOCAL_ONNX)

    print(f"  Uploaded: {LOCAL_ONNX} → {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Step 1: Build Docker image
# ---------------------------------------------------------------------------
def build_image(image_uri: str) -> None:
    """Build the Docker image locally."""
    print(f"\n[Step 1] Building Docker image: {image_uri}")
    run(["docker", "build", "-t", image_uri, "."], "docker build")


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
    print(f"\n[Step 3] Registering model in Vertex AI Model Registry...")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=f"yolo-server-{user}",
        serving_container_image_uri=image_uri,
        serving_container_environment_variables={"MODEL_GCS_URI": gcs_uri},
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
        serving_container_ports=[8080],
    )

    print(f"  Model registered: {model.resource_name}")
    return model.resource_name


# ---------------------------------------------------------------------------
# Step 4: Create endpoint
# ---------------------------------------------------------------------------
def create_endpoint(user: str) -> str:
    """Create a Vertex AI endpoint and return its resource name."""
    print(f"\n[Step 4] Creating Vertex AI endpoint...")

    endpoint = aiplatform.Endpoint.create(
        display_name=f"yolo-endpoint-{user}",
    )

    print(f"  Endpoint created: {endpoint.resource_name}")
    return endpoint.resource_name


# ---------------------------------------------------------------------------
# Step 5: Deploy model to endpoint
# ---------------------------------------------------------------------------
def deploy_model(model_name: str, endpoint_name: str, user: str) -> None:
    """Deploy the model to the endpoint."""
    print(f"\n[Step 5] Deploying model to endpoint (this takes ~5-10 minutes)...")

    model = aiplatform.Model(model_name)
    endpoint = aiplatform.Endpoint(endpoint_name)

    deployed = endpoint.deploy(
        model=model,
        deployed_model_display_name=f"yolo-{user}",
        machine_type=MACHINE_TYPE,
        traffic_percentage=100,
    )

    print(f"  Deployment complete.")
    print(f"  Endpoint: {endpoint.resource_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy YOLO model to Vertex AI custom container."
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

    gcs_uri = f"gs://{GCS_BUCKET}/models/{user}/yolo.onnx"
    image_uri = f"{AR_REPO}/yolo-server:{user}"

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
