terraform {
  backend "gcs" {
    bucket = "ai-training-terraform-state"
    prefix = "terraform/state"
  }
}
