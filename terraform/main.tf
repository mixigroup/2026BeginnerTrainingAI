locals {
  # "takahiro.kinouchi@mixi.co.jp" → "takahiro-kinouchi"
  workbench_instances = {
    for email in var.workbench_emails :
    replace(split("@", email)[0], ".", "-") => email
  }
}

# ==============================================================================
# GCS Buckets
# ==============================================================================

resource "google_storage_bucket" "terraform_state" {
  name                        = "ai-training-terraform-state"
  location                    = "ASIA-NORTHEAST1"
  project                     = var.project_id
  public_access_prevention    = "enforced"
  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "notebook_utils" {
  name                        = "mixi-ml-workbench-notebook-utils"
  location                    = "ASIA"
  project                     = var.project_id
  storage_class               = "STANDARD"
  public_access_prevention    = "enforced"
  uniform_bucket_level_access = true

  soft_delete_policy {
    retention_duration_seconds = 604800
  }
}

resource "google_storage_bucket" "ai_trainig_bucket" {
  name                        = "hr-mixi-ml-handson"
  location                    = "ASIA-NORTHEAST1"
  project                     = var.project_id
  public_access_prevention    = "enforced"
  uniform_bucket_level_access = true
}

# ==============================================================================
# Service Accounts
# ==============================================================================

resource "google_service_account" "ml_workbench_vm" {
  project      = var.project_id
  account_id   = "ml-workbench-vm"
  display_name = "ML Workbench VM Service Account"
  description  = "研修用 Workbench インスタンスに割り当てるサービスアカウント"
}

resource "google_project_iam_member" "ml_workbench_vm_notebooks_admin" {
  project = var.project_id
  role    = "roles/notebooks.admin"
  member  = "serviceAccount:${google_service_account.ml_workbench_vm.email}"
}

resource "google_project_iam_member" "ml_workbench_vm_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.ml_workbench_vm.email}"
}

# ==============================================================================
# Workbench Instances
# ==============================================================================

resource "google_workbench_instance" "workbench" {
  for_each = local.workbench_instances

  name     = each.key
  location = var.zone
  project  = var.project_id

  gce_setup {
    machine_type = "n1-standard-4"

    accelerator_configs {
      type       = "NVIDIA_TESLA_T4"
      core_count = 1
    }

    boot_disk {
      disk_size_gb    = 150
      disk_encryption = "GMEK"
    }

    data_disks {
      disk_size_gb    = 100
      disk_encryption = "GMEK"
    }

    network_interfaces {
      network = "default"
      subnet  = "default"
    }

    service_accounts {
      email = google_service_account.ml_workbench_vm.email
    }

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_vtpm                 = true
      enable_secure_boot          = false
    }

    tags = ["deeplearning-vm", "notebook-instance"]

    metadata = {
      disable-mixer                = "false"
      disable-swap-binaries        = "true"
      idle-timeout-seconds         = "10800"

      # 一定時間 cpuが使われてなかったら落とすcron jobを設定する
      # 詳細はbucketの中のスクリプトを参照
      post-startup-script          = "gs://mixi-ml-workbench-notebook-utils/entrypoint.sh" 
      post-startup-script-behavior = "run_every_start"
      proxy-mode                   = "service_account"
      serial-port-logging-enable   = "true"
    }
  }

  labels = {
    consumer-project-id     = "hr-mixi"
    consumer-project-number = "921041734393"
    notebooks-product       = "workbench-instances"
    resource-name           = each.key
  }

  lifecycle {
    ignore_changes = [
      gce_setup[0].network_interfaces,
      gce_setup[0].metadata,
    ]
  }
}
