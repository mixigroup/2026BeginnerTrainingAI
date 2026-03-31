variable "project_id" {
  description = "GCPプロジェクトID"
  type        = string
  default     = "hr-mixi"
}

variable "region" {
  description = "デフォルトリージョン"
  type        = string
  default     = "asia-northeast1"
}

variable "zone" {
  description = "Workbenchインスタンスのゾーン"
  type        = string
  default     = "asia-northeast1-a"
}

variable "workbench_emails" {
  description = "Workbenchインスタンスを作成するユーザーのメールアドレスリスト"
  type        = list(string)
}
