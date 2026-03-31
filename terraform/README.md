# Terraform - AI研修インフラ管理

GCPプロジェクト `hr-mixi` のインフラをTerraformで管理します。

## 管理対象リソース

- **GCS バケット**
  - `ai-training-terraform-state` — Terraform stateの保存先
  - `mixi-ml-workbench-notebook-utils` — Workbench用ユーティリティスクリプト
- **サービスアカウント**
  - `ml-workbench-vm@hr-mixi.iam.gserviceaccount.com` — Workbench VM 用
  - 付与ロール: `roles/notebooks.admin`, `roles/storage.admin`
- **Vertex AI Workbench インスタンス** — メールアドレスリストから一括作成

## 前提条件

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- GCPプロジェクト `hr-mixi` への適切な権限（Editor以上）

## 使い方

### ユーザーの追加

`terraform.tfvars` にメールアドレスを追加します。
`terraform.tfvars.example` をコピーして作成してください。

```bash
cp terraform.tfvars.example terraform.tfvars
```

```hcl
workbench_emails = [
  "taro.yamada@mixi.co.jp",
  "hanako.suzuki@mixi.co.jp",
]
```

メールアドレスからインスタンス名が自動生成されます（例: `taro.yamada@mixi.co.jp` → `taro-yamada`）。

### 変更の確認と適用

```bash
# 変更内容を確認
terraform plan

# 問題なければ適用
terraform apply
```

### ユーザーの削除

`terraform.tfvars` からメールアドレスを削除し、`terraform apply` を実行します。

> **注意**: インスタンスの削除はデータの喪失を伴います。削除前にユーザーに通知してください。

## Workbenchインスタンスの構成

| 項目 | 値 |
|------|-----|
| マシンタイプ | n1-standard-4 |
| GPU | NVIDIA Tesla T4 × 1 |
| ブートディスク | 150GB |
| データディスク | 100GB |
| リージョン | asia-northeast1-a |
| ネットワーク | default |
| サービスアカウント | <ml-workbench-vm@hr-mixi.iam.gserviceaccount.com> |
| 起動スクリプト | gs://mixi-ml-workbench-notebook-utils/entrypoint.sh |
| アイドルタイムアウト | 3時間 (10800秒) |
