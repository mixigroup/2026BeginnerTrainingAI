# Terraform - AI研修インフラ管理

GCPプロジェクト `hr-mixi` のインフラをTerraformで管理します。

## 管理対象リソース

- **GCS バケット**
  - `ai-training-terraform-state` — Terraform stateの保存先
  - `mixi-ml-workbench-notebook-utils` — Workbench用ユーティリティスクリプト
- **Vertex AI Workbench インスタンス** — メールアドレスリストから一括作成

## 前提条件

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- GCPプロジェクト `hr-mixi` への適切な権限（Editor以上）

## 初期セットアップ

### 1. Stateバケットの作成

Terraform backendに使用するGCSバケットを手動で作成します。

```bash
gcloud storage buckets create gs://ai-training-terraform-state \
  --project=hr-mixi \
  --location=asia-northeast1 \
  --uniform-bucket-level-access
```

### 2. Terraform初期化

```bash
cd terraform
terraform init
```

### 3. 既存リソースのインポート

既に存在するリソースをTerraform管理下に置きます。

```bash
# Stateバケット自体をインポート
terraform import google_storage_bucket.terraform_state hr-mixi/ai-training-terraform-state

# notebook-utilsバケットをインポート
terraform import google_storage_bucket.notebook_utils hr-mixi/mixi-ml-workbench-notebook-utils

# 既存のWorkbenchインスタンスをインポート（例: takahiro-kinouchi）
terraform import 'google_workbench_instance.workbench["takahiro-kinouchi"]' \
  projects/hr-mixi/locations/asia-northeast1-a/instances/takahiro-kinouchi
```

### 4. 差分確認

```bash
terraform plan
```

`No changes` と表示されれば成功です。差分がある場合は `main.tf` を調整してください。

## 使い方

### ユーザーの追加

`terraform.tfvars` にメールアドレスを追加します。

```hcl
workbench_emails = [
  "takahiro.kinouchi@mixi.co.jp",
  "taro.yamada@mixi.co.jp",      # 追加
  "hanako.suzuki@mixi.co.jp",    # 追加
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
| マシンタイプ | e2-standard-4 |
| ブートディスク | 150GB |
| データディスク | 100GB |
| リージョン | asia-northeast1-a |
| ネットワーク | default |
| サービスアカウント | ml-workbench-vm@hr-mixi.iam.gserviceaccount.com |
| 起動スクリプト | gs://mixi-ml-workbench-notebook-utils/entrypoint.sh |
| アイドルタイムアウト | 3時間 (10800秒) |
