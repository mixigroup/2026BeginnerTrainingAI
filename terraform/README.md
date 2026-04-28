# Terraform - AI研修インフラ管理

GCPプロジェクト `hr-mixi` のインフラをTerraformで管理します。

## 管理対象リソース

- **GCS バケット**
  - `ai-training-terraform-state` — Terraform stateの保存先
  - `mixi-ml-workbench-notebook-utils` — Workbench用ユーティリティスクリプト（`workbench_scritps/` 配下のファイルを配置）
  - `hr-mixi-ml-hands-on` — ML研修ハンズオン用データバケット
- **Artifact Registry**
  - `ml-hands-on` — ML研修用 Docker イメージリポジトリ
- **サービスアカウント**
  - `ml-workbench-vm@hr-mixi.iam.gserviceaccount.com` — Workbench VM 用
  - 付与ロール: `roles/notebooks.admin`, `roles/storage.admin`, `roles/aiplatform.user`, `roles/artifactregistry.writer`（リポジトリ単位）
- **ユーザー単位の IAM** — `workbench_emails` の各ユーザーに以下を付与
  - `roles/iap.httpsResourceAccessor` — IAP 経由で Workbench UI にアクセス
  - `roles/iap.tunnelResourceAccessor` — IAP トンネル経由で SSH
  - `roles/compute.osLogin` — OS Login で SSH ログイン
  - `roles/iam.serviceAccountUser` — VM にアタッチされた SA 経由で SSH
  - `roles/aiplatform.user` — ローカル ADC から Vertex AI API を呼び出す
- **Vertex AI Workbench インスタンス** — メールアドレスリストから一括作成

## 前提条件

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- GCPプロジェクト `hr-mixi` への適切な権限（Editor以上）

## 初回セットアップ（環境を再現する手順）

新しい環境で本構成を一から再現する場合は、次の順序で実行します。

### 1. gcloud 認証

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project hr-mixi
```

### 2. Terraform state 用バケットの事前作成

`backend.tf` で指定している state バケットは Terraform 管理外のため、初回のみ手動で作成します。

```bash
gcloud storage buckets create gs://ai-training-terraform-state \
  --project=hr-mixi \
  --location=ASIA-NORTHEAST1 \
  --uniform-bucket-level-access \
  --public-access-prevention

# バージョニングを有効化（state の世代管理用）
gcloud storage buckets update gs://ai-training-terraform-state --versioning
```

### 3. `terraform.tfvars` の作成

```bash
cp terraform.tfvars.example terraform.tfvars
# Workbench を作成したいユーザーのメールアドレスを追記
```

### 4. Terraform の初期化と適用

```bash
terraform init
terraform plan
terraform apply
```

### 5. Workbench 起動スクリプトのアップロード

`mixi-ml-workbench-notebook-utils` バケット（`terraform apply` で作成済み）に、`workbench_scritps/` 配下のスクリプトをアップロードします。これらは Workbench インスタンスの起動時 (`post-startup-script`) と、アイドル監視 cron で参照されるため、**バケット作成後に必ずアップロードする必要があります**。

```bash
gcloud storage cp workbench_scritps/entrypoint.sh \
  gs://mixi-ml-workbench-notebook-utils/entrypoint.sh

gcloud storage cp workbench_scritps/notebook-auto-shutdown.sh \
  gs://mixi-ml-workbench-notebook-utils/notebook-auto-shutdown.sh
```

スクリプトを更新した場合も、同じコマンドで再アップロードしてください。次回 VM 起動時 (`post-startup-script-behavior = run_every_start`) に反映されます。

#### スクリプトの役割

| ファイル | 役割 |
|---------|------|
| `entrypoint.sh` | Workbench VM 起動時に GCS から `notebook-auto-shutdown.sh` を取得し、10分間隔の cron に登録 |
| `notebook-auto-shutdown.sh` | GPU プロセスまたは CPU 使用率を監視し、8時間連続でアイドルなら `gcloud workbench instances stop` で VM を自動停止 |

## 運用

### ユーザーの追加

`terraform.tfvars` にメールアドレスを追加します。

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
| アイドルタイムアウト（Workbench組み込み） | 3時間 (10800秒) |
| アイドル時自動停止（カスタム cron） | 8時間連続アイドルで `instances stop` |
