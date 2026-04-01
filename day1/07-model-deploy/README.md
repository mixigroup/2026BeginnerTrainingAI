# 07-model-deploy: Vertex AI デプロイ ハンズオン

06で作ったSAM（Segment Anything Model）を **FastAPI + カスタムコンテナ** で Vertex AI にデプロイします。

## 学習目標

- **MLOpsの定石** 「モデルとコンテナを分離する」を体験する
- GCS へのモデル管理 → Artifact Registry へのコンテナ push → Vertex AI デプロイの一連の流れを理解する
- Vertex AI でのトラフィック分割（A/Bテスト）の仕組みを知る

---

## 事前準備

### 必要なツール

```bash
# Google Cloud CLI の確認
gcloud --version

# Docker の確認
docker --version

# uv の確認
uv --version
```

### GCP 認証

```bash
# アプリケーションデフォルト認証（ローカル開発用）
gcloud auth application-default login

# Docker が Artifact Registry を使えるよう設定
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
```

### 環境変数（自分の名前を入れる）

```bash
export USER=your_name          # TODO: 自分の名前（英字小文字）に変更
export PROJECT_ID=hr-mixi
export REGION=asia-northeast1
export GCS_BUCKET=hr-mixi-ml-hands-on
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-hands-on/sam-server:${USER}"
```

---

## Step 0: モデルを GCS にアップロード

SAM モデルを `save_pretrained()` で保存し、GCS にアップロードします。

```bash
# SAM モデルの保存（Python で実行）
python -c "
from transformers import SamModel, SamProcessor
model = SamModel.from_pretrained('facebook/sam-vit-base')
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
model.save_pretrained('sam-vit-base')
processor.save_pretrained('sam-vit-base')
"

# GCS にアップロード
gcloud storage cp -r sam-vit-base/* \
    gs://${GCS_BUCKET}/2026/models/${USER}/sam-model/
```

---

## Step 1: FastAPI サーバの確認

`src/app.py` と `src/predictor.py` を確認してください。

**設計のポイント**:
- Vertex AI 上では `artifact_uri` で指定したモデルが管理バケットにコピーされ、`AIP_STORAGE_URI` 環境変数が自動設定される
- ローカルテスト時は `MODEL_GCS_URI` 環境変数でフォールバック
- サーバ起動時（`lifespan`）に GCS からモデルをダウンロード
- モデルはコンテナに含まない → コンテナを再ビルドしなくても良い

---

## Step 2: ローカルで Docker テスト

```bash
# イメージをビルド
docker build -t sam-server .

# ローカルで起動（GCS認証はホスト側の設定をマウント）
docker run -p 8080:8080 \
    -e MODEL_GCS_URI="gs://${GCS_BUCKET}/2026/models/${USER}/sam-model/" \
    -e GOOGLE_CLOUD_PROJECT="${PROJECT_ID}" \
    -v ~/.config/gcloud:/root/.config/gcloud:ro \
    sam-server

# 別ターミナルでヘルスチェック
curl http://localhost:8080/health

# 推論テスト（サンプル画像で確認）
IMAGE_B64=$(base64 -i images/sample.jpeg)
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"instances": [{"image": "'"$IMAGE_B64"'", "input_points": [[100, 100]], "input_labels": [1]}]}'
```

---

## Step 3: Artifact Registry にコンテナを push

```bash
docker tag sam-server ${IMAGE_URI}
docker push ${IMAGE_URI}
```

---

## Step 4: Vertex AI にデプロイ

### モデル登録

```bash
gcloud ai models upload \
    --region=${REGION} \
    --display-name=sam-server-${USER} \
    --artifact-uri=gs://${GCS_BUCKET}/2026/models/${USER}/sam-model/ \
    --container-image-uri=${IMAGE_URI} \
    --container-health-route=/health \
    --container-predict-route=/predict \
    --container-ports=8080
```

### エンドポイント作成

```bash
gcloud ai endpoints create \
    --region=${REGION} \
    --display-name=sam-endpoint-${USER}
```

### デプロイ

```bash
# TODO: 上記コマンドの出力から ENDPOINT_ID と MODEL_ID を取得して入力
ENDPOINT_ID=___
MODEL_ID=___

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
    --region=${REGION} \
    --model=${MODEL_ID} \
    --display-name=sam-server-${USER} \
    --machine-type=n1-standard-2
```

---

## Step 5: エンドポイント推論テスト

`notebook.py` の Step 5 セクションを参照してください。

---

## Step 6: A/B テスト（トラフィック分割）

新しいモデル（v2）をデプロイして、20% のトラフィックを割り当てる例:

```bash
# v2 モデルを GCS にアップロード
gcloud storage cp -r sam-vit-base-v2/* gs://${GCS_BUCKET}/2026/models/${USER}/sam-model-v2/

# 新しいモデルをモデルレジストリに登録
gcloud ai models upload \
    --region=${REGION} \
    --display-name=sam-server-v2-${USER} \
    --artifact-uri=gs://${GCS_BUCKET}/2026/models/${USER}/sam-model-v2/ \
    --container-image-uri=${IMAGE_URI} \
    --container-health-route=/health \
    --container-predict-route=/predict \
    --container-ports=8080

MODEL_V2_ID=___  # TODO: 新しいモデルのIDを入力

# v1を80%, v2を20%でデプロイ
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
    --region=${REGION} \
    --model=${MODEL_V2_ID} \
    --display-name=sam-model-v2 \
    --traffic-split=0=80,${MODEL_V2_DEPLOYMENT_ID}=20 \
    --machine-type=n1-standard-2
```

---

## ファイル構成

```
07-model-deploy/
├── notebook.py        # marimoノートブック（メインのハンズオン）
├── src/
│   ├── app.py         # FastAPI推論サーバ
│   └── predictor.py   # SAMモデル読み込み・推論
├── Dockerfile         # カスタムコンテナ定義（モデルを含まない）
├── deploy.py          # ワンショットデプロイスクリプト
├── pyproject.toml     # 依存関係
└── README.md          # このファイル
```

---

## GCP リソース情報

| 項目 | 値 |
|---|---|
| プロジェクト ID | `hr-mixi` |
| リージョン | `asia-northeast1`（東京） |
| GCS バケット | `gs://hr-mixi-ml-hands-on/` |
| Artifact Registry | `asia-northeast1-docker.pkg.dev/hr-mixi/ml-hands-on` |
