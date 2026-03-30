import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MLモデルのデプロイ ハンズオン

    このノートブックでは、06で作った **YOLO モデル（ONNX形式）** を
    **FastAPI + カスタムコンテナ** で Vertex AI にデプロイする流れを体験します。

    ## 全体の流れ

    ```
    Step 0: モデルをGCSにアップロード（06の成果物）
      ↓
    Step 1: FastAPIサーバのコード確認
      ↓
    Step 2: ローカルでDockerテスト
      ↓
    Step 3: Artifact Registryにコンテナをpush
      ↓
    Step 4: Vertex AIにデプロイ（gcloud CLI）
      ↓
    Step 5: エンドポイントに推論リクエスト（Python SDK）
      ↓
    [拡張] Step 6: A/Bテスト（トラフィック分割）
    [拡張] Step 7: Gradioデモアプリ
    ```

    ## 今日のキーコンセプト：モデルとコンテナを分離する

    | ❌ アンチパターン | ✅ 今回学ぶパターン |
    |---|---|
    | モデルをDockerイメージに焼き込む | モデルをGCSに保管 |
    | モデル更新 → コンテナリビルド必要 | モデル更新だけで済む |
    | コンテナサイズが肥大化 | コンテナは軽量 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セットアップ
    """)
    return


@app.cell
def _():
    # --- TODO: 自分の名前（英字小文字）を入れてください ---
    USER = "your_name"

    if USER == "your_name":
        raise ValueError("USER を自分の名前（英字小文字）に変更してください！")

    PROJECT_ID = "hr-mixi"
    REGION = "asia-northeast1"
    GCS_BUCKET = "mixi-ml-handson-2026"

    MODEL_GCS_URI = f"gs://{GCS_BUCKET}/models/{USER}/yolo.onnx"
    IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-handson/yolo-server:{USER}"

    print(f"USER          : {USER}")
    print(f"MODEL_GCS_URI : {MODEL_GCS_URI}")
    print(f"IMAGE_URI     : {IMAGE_URI}")
    return GCS_BUCKET, IMAGE_URI, MODEL_GCS_URI, PROJECT_ID, REGION, USER


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 0: モデルを GCS にアップロード

    ### なぜ GCS にモデルを置くのか？

    MLOps の定石として、**モデルとサービングコードを分離**します。

    - モデルファイル（数百MB）をコンテナに含めると、更新のたびに `docker build → push` が必要
    - GCS に置くことで、モデルの更新はファイルのアップロードだけで完結
    - コンテナはサービングロジックのみ管理 → 責務の明確化

    > Vertex AI も公式にこのパターンを推奨しており、`AIP_STORAGE_URI` という環境変数でモデルパスを渡す仕組みがあります。
    """)
    return


@app.cell
def _(GCS_BUCKET, USER, mo):
    import subprocess

    upload_cmd = f"gsutil cp ../06-accelerate-ml-model/yolo26m-pose.onnx gs://{GCS_BUCKET}/models/{USER}/yolo.onnx"
    mo.md(
        f"""
        以下のコマンドを実行してモデルをアップロードしてください：

        ```bash
        {upload_cmd}
        ```

        アップロードできたか確認：

        ```bash
        gsutil ls gs://{GCS_BUCKET}/models/{USER}/
        ```
        """
    )
    return (subprocess,)


@app.cell
def _(GCS_BUCKET, USER, mo, subprocess):
    # Verify the model exists on GCS
    verify_result = subprocess.run(
        ["gsutil", "ls", f"gs://{GCS_BUCKET}/models/{USER}/"],
        capture_output=True,
        text=True,
    )

    if verify_result.returncode == 0 and verify_result.stdout.strip():
        mo.md(f"""
        **GCS 確認完了！**

        ```
        {verify_result.stdout.strip()}
        ```
        """)
    else:
        mo.md("""
        > **GCS にモデルが見つかりません。**
        >
        > 上のコマンドを実行してからこのセルを再実行してください。
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 1: FastAPI サーバのコード確認

    ### 設計のポイント

    コンテナ起動時（`lifespan`）に GCS からモデルをダウンロードします。
    モデルはコンテナに含まれていません。

    ### `src/predictor.py` - モデルの読み込みと推論
    """)
    return


@app.cell
def _(mo):
    with open("src/predictor.py") as f:
        predictor_code = f.read()

    mo.md(f"```python\n{predictor_code}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### `src/app.py` - FastAPI サーバ
    """)
    return


@app.cell
def _(mo):
    with open("src/app.py") as f2:
        app_code = f2.read()

    mo.md(f"```python\n{app_code}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ポイント：`lifespan` でモデルをダウンロード

    ```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # サーバ起動時に GCS からモデルをダウンロード
        download_model(MODEL_GCS_URI, LOCAL_MODEL_PATH)
        app.state.predictor = YOLOPredictor(LOCAL_MODEL_PATH)
        yield
    ```

    - `MODEL_GCS_URI` は**環境変数**で渡す → コンテナを変えずにモデルを切り替えられる
    - `/health` エンドポイント → Vertex AI のヘルスチェックに必要
    - `/predict` エンドポイント → 画像ファイルを受け取り、検出結果をJSONで返す
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 2: ローカルで Docker テスト

    ### Dockerfile の確認

    モデルはコンテナに含まれていないことを確認してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    with open("Dockerfile") as f3:
        dockerfile_code = f3.read()

    mo.md(f"```dockerfile\n{dockerfile_code}\n```")
    """)
    return


@app.cell(hide_code=True)
def _(MODEL_GCS_URI, mo):
    mo.md(f"""
    ### Docker ビルドと起動

    以下のコマンドをターミナルで実行してください：

    ```bash
    # 1. イメージをビルド
    docker build -t yolo-server .

    # 2. ローカルで起動（GCS認証をマウント）
    docker run -p 8080:8080 \\
        -e MODEL_GCS_URI="{MODEL_GCS_URI}" \\
        -v ~/.config/gcloud:/root/.config/gcloud:ro \\
        yolo-server
    ```

    別のターミナルで動作確認：

    ```bash
    # ヘルスチェック
    curl http://localhost:8080/health
    # 期待する結果: {{"status": "ok"}}

    # 推論テスト
    curl -X POST http://localhost:8080/predict \\
        -F "file=@../06-accelerate-ml-model/images/bus.jpg"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ローカルテストの確認

    以下を実行して、正常に応答が返ってくることを確認してください：

    - `/health` → `{"status": "ok"}` が返る
    - `/predict` → `{"detections": [...]}` 形式の JSON が返る

    問題なければ次のステップへ進みましょう。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 3: Artifact Registry にコンテナを push

    ビルドしたイメージを Google Cloud の **Artifact Registry** に push します。
    Vertex AI はここからコンテナイメージを取得してデプロイします。
    """)
    return


@app.cell(hide_code=True)
def _(IMAGE_URI, mo):
    mo.md(f"""
    ```bash
    # タグをつける
    docker tag yolo-server {IMAGE_URI}

    # push
    docker push {IMAGE_URI}
    ```

    push が完了したら確認：

    ```bash
    gcloud artifacts docker images list \\
        asia-northeast1-docker.pkg.dev/hr-mixi/ml-handson \\
        --filter="package=yolo-server"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 4: Vertex AI にデプロイ

    ### 4-1. モデルをモデルレジストリに登録

    コンテナイメージと環境変数（モデルのGCSパス）を組み合わせて登録します。
    """)
    return


@app.cell
def _(IMAGE_URI, MODEL_GCS_URI, REGION, USER, mo):
    mo.md(f"""
    ```bash
    gcloud ai models upload \\
        --region={REGION} \\
        --display-name=yolo-server-{USER} \\
        --container-image-uri={IMAGE_URI} \\
        --container-env-vars=MODEL_GCS_URI={MODEL_GCS_URI} \\
        --container-health-route=/health \\
        --container-predict-route=/predict \\
        --container-ports=8080
    ```

    コマンド実行後、出力された `MODEL_ID` をメモしておいてください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4-2. エンドポイントの作成
    """)
    return


@app.cell(hide_code=True)
def _(REGION, USER, mo):
    mo.md(f"""
    ```bash
    gcloud ai endpoints create \\
        --region={REGION} \\
        --display-name=yolo-endpoint-{USER}
    ```

    コマンド実行後、出力された `ENDPOINT_ID` をメモしておいてください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4-3. エンドポイントにモデルをデプロイ
    """)
    return


@app.cell
def _():
    # --- TODO: gcloud コマンドの出力から ID を入力してください ---
    ENDPOINT_ID = "___"  # TODO: gcloud ai endpoints create の出力から
    MODEL_ID = "___"  # TODO: gcloud ai models upload の出力から
    return ENDPOINT_ID, MODEL_ID


@app.cell(hide_code=True)
def _(ENDPOINT_ID, MODEL_ID, REGION, USER, mo):
    mo.md(f"""
    ```bash
    gcloud ai endpoints deploy-model {ENDPOINT_ID} \\
        --region={REGION} \\
        --model={MODEL_ID} \\
        --display-name=yolo-server-{USER} \\
        --machine-type=n1-standard-2
    ```

    > デプロイには **5〜10分** かかります。
    > `gcloud ai endpoints describe {ENDPOINT_ID} --region={REGION}` でステータスを確認できます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 5: エンドポイントへの推論テスト

    Vertex AI Python SDK を使って、デプロイしたエンドポイントに推論リクエストを送ります。
    """)
    return


@app.cell
def _(ENDPOINT_ID, PROJECT_ID, REGION, mo):
    import base64

    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)

    if ENDPOINT_ID == "___":
        mo.md("> **TODO**: 上のセルで `ENDPOINT_ID` を設定してください。")
    else:
        mo.md(f"Vertex AI SDK 初期化完了（project={PROJECT_ID}, region={REGION}）")
    return aiplatform, base64


@app.cell
def _(ENDPOINT_ID, aiplatform, base64, mo):
    import pathlib

    # Use a sample image from 06
    sample_image_path = pathlib.Path("../06-accelerate-ml-model/images/bus.jpg")

    if ENDPOINT_ID == "___":
        mo.md("> `ENDPOINT_ID` を設定してから実行してください。")
    elif not sample_image_path.exists():
        mo.md(f"> サンプル画像が見つかりません: `{sample_image_path}`")
    else:
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)

        with open(sample_image_path, "rb") as img_f:
            image_b64 = base64.b64encode(img_f.read()).decode("utf-8")

        # Vertex AI SDK sends the request as {"instances": [...]} to /predict
        # Our app.py returns {"predictions": [...]} in the same format
        response = endpoint.predict(instances=[{"image": image_b64}])

        # response.predictions is a list matching the request instances
        mo.md(
            f"""
            ### 推論結果

            検出数: **{len(response.predictions[0].get("detections", []))}** 件

            ```python
            {response.predictions}
            ```
            """
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 6: A/B テスト（トラフィック分割）

    ### なぜ A/B テストが必要か

    - オフライン指標（mAP など）が良くても、**本番データでの挙動は別**
    - 新モデルをいきなり100%に切り替えるのはリスクが高い
    - 少量のトラフィックで本番環境の動作を確認してから段階的に移行する

    ### 段階的なロールアウト

    ```
    新モデル20% ─→ 新モデル50% ─→ 新モデル100%
               測定             測定
    ```

    各段階で「エラー率」「レイテンシ」「検出精度」を確認します。

    ### Vertex AI でのトラフィック分割

    Vertex AI のエンドポイントは**複数のモデルを同時にデプロイ**できます。
    """)
    return


@app.cell(hide_code=True)
def _(ENDPOINT_ID, REGION, USER, mo):
    mo.md(f"""
    ```bash
    # v2 モデルを GCS にアップロード（再学習モデルや別のモデル）
    gsutil cp new_model.onnx gs://mixi-ml-handson-2026/models/{USER}/yolo_v2.onnx

    # v2 をモデルレジストリに登録
    gcloud ai models upload \\
        --region={REGION} \\
        --display-name=yolo-server-v2-{USER} \\
        --container-image-uri=IMAGE_URI \\
        --container-env-vars=MODEL_GCS_URI=gs://mixi-ml-handson-2026/models/{USER}/yolo_v2.onnx \\
        --container-health-route=/health \\
        --container-predict-route=/predict \\
        --container-ports=8080

    MODEL_V2_ID=___  # TODO: 上記コマンドの出力から

    # v1 を 80%、v2 を 20% にトラフィック分割
    gcloud ai endpoints deploy-model {ENDPOINT_ID} \\
        --region={REGION} \\
        --model=$MODEL_V2_ID \\
        --display-name=yolo-model-v2 \\
        --traffic-split=0=80,$MODEL_V2_DEPLOYMENT_ID=20 \\
        --machine-type=n1-standard-2
    ```

    同じエンドポイントに送るだけで、自動的に v1/v2 に振り分けられます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 7（拡張）: Gradio デモアプリ

    Vertex AI エンドポイントに接続した **Gradio** デモを作ります。
    画像をアップロードすると、検出結果を重ねて表示します。
    """)
    return


@app.cell
def _(ENDPOINT_ID, PROJECT_ID, REGION, base64, mo):
    import numpy as np
    from PIL import Image, ImageDraw

    def draw_detections(image_np: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and keypoints on the image."""
        pil_img = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_img)

        for det in detections:
            bbox = det.get("bbox", [])
            score = det.get("score", 0.0)
            class_id = det.get("class_id", 0)

            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 12), f"cls={class_id} {score:.2f}", fill="red")

            for kp in det.get("keypoints", []):
                cx, cy = kp["x"], kp["y"]
                r = 4
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="lime")

        return np.array(pil_img)

    if ENDPOINT_ID == "___":
        mo.md("> **TODO**: `ENDPOINT_ID` を設定してから実行してください。")
    else:
        import gradio as gr
        from google.cloud import aiplatform as _aiplatform

        _aiplatform.init(project=PROJECT_ID, location=REGION)
        _endpoint = _aiplatform.Endpoint(ENDPOINT_ID)

        def detect(image: np.ndarray):
            """Send image to Vertex AI endpoint and visualize results."""
            if image is None:
                return None
            pil = Image.fromarray(image)
            import io as _io

            buf = _io.BytesIO()
            pil.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            # Vertex AI SDK sends {"instances": [...]} → app.py returns {"predictions": [...]}
            resp = _endpoint.predict(instances=[{"image": img_b64}])
            detections = (
                resp.predictions[0].get("detections", []) if resp.predictions else []
            )
            return draw_detections(image, detections)

        demo = gr.Interface(
            fn=detect,
            inputs=gr.Image(label="Input Image"),
            outputs=gr.Image(label="Detection Result"),
            title="YOLO Inference Demo (Vertex AI)",
        )
        demo.launch(share=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## まとめ

    今日体験したこと：

    | ステップ | 学んだこと |
    |---|---|
    | Step 0 | GCSにモデルを置く → MLOpsのバージョン管理 |
    | Step 1 | FastAPI + lifespan でGCSからモデルをDL |
    | Step 2 | Dockerカスタムコンテナ（モデルを含まない設計） |
    | Step 3 | Artifact Registry へのコンテナ管理 |
    | Step 4 | Vertex AI へのデプロイ（モデル登録 → エンドポイント） |
    | Step 5 | Python SDK での推論リクエスト |
    | Step 6 | トラフィック分割でリスクを抑えたモデル更新 |

    ### 重要な設計原則

    > **「モデルとコンテナを分離する」**
    >
    > - コンテナ：サービングロジックのみ（再利用性が高い）
    > - モデル：GCSで管理（更新が独立している）
    > - 環境変数：2つをつなぐインターフェース
    """)
    return


if __name__ == "__main__":
    app.run()
