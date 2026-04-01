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

    このノートブックでは、06で作った **SAM (Segment Anything Model)** を
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
    USER = "takahiro_kinouchi"

    if USER == "your_name":
        raise ValueError("USER を自分の名前（英字小文字）に変更してください！")

    PROJECT_ID = "hr-mixi"
    REGION = "asia-northeast1"
    GCS_BUCKET = "hr-mixi-ml-hands-on"

    MODEL_GCS_URI = f"gs://{GCS_BUCKET}/2026/models/{USER}/sam-model/"
    IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-hands-on/sam-server:{USER}"

    print(f"USER          : {USER}")
    print(f"MODEL_GCS_URI : {MODEL_GCS_URI}")
    print(f"IMAGE_URI     : {IMAGE_URI}")
    return IMAGE_URI, MODEL_GCS_URI, PROJECT_ID, REGION, USER


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

    ### SAM モデルの保存

    まず、06 で使った SAM モデルを `save_pretrained()` で保存します。
    これにより `sam-vit-base/` ディレクトリに `config.json`, `model.safetensors` 等が保存されます。
    """)
    return


@app.cell
def _():
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    model.save_pretrained("sam-vit-base")
    processor.save_pretrained("sam-vit-base")
    return


@app.cell(hide_code=True)
def _(MODEL_GCS_URI, mo):
    mo.md(f"""
    保存したモデルディレクトリを GCS にアップロードしてください：

    ```bash
    gcloud storage cp -r sam-vit-base/* {MODEL_GCS_URI}
    ```

    アップロードできたか確認：

    ```bash
    gcloud storage ls {MODEL_GCS_URI}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 1: FastAPI サーバのコード確認

    ### コードの確認

    - src/predictor.py
    - src/app.py

    ### 設計のポイント

    コンテナ起動時（`lifespan`）に GCS からモデルをダウンロードします。
    モデルはコンテナに含まれていません。

    ```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # サーバ起動時に GCS からモデルをダウンロード
        download_model(MODEL_GCS_URI, LOCAL_MODEL_DIR)
        app.state.predictor = SAMPredictor(LOCAL_MODEL_DIR)
        yield
    ```

    - `MODEL_GCS_URI` は**環境変数**で渡す → コンテナを変えずにモデルを切り替えられる
    - `/health` エンドポイント → Vertex AI のヘルスチェックに必要
    - `/predict` エンドポイント → 画像 + ポイント座標を受け取り、セグメンテーションマスクをJSONで返す

    ### API の入出力

    **リクエスト:**
    ```json
    {
      "instances": [
        {
          "image": "<base64 画像>",
          "input_points": [[x, y]],
          "input_labels": [1]
        }
      ]
    }
    ```

    **レスポンス:**
    ```json
    {
      "predictions": [
        {
          "mask_b64": "<base64 PNG マスク画像>",
          "iou_score": 0.95
        }
      ]
    }
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 2: ローカルで Docker テスト

    ### Dockerfile の確認

    `Dockerfile` を確認して、モデルはコンテナに含まれていないことを確認してください。
    """)
    return


@app.cell(hide_code=True)
def _(MODEL_GCS_URI, PROJECT_ID, mo):
    mo.md(f"""
    ### Docker ビルドと起動

    以下のコマンドをターミナルで実行してください：

    ```bash
    # 1. イメージをビルド
    docker build -t sam-server .
    ```

    #### Mac（ローカル）の場合

    `~/.config/gcloud` の認証情報をマウントして使います。

    ```bash
    docker run -p 8080:8080 \\
        -e MODEL_GCS_URI="{MODEL_GCS_URI}" \\
        -e GOOGLE_CLOUD_PROJECT="{PROJECT_ID}" \\
        -v ~/.config/gcloud:/root/.config/gcloud:ro \\
        sam-server
    ```

    #### Vertex AI Workbench の場合

    GCE メタデータサーバー経由で認証するため、`--network host` を指定します。
    `~/.config/gcloud` のマウントは不要です。

    ```bash
    docker run --network host \\
        -e MODEL_GCS_URI="{MODEL_GCS_URI}" \\
        -e GOOGLE_CLOUD_PROJECT="{PROJECT_ID}" \\
        sam-server
    ```

    > `--network host` はコンテナがホストのネットワークを共有するため、`-p 8080:8080` は不要です。

    ---

    別のターミナルで動作確認：

    ```bash
    # ヘルスチェック
    curl http://localhost:8080/health
    # 期待する結果: {{"status": "ok"}}

    # 推論テスト（base64 JSON形式）
    IMAGE_B64=$(base64 -i ./images/sample.jpeg)
    curl -X POST http://localhost:8080/predict \\
        -H "Content-Type: application/json" \\
        -d '{{"instances": [{{"image": "'"$IMAGE_B64"'", "input_points": [[100, 100]], "input_labels": [1]}}]}}'
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ローカルテストの確認

    以下を実行して、正常に応答が返ってくることを確認してください：

    - `/health` → `{"status": "ok"}` が返る
    - `/predict` → `{"predictions": [{"mask_b64": "...", "iou_score": ...}]}` 形式の JSON が返る

    問題なければ次のステップへ進みましょう。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ローカルテストの可視化

    curl だと `mask_b64` が長い文字列で結果がわかりません。
    以下のセルを実行すると、ローカルサーバに Python でリクエストを送り、
    マスクを元画像にオーバーレイして表示します。

    > **注意**: Docker コンテナが `localhost:8080` で起動中であることを確認してください。
    """)
    return


@app.cell
def _(mo):
    import base64 as _b64
    import io as _io
    import pathlib as _pathlib

    import numpy as _np
    import requests as _requests
    from PIL import Image as _Image

    # サンプル画像のパス
    _sample_path = _pathlib.Path("images/sample.jpeg")
    if not _sample_path.exists():
        _sample_path = _pathlib.Path("../06-accelerate-ml-model/images/sample.jpg")

    # 画像を読み込み base64 エンコード
    _img = _Image.open(_sample_path).convert("RGB")
    _buf = _io.BytesIO()
    _img.save(_buf, format="JPEG")
    _image_b64 = _b64.b64encode(_buf.getvalue()).decode("utf-8")

    # 画像の中心付近をポイントとして指定
    _cx, _cy = _img.width // 2, _img.height // 2

    # ローカルサーバにリクエスト送信
    _resp = _requests.post(
        "http://localhost:8080/predict",
        json={
            "instances": [
                {
                    "image": _image_b64,
                    "input_points": [[_cx, _cy]],
                    "input_labels": [1],
                }
            ]
        },
        timeout=60,
    )
    _resp.raise_for_status()
    _pred = _resp.json()["predictions"][0]

    # マスクをデコードしてオーバーレイ
    _mask_bytes = _b64.b64decode(_pred["mask_b64"])
    _mask_array = _np.array(_Image.open(_io.BytesIO(_mask_bytes)))
    _img_array = _np.array(_img)
    _overlay = _img_array.copy()
    _overlay[_mask_array > 0] = (
        _overlay[_mask_array > 0] * 0.5 + _np.array([30, 144, 255]) * 0.5
    ).astype(_np.uint8)
    _result_image = _Image.fromarray(_overlay)

    mo.md(
        f"""
    ### ローカルテスト推論結果

    - ポイント座標: ({_cx}, {_cy})
    - IoU スコア: **{_pred["iou_score"]:.3f}**
    """
    )
    mo.image(_result_image)
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
    docker tag sam-server {IMAGE_URI}

    # push
    docker push {IMAGE_URI}
    ```

    push が完了したら確認：

    ```bash
    gcloud artifacts docker images list \\
        asia-northeast1-docker.pkg.dev/hr-mixi/ml-hands-on \\
        --filter="package=sam-server"
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


@app.cell(hide_code=True)
def _(IMAGE_URI, MODEL_GCS_URI, REGION, USER, mo):
    mo.md(f"""
    ```bash
    gcloud ai models upload \\
        --region={REGION} \\
        --display-name=sam-server-{USER} \\
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
        --display-name=sam-endpoint-{USER}
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
    ENDPOINT_ID = "1572330215022002176"  # TODO: gcloud ai endpoints create の出力から
    MODEL_ID = "5209393733425954816"  # TODO: gcloud ai models upload の出力から
    return ENDPOINT_ID, MODEL_ID


@app.cell(hide_code=True)
def _(ENDPOINT_ID, MODEL_ID, REGION, USER, mo):
    mo.md(f"""
    ```bash
    gcloud ai endpoints deploy-model {ENDPOINT_ID} \\
        --region={REGION} \\
        --model={MODEL_ID} \\
        --display-name=sam-server-{USER} \\
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
    画像とポイント座標を送ると、セグメンテーションマスクが返ってきます。
    """)
    return


@app.cell
def _(ENDPOINT_ID, PROJECT_ID, REGION, mo):
    import base64

    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)

    if ENDPOINT_ID == "___":
        raise RuntimeError("TODO: 上のセルで ENDPOINT_ID を設定してください。")

    mo.md(f"Vertex AI SDK 初期化完了（project={PROJECT_ID}, region={REGION}）")
    return aiplatform, base64


@app.cell
def _(ENDPOINT_ID, aiplatform, base64, mo):
    import io
    import pathlib

    import numpy as np
    from PIL import Image

    # 06 のサンプル画像を使用
    sample_image_path = pathlib.Path("../06-accelerate-ml-model/images/sample.jpg")

    if ENDPOINT_ID == "___":
        raise RuntimeError("ENDPOINT_ID を設定してから実行してください。")
    if not sample_image_path.exists():
        raise RuntimeError(f"サンプル画像が見つかりません: {sample_image_path}")

    endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    with open(sample_image_path, "rb") as img_f:
        image_b64 = base64.b64encode(img_f.read()).decode("utf-8")

    # 画像の中心付近をポイントとして指定
    img = Image.open(sample_image_path)
    cx, cy = img.width // 2, img.height // 2

    response = endpoint.predict(
        instances=[
            {
                "image": image_b64,
                "input_points": [[cx, cy]],
                "input_labels": [1],
            }
        ]
    )

    # マスク画像をデコードして表示
    pred = response.predictions[0]
    mask_bytes = base64.b64decode(pred["mask_b64"])
    mask_image = Image.open(io.BytesIO(mask_bytes))
    mask_array = np.array(mask_image)

    # 元画像にマスクをオーバーレイ
    img_array = np.array(img)
    overlay = img_array.copy()
    overlay[mask_array > 0] = (
        overlay[mask_array > 0] * 0.5 + np.array([30, 144, 255]) * 0.5
    ).astype(np.uint8)
    result_image = Image.fromarray(overlay)

    mo.md(
        f"""
        ### 推論結果

        - ポイント座標: ({cx}, {cy})
        - IoU スコア: **{pred["iou_score"]:.3f}**
        """
    )
    mo.image(result_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 6: A/B テスト（トラフィック分割）

    ### なぜ A/B テストが必要か

    - オフライン指標が良くても、**本番データでの挙動は別**
    - 新モデルをいきなり100%に切り替えるのはリスクが高い
    - 少量のトラフィックで本番環境の動作を確認してから段階的に移行する

    ### 段階的なロールアウト

    ```
    新モデル20% ─→ 新モデル50% ─→ 新モデル100%
               測定             測定
    ```

    各段階で「エラー率」「レイテンシ」「セグメンテーション品質」を確認します。

    ### Vertex AI でのトラフィック分割

    Vertex AI のエンドポイントは**複数のモデルを同時にデプロイ**できます。
    """)
    return


@app.cell(hide_code=True)
def _(ENDPOINT_ID, REGION, USER, mo):
    mo.md(f"""
    ```bash
    # v2 モデルを GCS にアップロード（再学習モデルや別バリアント）
    gcloud storage cp -r sam-vit-base-v2/* gs://mixi-ml-handson-2026/models/{USER}/sam-model-v2/

    # v2 をモデルレジストリに登録
    gcloud ai models upload \\
        --region={REGION} \\
        --display-name=sam-server-v2-{USER} \\
        --container-image-uri=IMAGE_URI \\
        --container-env-vars=MODEL_GCS_URI=gs://mixi-ml-handson-2026/models/{USER}/sam-model-v2/ \\
        --container-health-route=/health \\
        --container-predict-route=/predict \\
        --container-ports=8080

    MODEL_V2_ID=___  # TODO: 上記コマンドの出力から

    # v1 を 80%、v2 を 20% にトラフィック分割
    gcloud ai endpoints deploy-model {ENDPOINT_ID} \\
        --region={REGION} \\
        --model=$MODEL_V2_ID \\
        --display-name=sam-model-v2 \\
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
    画像をアップロードしてクリックすると、その箇所のセグメンテーション結果を表示します。
    """)
    return


@app.cell
def _(ENDPOINT_ID, PROJECT_ID, REGION, base64):
    import numpy as _np
    from PIL import Image as _Image

    if ENDPOINT_ID == "___":
        raise RuntimeError("TODO: ENDPOINT_ID を設定してから実行してください。")

    import io as _io

    import gradio as gr
    from google.cloud import aiplatform as _aiplatform

    _aiplatform.init(project=PROJECT_ID, location=REGION)
    _endpoint = _aiplatform.Endpoint(ENDPOINT_ID)

    def segment(input_image: _Image.Image | None, evt: gr.SelectData):
        """画像上のクリック位置をもとにセグメントを実行する。"""
        if input_image is None:
            return None, "画像をアップロードしてください。"

        # クリック座標を取得
        click_x, click_y = evt.index

        # 画像を base64 エンコード
        buf = _io.BytesIO()
        input_image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Vertex AI エンドポイントに送信
        resp = _endpoint.predict(
            instances=[
                {
                    "image": img_b64,
                    "input_points": [[click_x, click_y]],
                    "input_labels": [1],
                }
            ]
        )

        pred = resp.predictions[0]
        iou_score = pred.get("iou_score", 0.0)

        # マスクをデコード
        mask_bytes = base64.b64decode(pred["mask_b64"])
        mask = _np.array(_Image.open(_io.BytesIO(mask_bytes)))

        # マスクを画像にオーバーレイ
        img_array = _np.array(input_image)
        overlay = img_array.copy()
        overlay[mask > 0] = (
            overlay[mask > 0] * 0.5 + _np.array([30, 144, 255]) * 0.5
        ).astype(_np.uint8)

        # クリック位置にマーカーを描画
        radius = max(5, min(img_array.shape[:2]) // 80)
        y_min = max(0, click_y - radius)
        y_max = min(img_array.shape[0], click_y + radius)
        x_min = max(0, click_x - radius)
        x_max = min(img_array.shape[1], click_x + radius)
        overlay[y_min:y_max, x_min:x_max] = [255, 0, 0]

        result_image = _Image.fromarray(overlay)
        perf = f"IoU Score: {iou_score:.3f}"

        return result_image, perf

    with gr.Blocks() as demo:
        gr.Markdown("## SAM セグメンテーションデモ (Vertex AI)")
        gr.Markdown(
            "画像をアップロードしてクリックすると、その箇所のセグメントが表示されます。"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="入力画像（クリックでポイント指定）",
                )
            with gr.Column():
                output_image = gr.Image(type="pil", label="セグメント結果")
                perf_text = gr.Textbox(label="結果", lines=2)

        input_image.select(
            segment,
            inputs=[input_image],
            outputs=[output_image, perf_text],
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
