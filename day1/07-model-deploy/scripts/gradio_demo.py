"""Gradio デモアプリ — Vertex AI エンドポイント経由で SAM セグメンテーションを実行する。

Usage:
    uv run python scripts/gradio_demo.py \
        --project <PROJECT_ID> \
        --region <REGION> \
        --endpoint <ENDPOINT_ID>
"""

from __future__ import annotations

import argparse
import base64
import io

import gradio as gr
import numpy as np
from google.cloud import aiplatform
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM Gradio Demo (Vertex AI)")
    parser.add_argument("--project", required=True, help="GCP プロジェクト ID")
    parser.add_argument(
        "--region", default="asia-northeast1", help="Vertex AI リージョン"
    )
    parser.add_argument("--endpoint", required=True, help="Vertex AI エンドポイント ID")
    parser.add_argument(
        "--share", action="store_true", help="Gradio share リンクを有効にする"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(args.endpoint)

    def segment(input_image: Image.Image | None, evt: gr.SelectData):
        """画像上のクリック位置をもとにセグメントを実行する。"""
        if input_image is None:
            return None, "画像をアップロードしてください。"

        input_image = input_image.convert("RGB")

        # クリック座標を取得
        click_x, click_y = evt.index

        # 画像を base64 エンコード
        buf = io.BytesIO()
        input_image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Vertex AI エンドポイントに送信
        resp = endpoint.predict(
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
        mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))

        # マスクを画像にオーバーレイ
        img_array = np.array(input_image)
        overlay = img_array.copy()
        overlay[mask > 0] = (
            overlay[mask > 0] * 0.5 + np.array([30, 144, 255]) * 0.5
        ).astype(np.uint8)

        # クリック位置にマーカーを描画
        radius = max(5, min(img_array.shape[:2]) // 80)
        y_min = max(0, click_y - radius)
        y_max = min(img_array.shape[0], click_y + radius)
        x_min = max(0, click_x - radius)
        x_max = min(img_array.shape[1], click_x + radius)
        overlay[y_min:y_max, x_min:x_max] = [255, 0, 0]

        result_image = Image.fromarray(overlay)
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

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
