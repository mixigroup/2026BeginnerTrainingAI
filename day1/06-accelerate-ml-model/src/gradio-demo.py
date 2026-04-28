"""SAM (Segment Anything Model) Gradio デモ。

画像をアップロードしてクリックした箇所をセグメントします。
Encoder の実装（PyTorch / ONNX / INT8 / Pruned）を切り替えて推論時間を比較できます。
"""

import os
import time

import gradio as gr
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# --- モデル設定 ---
MODEL_ID = "facebook/sam-vit-base"

# 使用する Encoder バリアント
MODEL_FILES: dict[str, str | None] = {
    "PyTorch": None,  # PyTorch の vision_encoder を直接使用
    "ONNX FP32": "sam-vit-b-encoder.onnx",
    "ONNX INT8": "sam-vit-b-encoder-quantized.onnx",
    "Pruned ONNX": "sam-vit-b-encoder-pruned.onnx",
}

# --- モデルのロード ---
print("SAM モデルをロード中...")
sam_model = SamModel.from_pretrained(MODEL_ID)
sam_model.eval()
processor = SamProcessor.from_pretrained(MODEL_ID)

# ONNX セッションのプリロード
ort_sessions: dict[str, ort.InferenceSession] = {}
available_models: list[str] = ["PyTorch"]

for name, path in MODEL_FILES.items():
    if path is None:
        continue
    if os.path.exists(path):
        ort_sessions[name] = ort.InferenceSession(path)
        available_models.append(name)

if not available_models:
    raise RuntimeError("利用可能なモデルが見つかりません。")

print(f"利用可能なモデル: {available_models}")


def run_encoder_pytorch(pixel_values: torch.Tensor) -> np.ndarray:
    """PyTorch で Image Encoder を実行する。"""
    with torch.no_grad():
        embeddings = sam_model.get_image_embeddings(pixel_values=pixel_values)
    return embeddings.numpy()


def run_encoder_onnx(
    pixel_values: torch.Tensor, session: ort.InferenceSession
) -> np.ndarray:
    """ONNX Runtime で Image Encoder を実行する。"""
    inputs = {"pixel_values": pixel_values.numpy()}
    (embeddings,) = session.run(None, inputs)
    return embeddings


def segment(
    input_image: Image.Image | None,
    model_name: str,
    evt: gr.SelectData,
) -> tuple[Image.Image | None, str]:
    """画像上のクリック位置をもとにセグメントを実行する。"""
    if input_image is None:
        return None, "画像をアップロードしてください。"

    # クリック座標を取得
    click_x, click_y = evt.index

    # 前処理
    input_points = [[[click_x, click_y]]]
    inputs = processor(
        images=input_image,
        input_points=input_points,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]

    # --- Encoder ---
    t_enc_start = time.perf_counter()

    if model_name == "PyTorch":
        image_embeddings = run_encoder_pytorch(pixel_values)
    else:
        session = ort_sessions.get(model_name)
        if session is None:
            return None, f"モデル '{model_name}' がロードされていません。"
        image_embeddings = run_encoder_onnx(pixel_values, session)

    t_enc_ms = (time.perf_counter() - t_enc_start) * 1000

    # --- Decoder (PyTorch) ---
    t_dec_start = time.perf_counter()

    image_embeddings_tensor = torch.tensor(image_embeddings)

    with torch.no_grad():
        outputs = sam_model(
            pixel_values=None,
            input_points=inputs["input_points"],
            image_embeddings=image_embeddings_tensor,
        )

    t_dec_ms = (time.perf_counter() - t_dec_start) * 1000
    t_total_ms = t_enc_ms + t_dec_ms

    # マスクの後処理
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )

    # IoU スコアが最も高いマスクを選択
    iou_scores = outputs.iou_scores[0, 0]
    best_idx = iou_scores.argmax().item()
    mask = masks[0][0, best_idx].numpy()

    # マスクを画像にオーバーレイ
    img_array = np.array(input_image)
    overlay = img_array.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.5 + np.array([30, 144, 255]) * 0.5
    ).astype(np.uint8)

    # クリック位置にマーカーを描画
    cy, cx = click_y, click_x
    radius = max(5, min(img_array.shape[:2]) // 80)
    y_min = max(0, cy - radius)
    y_max = min(img_array.shape[0], cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(img_array.shape[1], cx + radius)
    overlay[y_min:y_max, x_min:x_max] = [255, 0, 0]

    result_image = Image.fromarray(overlay)

    # パフォーマンス情報
    perf = (
        f"Encoder ({model_name}): {t_enc_ms:.1f} ms\n"
        f"Decoder (PyTorch)    : {t_dec_ms:.1f} ms\n"
        f"Total                : {t_total_ms:.1f} ms\n"
        f"IoU Score            : {iou_scores[best_idx]:.3f}"
    )

    return result_image, perf


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## SAM (Segment Anything) — クリックでセグメント")
    gr.Markdown(
        "画像をアップロードしてクリックすると、その箇所のセグメントが表示されます。"
    )

    model_radio = gr.Radio(
        choices=available_models,
        value=available_models[0],
        label="Encoder モデル",
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="入力画像（クリックでポイント指定）",
            )
        with gr.Column():
            output_image = gr.Image(type="pil", label="セグメント結果")
            perf_text = gr.Textbox(label="Performance", lines=4)

    input_image.select(
        segment,
        inputs=[input_image, model_radio],
        outputs=[output_image, perf_text],
    )

if __name__ == "__main__":
    demo.launch(share=False)
